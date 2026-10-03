from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw

from .template import NormalizedRect, TemplateSpec


class VisionDependencyError(RuntimeError):
    pass


@dataclass(slots=True)
class NormalizedDocument:
    image: np.ndarray
    corners: np.ndarray | None
    used_perspective_warp: bool
    method: str = "resize"


def load_document(path: str | Path, *, dpi: int = 250) -> np.ndarray:
    """Carrega a primeira pagina de um PDF ou uma imagem RGB."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)

    if source.suffix.lower() == ".pdf":
        return _render_pdf(source, dpi=dpi)

    with Image.open(source) as image:
        return np.asarray(image.convert("RGB"))


def _render_pdf(path: Path, *, dpi: int) -> np.ndarray:
    try:
        import pymupdf  # type: ignore
    except ImportError as exc:
        pdftoppm = shutil.which("pdftoppm")
        if pdftoppm is None:
            raise VisionDependencyError(
                "PDF requer PyMuPDF ou pdftoppm/Poppler. "
                "Instale com: pip install -e .[vision]"
            ) from exc

        with tempfile.TemporaryDirectory(prefix="sumula-reader-") as tmp:
            output_base = Path(tmp) / "page"
            command = [
                pdftoppm,
                "-f",
                "1",
                "-singlefile",
                "-r",
                str(dpi),
                "-png",
                str(path),
                str(output_base),
            ]
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "Failed to render PDF with pdftoppm: "
                    + completed.stderr.strip()
                )
            with Image.open(output_base.with_suffix(".png")) as image:
                return np.asarray(image.convert("RGB"))

    scale = dpi / 72.0
    with pymupdf.open(path) as document:
        if document.page_count < 1:
            raise ValueError("The PDF has no pages")
        page = document.load_page(0)
        pix = page.get_pixmap(
            matrix=pymupdf.Matrix(scale, scale),
            alpha=False,
        )
        array = np.frombuffer(pix.samples, dtype=np.uint8)
        return array.reshape(pix.height, pix.width, pix.n)[..., :3].copy()


def normalize_document(
    image: np.ndarray,
    template: TemplateSpec,
    *,
    strict: bool = False,
) -> NormalizedDocument:
    """Alinha a folha ao tamanho canonico do template.

    O caminho preferencial procura o maior quadrilatero que represente a
    outer scoresheet border and applies a homography. In non-strict mode, if
    the border cannot be found, it resizes the image and records that no warp
    occurred; this is useful while calibrating in debug mode.
    """
    rgb = _ensure_rgb(image)
    try:
        cv2 = _cv2()
    except VisionDependencyError:
        corners = _find_document_corners_numpy(rgb)
        if corners is not None:
            warped = _warp_with_pillow(
                rgb,
                corners,
                template.canonical_width,
                template.canonical_height,
            )
            return NormalizedDocument(
                image=warped,
                corners=corners,
                used_perspective_warp=True,
                method="pillow_quad",
            )
        if strict:
            raise ValueError("Could not locate the scoresheet border")
        return _resize_fallback(rgb, template)

    corners = _find_document_corners(rgb, cv2)
    if corners is None:
        corners = _find_document_corners_numpy(rgb)
        if corners is not None:
            warped = _warp_with_pillow(
                rgb,
                corners,
                template.canonical_width,
                template.canonical_height,
            )
            return NormalizedDocument(
                image=warped,
                corners=corners,
                used_perspective_warp=True,
                method="pillow_quad",
            )
        if strict:
            raise ValueError("Could not locate the scoresheet border")
        resized = cv2.resize(
            rgb,
            (template.canonical_width, template.canonical_height),
            interpolation=cv2.INTER_AREA,
        )
        return NormalizedDocument(
            image=resized,
            corners=None,
            used_perspective_warp=False,
            method="resize",
        )

    target = np.array(
        [
            [0, 0],
            [template.canonical_width - 1, 0],
            [template.canonical_width - 1, template.canonical_height - 1],
            [0, template.canonical_height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), target)
    warped = cv2.warpPerspective(
        rgb,
        matrix,
        (template.canonical_width, template.canonical_height),
        borderValue=(255, 255, 255),
    )
    return NormalizedDocument(
        image=warped,
        corners=corners,
        used_perspective_warp=True,
        method="opencv_homography",
    )


def crop_region(image: np.ndarray, rect: NormalizedRect) -> np.ndarray:
    rgb = _ensure_rgb(image)
    height, width = rgb.shape[:2]
    left, top, right, bottom = rect.pixels(width, height)
    return rgb[top:bottom, left:right].copy()


def extract_regions(
    image: np.ndarray,
    template: TemplateSpec,
    names: Iterable[str] | None = None,
) -> dict[str, np.ndarray]:
    selected = tuple(names) if names is not None else tuple(template.regions)
    return {
        name: crop_region(image, template.region(name))
        for name in selected
    }


def save_debug_bundle(
    document: NormalizedDocument,
    template: TemplateSpec,
    output_dir: str | Path,
) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    normalized_path = output / "normalized.png"
    Image.fromarray(document.image).save(normalized_path)
    written.append(normalized_path)

    overlay = Image.fromarray(document.image.copy())
    drawer = ImageDraw.Draw(overlay)
    for name, rect in template.regions.items():
        left, top, right, bottom = rect.pixels(*overlay.size)
        drawer.rectangle((left, top, right, bottom), outline=(255, 0, 0), width=4)
        drawer.text((left + 4, top + 4), name, fill=(255, 0, 0))
    overlay_path = output / "regions-overlay.png"
    overlay.save(overlay_path)
    written.append(overlay_path)

    crops_dir = output / "crops"
    crops_dir.mkdir(exist_ok=True)
    for name, crop in extract_regions(document.image, template).items():
        path = crops_dir / f"{name}.png"
        Image.fromarray(crop).save(path)
        written.append(path)

    return written


def _resize_fallback(image: np.ndarray, template: TemplateSpec) -> NormalizedDocument:
    resized = Image.fromarray(image).resize(
        (template.canonical_width, template.canonical_height),
        Image.Resampling.LANCZOS,
    )
    return NormalizedDocument(
        image=np.asarray(resized),
        corners=None,
        used_perspective_warp=False,
        method="resize",
    )


def _ensure_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("The image must have HxWx3 shape")
    return array[..., :3].astype(np.uint8, copy=False)


def _cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise VisionDependencyError(
            "OpenCV is not installed. Install it with: pip install -e .[vision]"
        ) from exc
    return cv2


def _find_document_corners(image: np.ndarray, cv2) -> np.ndarray | None:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    min_area = image.shape[0] * image.shape[1] * 0.35
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < min_area:
            break
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(polygon) == 4:
            return order_corners(polygon.reshape(4, 2).astype(np.float32))
    return None


def _find_document_corners_numpy(image: np.ndarray) -> np.ndarray | None:
    """Locate the four outer borders without depending on OpenCV.

    A folha FECABA possui uma moldura escura longa. Para cada faixa esperada
    (esquerda, direita, topo e base), coletamos a posicao do primeiro pixel
    escuro e ajustamos uma reta robusta. Isso tolera pequenas inclinacoes e
    perspectiva de scans de celular/CamScanner.
    """
    rgb = _ensure_rgb(image).astype(np.float32)
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    dark = gray < 145
    height, width = dark.shape

    left = _fit_vertical_border(dark, 0.01, 0.18, choose="first")
    right = _fit_vertical_border(dark, 0.80, 0.99, choose="last")
    top = _fit_horizontal_border(dark, 0.01, 0.18, choose="first")
    bottom = _fit_horizontal_border(dark, 0.80, 0.99, choose="last")
    if any(line is None for line in (left, right, top, bottom)):
        return None

    assert left is not None and right is not None
    assert top is not None and bottom is not None
    tl = _line_intersection(left, top)
    tr = _line_intersection(right, top)
    br = _line_intersection(right, bottom)
    bl = _line_intersection(left, bottom)
    corners = np.array([tl, tr, br, bl], dtype=np.float32)

    if not np.isfinite(corners).all():
        return None
    polygon_width = max(
        np.linalg.norm(tr - tl),
        np.linalg.norm(br - bl),
    )
    polygon_height = max(
        np.linalg.norm(bl - tl),
        np.linalg.norm(br - tr),
    )
    if polygon_width < width * 0.60 or polygon_height < height * 0.60:
        return None
    return corners


def _fit_vertical_border(
    dark: np.ndarray,
    start_ratio: float,
    end_ratio: float,
    *,
    choose: str,
) -> tuple[float, float, float] | None:
    height, width = dark.shape
    start = max(0, round(width * start_ratio))
    end = min(width, round(width * end_ratio))
    ys = np.arange(round(height * 0.06), round(height * 0.94), 3)
    points: list[tuple[float, float]] = []
    for y in ys:
        xs = np.flatnonzero(dark[y, start:end])
        if xs.size:
            x = xs[0] if choose == "first" else xs[-1]
            points.append((float(y), float(start + x)))
    return _robust_line(points, vertical=True)


def _fit_horizontal_border(
    dark: np.ndarray,
    start_ratio: float,
    end_ratio: float,
    *,
    choose: str,
) -> tuple[float, float, float] | None:
    height, width = dark.shape
    start = max(0, round(height * start_ratio))
    end = min(height, round(height * end_ratio))
    xs = np.arange(round(width * 0.06), round(width * 0.94), 3)
    points: list[tuple[float, float]] = []
    for x in xs:
        ys = np.flatnonzero(dark[start:end, x])
        if ys.size:
            y = ys[0] if choose == "first" else ys[-1]
            points.append((float(x), float(start + y)))
    return _robust_line(points, vertical=False)


def _robust_line(
    points: list[tuple[float, float]],
    *,
    vertical: bool,
) -> tuple[float, float, float] | None:
    if len(points) < 20:
        return None
    values = np.asarray(points, dtype=np.float64)
    independent = values[:, 0]
    dependent = values[:, 1]

    median = np.median(dependent)
    mad = np.median(np.abs(dependent - median))
    tolerance = max(6.0, mad * 3.5)
    keep = np.abs(dependent - median) <= tolerance
    if keep.sum() < 20:
        return None

    slope, intercept = np.polyfit(independent[keep], dependent[keep], 1)
    # Ax + By + C = 0
    if vertical:
        # x = slope*y + intercept
        return (1.0, -float(slope), -float(intercept))
    # y = slope*x + intercept
    return (-float(slope), 1.0, -float(intercept))


def _line_intersection(
    line_a: tuple[float, float, float],
    line_b: tuple[float, float, float],
) -> np.ndarray:
    a1, b1, c1 = line_a
    a2, b2, c2 = line_b
    determinant = a1 * b2 - a2 * b1
    if abs(determinant) < 1e-9:
        return np.array([np.nan, np.nan], dtype=np.float32)
    x = (b1 * c2 - b2 * c1) / determinant
    y = (c1 * a2 - c2 * a1) / determinant
    return np.array([x, y], dtype=np.float32)


def _warp_with_pillow(
    image: np.ndarray,
    corners: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    tl, tr, br, bl = order_corners(corners)
    quad = (
        float(tl[0]),
        float(tl[1]),
        float(bl[0]),
        float(bl[1]),
        float(br[0]),
        float(br[1]),
        float(tr[0]),
        float(tr[1]),
    )
    source = Image.fromarray(_ensure_rgb(image))
    warped = source.transform(
        (width, height),
        Image.Transform.QUAD,
        quad,
        resample=Image.Resampling.BICUBIC,
        fillcolor=(255, 255, 255),
    )
    return np.asarray(warped)


def order_corners(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError("Sao esperados quatro pontos 2D")

    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    return np.array(
        [
            pts[np.argmin(sums)],
            pts[np.argmin(diffs)],
            pts[np.argmax(sums)],
            pts[np.argmax(diffs)],
        ],
        dtype=np.float32,
    )
