from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np

from .decision import DecisionEngine
from .imaging import crop_region
from .models import DecisionStatus, FoulEvent, FoulKind, TeamFoulIndicator
from .recognition import RecognitionResult
from .scoring import InkColor, colored_ink_masks
from .template import FECABA_V1, NormalizedRect, TemplateSpec


class TeamFoulCellKind(str, Enum):
    EMPTY = "empty"
    X = "x"
    UNUSED = "unused"
    AMBIGUOUS = "ambiguous"


@dataclass(slots=True)
class TeamFoulCellObservation:
    kind: TeamFoulCellKind
    ink_color: InkColor
    confidence: float
    ink_ratio: float
    elongation: float


@dataclass(slots=True)
class HalfSeparatorObservation:
    first_half_slots: int | None
    confidence: float
    boundary_scores: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class TeamFoulGridSpec:
    group_left: float = 0.045
    group_width: float = 0.385
    group_right: float = 0.615
    row_top: float = 0.18
    row_height: float = 0.34
    row_bottom: float = 0.61


@dataclass(frozen=True, slots=True)
class TeamFoulGridDetection:
    left_boundaries: tuple[int, ...]
    right_boundaries: tuple[int, ...]
    row_bounds: tuple[tuple[int, int], ...]
    confidence: float

    @property
    def valid(self) -> bool:
        return (
            len(self.left_boundaries) == 5
            and len(self.right_boundaries) == 5
            and len(self.row_bounds) == 2
        )


@dataclass(frozen=True, slots=True)
class PlayerFoulGridSpec:
    header_fraction: float = 0.085
    roster_rows: int = 12
    slots: int = 5
    inner_margin: float = 0.12


TEAM_FOUL_GRID = TeamFoulGridSpec()
PLAYER_FOUL_GRID = PlayerFoulGridSpec()


class FoulSymbolRecognizer(Protocol):
    def recognize(self, image: np.ndarray) -> RecognitionResult: ...


FOUL_SYMBOLS = {
    "P": (FoulKind.PERSONAL, None),
    "P1": (FoulKind.PERSONAL, 1),
    "P2": (FoulKind.PERSONAL, 2),
    "P3": (FoulKind.PERSONAL, 3),
    "T": (FoulKind.TECHNICAL, None),
    "T1": (FoulKind.TECHNICAL, 1),
    "U": (FoulKind.UNSPORTSMANLIKE, None),
    "U1": (FoulKind.UNSPORTSMANLIKE, 1),
    "U2": (FoulKind.UNSPORTSMANLIKE, 2),
    "D": (FoulKind.DISQUALIFYING, None),
    "GD": (FoulKind.DISQUALIFYING, None),
}


def parse_foul_symbol(symbol: str | None) -> tuple[FoulKind, int | None]:
    if symbol is None:
        return FoulKind.UNKNOWN, None
    normalized = symbol.strip().upper().replace(" ", "")
    return FOUL_SYMBOLS.get(normalized, (FoulKind.UNKNOWN, None))


def classify_team_foul_cell(
    image: np.ndarray,
    *,
    min_ink_ratio: float = 0.01,
) -> TeamFoulCellObservation:
    red_mask, blue_mask = colored_ink_masks(image)
    mask = red_mask | blue_mask
    ink_ratio = float(mask.mean())
    color = _dominant_color(red_mask, blue_mask)
    if ink_ratio < min_ink_ratio:
        return TeamFoulCellObservation(
            TeamFoulCellKind.EMPTY,
            InkColor.UNKNOWN,
            1.0,
            ink_ratio,
            1.0,
        )

    yy, xx = np.nonzero(mask)
    if len(xx) < 5:
        return TeamFoulCellObservation(
            TeamFoulCellKind.AMBIGUOUS,
            color,
            0.0,
            ink_ratio,
            1.0,
        )

    points = np.column_stack((xx, yy)).astype(np.float64)
    centered = points - points.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(len(points) - 1, 1)
    eigenvalues = np.linalg.eigvalsh(covariance)
    elongation = float(
        np.sqrt(
            max(float(eigenvalues[-1]), 1e-6)
            / max(float(eigenvalues[0]), 1e-6)
        )
    )
    width_span = (xx.max() - xx.min() + 1) / max(mask.shape[1], 1)
    height_span = (yy.max() - yy.min() + 1) / max(mask.shape[0], 1)
    row_coverage = mask.mean(axis=1)
    horizontal_coverage = float(row_coverage.max())
    strong_horizontal_rows = int(np.count_nonzero(row_coverage >= 0.65))

    # As caixas nao utilizadas sao fechadas com dois tracos horizontais.
    # Em sumulas reais esses tracos podem ser grossos ou cobrir uma rasura/X
    # anterior, o que torna a nuvem de pixels quase isotropica e engana o PCA.
    # Uma cobertura horizontal longa em varias linhas e, nesse caso, evidencia
    # mais forte do que a elongacao global.
    if horizontal_coverage >= 0.75 and strong_horizontal_rows >= 2:
        confidence = min(1.0, 0.55 + horizontal_coverage * 0.45)
        return TeamFoulCellObservation(
            kind=TeamFoulCellKind.UNUSED,
            ink_color=color,
            confidence=round(float(confidence), 4),
            ink_ratio=round(ink_ratio, 6),
            elongation=round(elongation, 4),
        )

    x_score = (
        max(0.0, 1.0 - abs(elongation - 1.0) / 1.8) * 0.55
        + min(width_span / 0.65, 1.0) * 0.20
        + min(height_span / 0.65, 1.0) * 0.25
    )
    unused_score = (
        min(elongation / 4.0, 1.0) * 0.55
        + min(width_span / 0.70, 1.0) * 0.30
        + max(0.0, 1.0 - height_span / 0.65) * 0.15
    )

    if x_score >= 0.57 and x_score > unused_score + 0.04:
        kind = TeamFoulCellKind.X
        confidence = x_score
    elif unused_score >= 0.57 and unused_score > x_score:
        kind = TeamFoulCellKind.UNUSED
        confidence = unused_score
    else:
        kind = TeamFoulCellKind.AMBIGUOUS
        confidence = max(x_score, unused_score) * 0.75

    return TeamFoulCellObservation(
        kind=kind,
        ink_color=color,
        confidence=round(float(min(confidence, 1.0)), 4),
        ink_ratio=round(ink_ratio, 6),
        elongation=round(elongation, 4),
    )


def iter_team_foul_cells(
    grid: TeamFoulGridSpec = TEAM_FOUL_GRID,
) -> tuple[tuple[int, int, NormalizedRect], ...]:
    result: list[tuple[int, int, NormalizedRect]] = []
    layouts = {
        1: (grid.group_left, grid.row_top),
        2: (grid.group_right, grid.row_top),
        3: (grid.group_left, grid.row_bottom),
        4: (grid.group_right, grid.row_bottom),
    }
    cell_width = grid.group_width / 4.0
    for period, (x, y) in layouts.items():
        for slot in range(4):
            result.append(
                (
                    period,
                    slot + 1,
                    NormalizedRect(
                        x + slot * cell_width,
                        y,
                        cell_width,
                        grid.row_height,
                    ),
                )
            )
    return tuple(result)


def detect_team_foul_grid(image: np.ndarray) -> TeamFoulGridDetection:
    """Detecta as caixas de faltas de equipe pelas linhas impressas."""
    rgb = _as_rgb(image).astype(np.float32, copy=False)
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    dark = gray < 125
    height, width = dark.shape

    horizontal = _line_centers(dark.mean(axis=1), threshold=0.55)
    row_bounds = _fit_team_foul_rows(horizontal, height)
    vertical = _line_centers(dark.mean(axis=0), threshold=0.55)
    left_boundaries, right_boundaries = _fit_team_foul_columns(vertical, width)

    if not (
        len(left_boundaries) == 5
        and len(right_boundaries) == 5
        and len(row_bounds) == 2
    ):
        return TeamFoulGridDetection(tuple(), tuple(), tuple(), 0.0)

    covered_width = right_boundaries[-1] - left_boundaries[0]
    covered_height = row_bounds[-1][1] - row_bounds[0][0]
    confidence = min(
        1.0,
        0.5 * covered_width / max(width, 1)
        + 0.5 * covered_height / max(height, 1),
    )
    return TeamFoulGridDetection(
        left_boundaries=tuple(left_boundaries),
        right_boundaries=tuple(right_boundaries),
        row_bounds=tuple(row_bounds),
        confidence=round(float(confidence), 4),
    )


def _as_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.repeat(image[..., None], 3, axis=2)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("Imagem deve ter formato HxW, HxWx3 ou HxWx4")
    return image[..., :3]


def _line_centers(values: np.ndarray, *, threshold: float) -> list[int]:
    indexes = np.flatnonzero(values >= threshold)
    if indexes.size == 0:
        return []
    centers: list[int] = []
    start = previous = int(indexes[0])
    for raw_index in indexes[1:]:
        index = int(raw_index)
        if index == previous + 1:
            previous = index
            continue
        centers.append(round((start + previous) / 2))
        start = previous = index
    centers.append(round((start + previous) / 2))
    return centers


def _fit_team_foul_rows(
    lines: list[int],
    height: int,
) -> list[tuple[int, int]]:
    if len(lines) < 3:
        return []
    for index in range(len(lines) - 2):
        first_height = lines[index + 1] - lines[index]
        row_gap = lines[index + 2] - lines[index + 1]
        if not (0.24 * height <= first_height <= 0.40 * height):
            continue
        if not (0.04 * height <= row_gap <= 0.18 * height):
            continue
        first = (lines[index], lines[index + 1])
        second_top = lines[index + 2]
        expected_bottom = second_top + first_height
        second_bottom = min(height - 1, expected_bottom)
        if index + 3 < len(lines):
            candidate = lines[index + 3]
            if abs(candidate - expected_bottom) <= 0.10 * height:
                second_bottom = candidate
        if second_bottom - second_top < 0.20 * height:
            continue
        return [first, (second_top, second_bottom)]
    return []


def _fit_team_foul_columns(
    lines: list[int],
    width: int,
) -> tuple[list[int], list[int]]:
    if len(lines) < 8:
        return [], []
    gaps = np.diff(lines)
    split = int(np.argmax(gaps)) + 1
    if gaps[split - 1] < 0.14 * width:
        return [], []
    left = _fit_five_boundaries(lines[:split], width)
    right = _fit_five_boundaries(lines[split:], width)
    if len(left) != 5 or len(right) != 5:
        return [], []
    return left, right


def _fit_five_boundaries(lines: list[int], width: int) -> list[int]:
    if len(lines) == 5:
        return lines
    if len(lines) == 4:
        spacing = int(round(float(np.median(np.diff(lines)))))
        if spacing <= 0:
            return []
        return [*lines, min(width, lines[-1] + spacing)]
    return []


def _iter_detected_team_foul_crops(
    block: np.ndarray,
    detection: TeamFoulGridDetection,
) -> tuple[tuple[int, np.ndarray], ...]:
    groups = {
        1: (detection.left_boundaries, detection.row_bounds[0]),
        2: (detection.right_boundaries, detection.row_bounds[0]),
        3: (detection.left_boundaries, detection.row_bounds[1]),
        4: (detection.right_boundaries, detection.row_bounds[1]),
    }
    crops: list[tuple[int, np.ndarray]] = []
    for period, (boundaries, (top, bottom)) in groups.items():
        margin_y = max(1, round((bottom - top) * 0.08))
        for slot in range(4):
            left = boundaries[slot]
            right = min(block.shape[1], boundaries[slot + 1])
            margin_x = max(1, round((right - left) * 0.08))
            crops.append(
                (
                    period,
                    block[
                        top + margin_y : max(top + margin_y + 1, bottom - margin_y),
                        left + margin_x : max(left + margin_x + 1, right - margin_x),
                    ],
                )
            )
    return tuple(crops)


def extract_team_foul_indicators(
    normalized_image: np.ndarray,
    *,
    team: str,
    template: TemplateSpec = FECABA_V1,
    decision_engine: DecisionEngine | None = None,
    decision_threshold: float = 0.85,
) -> list[TeamFoulIndicator]:
    side = team.upper()
    if side not in {"A", "B"}:
        raise ValueError("team deve ser A ou B")

    region_name = f"team_{side.lower()}_team_fouls"
    block = crop_region(normalized_image, template.region(region_name))
    observations: dict[int, list[TeamFoulCellObservation]] = {
        period: [] for period in range(1, 5)
    }
    detection = detect_team_foul_grid(block)
    if detection.valid:
        for period, cell in _iter_detected_team_foul_crops(block, detection):
            observations[period].append(classify_team_foul_cell(cell))
    else:
        for period, _, rect in iter_team_foul_cells():
            observations[period].append(
                classify_team_foul_cell(crop_region(block, rect))
            )

    if decision_engine is not None:
        for period, cells in observations.items():
            local_kinds = [cell.kind.value for cell in cells]
            for index, cell in enumerate(cells):
                if cell.kind is not TeamFoulCellKind.AMBIGUOUS:
                    continue
                try:
                    answer = decision_engine.classify_team_foul(
                        period=period,
                        slot=index + 1,
                        evidence={
                            "local_kind": cell.kind.value,
                            "local_confidence": cell.confidence,
                            "ink_color": cell.ink_color.value,
                            "ink_ratio": cell.ink_ratio,
                            "elongation": cell.elongation,
                        },
                        context={"period_cell_kinds": local_kinds},
                    )
                except RuntimeError:
                    continue
                if answer.confidence < decision_threshold:
                    continue
                resolved_kind = {
                    "x": TeamFoulCellKind.X,
                    "unused": TeamFoulCellKind.UNUSED,
                }.get(answer.choice)
                if resolved_kind is None:
                    continue
                cells[index] = TeamFoulCellObservation(
                    kind=resolved_kind,
                    ink_color=cell.ink_color,
                    confidence=round(float(answer.confidence), 4),
                    ink_ratio=cell.ink_ratio,
                    elongation=cell.elongation,
                )

    indicators: list[TeamFoulIndicator] = []
    for period in range(1, 5):
        cells = observations[period]
        x_count, needs_review = _reconcile_team_foul_cells(cells)
        confidence = min((cell.confidence for cell in cells), default=0.0)
        indicators.append(
            TeamFoulIndicator(
                period=period,
                x_count=x_count,
                confidence=round(float(confidence), 4),
                status=(
                    DecisionStatus.REVIEW
                    if needs_review
                    else DecisionStatus.ACCEPTED
                ),
            )
        )
    return indicators


def _reconcile_team_foul_cells(
    cells: list[TeamFoulCellObservation],
) -> tuple[int, bool]:
    """Aplica a semantica sequencial das quatro caixas de falta coletiva.

    Uma marca na caixa N significa que a equipe chegou pelo menos a N faltas.
    Assim, uma leitura fraca/ausente antes de um X posterior nao reduz a
    contagem; o periodo e marcado para revisao porque pode haver rasura ou
    erro de preenchimento. Marcas ambiguas apos o ultimo X tambem exigem
    revisao, pois podem representar a proxima falta ou o fechamento da linha.
    """
    x_slots = [
        index
        for index, cell in enumerate(cells)
        if cell.kind is TeamFoulCellKind.X
    ]
    if not x_slots:
        return (
            0,
            any(cell.kind is TeamFoulCellKind.AMBIGUOUS for cell in cells),
        )

    last_x = max(x_slots)
    prefix_has_gap = any(
        cell.kind is not TeamFoulCellKind.X for cell in cells[: last_x + 1]
    )
    suffix_is_ambiguous = any(
        cell.kind is TeamFoulCellKind.AMBIGUOUS for cell in cells[last_x + 1 :]
    )
    return last_x + 1, prefix_has_gap or suffix_is_ambiguous


def detect_half_separator(
    row_image: np.ndarray,
    *,
    slots: int = 5,
    band_fraction: float = 0.055,
    threshold: float = 0.15,
) -> HalfSeparatorObservation:
    _, blue_mask = colored_ink_masks(row_image)
    height, width = blue_mask.shape
    scores: list[float] = []

    for boundary in range(slots + 1):
        center = boundary / slots * width
        half_band = max(1, round(width * band_fraction / 2.0))
        left = max(0, round(center) - half_band)
        right = min(width, round(center) + half_band + 1)
        band = blue_mask[:, left:right]
        if band.size == 0:
            scores.append(0.0)
            continue
        vertical_coverage = float(np.any(band, axis=1).mean())
        density = float(band.mean())
        scores.append(vertical_coverage * 0.75 + density * 0.25)

    best = int(np.argmax(scores))
    best_score = scores[best]
    ordered = sorted(scores, reverse=True)
    runner_up = ordered[1] if len(ordered) > 1 else 0.0
    separation = max(0.0, best_score - runner_up)
    confidence = min(1.0, best_score * 1.2 + separation)
    return HalfSeparatorObservation(
        first_half_slots=best if best_score >= threshold else None,
        confidence=round(float(confidence), 4),
        boundary_scores=tuple(round(float(score), 4) for score in scores),
    )


def analyze_player_foul_row(
    row_image: np.ndarray,
    *,
    team: str,
    jersey: int,
    grid: PlayerFoulGridSpec = PLAYER_FOUL_GRID,
    recognizer: FoulSymbolRecognizer | None = None,
) -> list[FoulEvent]:
    separator = detect_half_separator(row_image, slots=grid.slots)
    height, width = row_image.shape[:2]
    slot_width = width / grid.slots
    events: list[FoulEvent] = []

    for index in range(grid.slots):
        left = round(index * slot_width)
        right = round((index + 1) * slot_width)
        margin_x = round((right - left) * grid.inner_margin)
        margin_y = round(height * grid.inner_margin)
        cell = row_image[
            margin_y : max(margin_y + 1, height - margin_y),
            left + margin_x : max(left + margin_x + 1, right - margin_x),
        ]
        red_mask, blue_mask = colored_ink_masks(cell)
        mask = red_mask | blue_mask
        if float(mask.mean()) < 0.006:
            continue

        color = _dominant_color(red_mask, blue_mask)
        period = _period_for_foul(
            slot=index + 1,
            first_half_slots=separator.first_half_slots,
            color=color,
        )
        raw_symbol: str | None = None
        kind = FoulKind.UNKNOWN
        free_throws: int | None = None
        symbol_confidence: float | None = None
        symbol_status = DecisionStatus.REVIEW
        if recognizer is not None:
            recognition = recognizer.recognize(cell)
            raw_symbol = recognition.value
            symbol_confidence = recognition.confidence
            symbol_status = recognition.status
            kind, free_throws = parse_foul_symbol(raw_symbol)

        period_confidence = separator.confidence if period is not None else None
        confidence_values = [
            value
            for value in (period_confidence, symbol_confidence)
            if value is not None
        ]
        confidence = min(confidence_values) if confidence_values else None
        accepted = (
            period is not None
            and kind is not FoulKind.UNKNOWN
            and symbol_status is DecisionStatus.ACCEPTED
        )
        events.append(
            FoulEvent(
                team=team.upper(),
                jersey=jersey,
                slot=index + 1,
                period=period,
                kind=kind,
                free_throws=free_throws,
                raw_symbol=raw_symbol,
                ink_color=color.value,
                confidence=confidence,
                status=(
                    DecisionStatus.ACCEPTED
                    if accepted
                    else DecisionStatus.REVIEW
                ),
            )
        )
    return events


def extract_player_fouls(
    normalized_image: np.ndarray,
    *,
    team: str,
    jerseys: list[int],
    template: TemplateSpec = FECABA_V1,
    grid: PlayerFoulGridSpec = PLAYER_FOUL_GRID,
    recognizer: FoulSymbolRecognizer | None = None,
) -> list[FoulEvent]:
    side = team.upper()
    if side not in {"A", "B"}:
        raise ValueError("team deve ser A ou B")
    if len(jerseys) > grid.roster_rows:
        raise ValueError(
            f"O template suporta no maximo {grid.roster_rows} linhas de jogadores"
        )

    region_name = f"team_{side.lower()}_player_fouls"
    block = crop_region(normalized_image, template.region(region_name))
    data_top = round(block.shape[0] * grid.header_fraction)
    data = block[data_top:]
    row_height = data.shape[0] / grid.roster_rows
    events: list[FoulEvent] = []

    for index, jersey in enumerate(jerseys):
        top = round(index * row_height)
        bottom = round((index + 1) * row_height)
        row = data[top:bottom]
        events.extend(
            analyze_player_foul_row(
                row,
                team=side,
                jersey=jersey,
                grid=grid,
                recognizer=recognizer,
            )
        )
    return events


def _period_for_foul(
    *,
    slot: int,
    first_half_slots: int | None,
    color: InkColor,
) -> int | None:
    if first_half_slots is None or color is InkColor.UNKNOWN:
        return None
    first_half = slot <= first_half_slots
    if first_half:
        return 1 if color is InkColor.RED else 2
    return 3 if color is InkColor.RED else 4


def _dominant_color(red_mask: np.ndarray, blue_mask: np.ndarray) -> InkColor:
    red_count = int(red_mask.sum())
    blue_count = int(blue_mask.sum())
    total = red_count + blue_count
    if total == 0:
        return InkColor.UNKNOWN
    if red_count / total >= 0.62:
        return InkColor.RED
    if blue_count / total >= 0.62:
        return InkColor.BLUE
    return InkColor.UNKNOWN
