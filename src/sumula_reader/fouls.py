from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np

from .decision import DecisionEngine
from .imaging import crop_region
from .models import (
    DecisionStatus,
    FoulEvent,
    FoulKind,
    FoulTerminal,
    FoulTerminalKind,
    TeamFoulIndicator,
)
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
    team_a_header_fraction: float = 0.26
    team_b_footer_fraction: float = 0.08
    roster_rows: int = 12
    slots: int = 5
    inner_margin: float = 0.12
    min_ink_ratio: float = 0.015


@dataclass(frozen=True, slots=True)
class PlayerFoulGridDetection:
    x_lines: tuple[int, ...]
    y_lines: tuple[int, ...]
    confidence: float

    @property
    def valid(self) -> bool:
        return len(self.x_lines) == 6 and len(self.y_lines) == 13


@dataclass(frozen=True, slots=True)
class FoulTerminalObservation:
    kind: FoulTerminalKind
    ink_color: InkColor
    confidence: float


@dataclass(frozen=True, slots=True)
class ParsedFoulSymbol:
    kind: FoulKind
    free_throws: int | None
    normalized: str | None
    cancelled_penalty: bool
    counts_as_team_foul: bool
    fighting: bool


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
    "U3": (FoulKind.UNSPORTSMANLIKE, 3),
    "D": (FoulKind.DISQUALIFYING, None),
    "D2": (FoulKind.DISQUALIFYING, 2),
    "GD": (FoulKind.DISQUALIFYING, None),
    "PC": (FoulKind.PERSONAL, None),
    "TC": (FoulKind.TECHNICAL, None),
    "UC": (FoulKind.UNSPORTSMANLIKE, None),
    "DC": (FoulKind.DISQUALIFYING, None),
}


def parse_foul_symbol(symbol: str | None) -> tuple[FoulKind, int | None]:
    parsed = parse_foul_symbol_details(symbol)
    return parsed.kind, parsed.free_throws


def parse_foul_symbol_details(symbol: str | None) -> ParsedFoulSymbol:
    if symbol is None:
        return ParsedFoulSymbol(FoulKind.UNKNOWN, None, None, False, False, False)
    normalized = symbol.strip().upper().replace(" ", "")
    kind, free_throws = FOUL_SYMBOLS.get(
        normalized,
        (FoulKind.UNKNOWN, None),
    )
    cancelled = normalized in {"PC", "TC", "UC", "DC"}
    fighting = normalized == "GD"
    counts_as_team_foul = (
        kind in {
            FoulKind.PERSONAL,
            FoulKind.TECHNICAL,
            FoulKind.UNSPORTSMANLIKE,
            FoulKind.DISQUALIFYING,
        }
        and not fighting
        and not cancelled
    )
    return ParsedFoulSymbol(
        kind=kind,
        free_throws=free_throws,
        normalized=normalized,
        cancelled_penalty=cancelled,
        counts_as_team_foul=counts_as_team_foul,
        fighting=fighting,
    )


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


def _fit_player_foul_columns(lines: list[int], width: int) -> list[int]:
    if len(lines) < 4:
        return []
    candidates = sorted(set(lines))
    if len(candidates) == 4:
        candidates = [0, *candidates, width - 1]
    elif len(candidates) == 5:
        if candidates[0] > width * 0.10:
            candidates = [0, *candidates]
        elif candidates[-1] < width * 0.90:
            candidates = [*candidates, width - 1]
    if len(candidates) < 6:
        return []

    best: tuple[float, list[int]] | None = None
    for start_index in range(len(candidates)):
        for end_index in range(start_index + 5, len(candidates)):
            start = candidates[start_index]
            end = candidates[end_index]
            spacing = (end - start) / 5.0
            if spacing <= 0:
                continue
            chosen: list[int] = []
            error = 0.0
            for slot in range(6):
                predicted = start + slot * spacing
                nearest = min(candidates, key=lambda value: abs(value - predicted))
                delta = abs(nearest - predicted)
                if delta > max(3.0, width * 0.035):
                    chosen = []
                    break
                chosen.append(nearest)
                error += delta
            if len(chosen) != 6 or len(set(chosen)) != 6:
                continue
            edge_penalty = abs(chosen[0]) + abs((width - 1) - chosen[-1])
            score = error + edge_penalty * 0.15
            if best is None or score < best[0]:
                best = (score, chosen)
    return best[1] if best is not None else []


def _fit_player_foul_rows(
    lines: list[int],
    *,
    expected_top: int,
    expected_bottom: int,
    rows: int,
) -> list[int]:
    if len(lines) < rows - 1:
        return []
    candidates = sorted(set(lines))
    span = expected_bottom - expected_top
    tolerance = max(5.0, span / max(rows, 1) * 0.45)
    if not any(abs(line - expected_top) <= tolerance for line in candidates):
        candidates.append(expected_top)
    if not any(abs(line - expected_bottom) <= tolerance for line in candidates):
        candidates.append(expected_bottom)
    candidates.sort()
    step = span / rows
    fitted: list[int] = []
    for index in range(rows + 1):
        predicted = expected_top + index * step
        nearest = min(candidates, key=lambda value: abs(value - predicted))
        if abs(nearest - predicted) > tolerance:
            return []
        fitted.append(nearest)
    if len(set(fitted)) != rows + 1:
        return []
    return fitted


def detect_player_foul_grid(
    block: np.ndarray,
    *,
    side: str,
    grid: PlayerFoulGridSpec = PLAYER_FOUL_GRID,
) -> PlayerFoulGridDetection:
    """Detecta as 5 colunas e as 12 linhas reais de faltas de jogadores."""
    normalized_side = side.upper()
    if normalized_side not in {"A", "B"}:
        raise ValueError("side deve ser A ou B")
    rgb = _as_rgb(block).astype(np.float32, copy=False)
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    dark = gray < 145
    height, width = dark.shape
    expected_top = _player_foul_data_top(block, side=normalized_side, grid=grid)
    expected_bottom = _player_foul_data_bottom(block, side=normalized_side, grid=grid)
    if expected_bottom <= expected_top:
        return PlayerFoulGridDetection(tuple(), tuple(), 0.0)

    data_dark = dark[expected_top:expected_bottom]
    x_lines: list[int] = []
    for threshold in (0.72, 0.62, 0.52, 0.42, 0.32):
        candidates = _line_centers(data_dark.mean(axis=0), threshold=threshold)
        fitted = _fit_player_foul_columns(candidates, width)
        if len(fitted) == 6:
            x_lines = fitted
            break
    if len(x_lines) != 6:
        return PlayerFoulGridDetection(tuple(), tuple(), 0.0)

    interior_left = max(0, x_lines[0] + 1)
    interior_right = min(width, x_lines[-1])
    if interior_right <= interior_left:
        return PlayerFoulGridDetection(tuple(), tuple(), 0.0)
    profile = dark[:, interior_left:interior_right].mean(axis=1)
    y_lines: list[int] = []
    for threshold in (0.58, 0.50, 0.42, 0.34):
        candidates = _line_centers(profile, threshold=threshold)
        fitted = _fit_player_foul_rows(
            candidates,
            expected_top=expected_top,
            expected_bottom=expected_bottom,
            rows=grid.roster_rows,
        )
        if len(fitted) == grid.roster_rows + 1:
            y_lines = fitted
            break
    if len(y_lines) != grid.roster_rows + 1:
        return PlayerFoulGridDetection(tuple(x_lines), tuple(), 0.0)

    x_span = x_lines[-1] - x_lines[0]
    y_span = y_lines[-1] - y_lines[0]
    confidence = min(
        1.0,
        0.5 * x_span / max(width, 1)
        + 0.5 * y_span / max(expected_bottom - expected_top, 1),
    )
    return PlayerFoulGridDetection(
        x_lines=tuple(x_lines),
        y_lines=tuple(y_lines),
        confidence=round(float(confidence), 4),
    )


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


def classify_foul_terminal(
    image: np.ndarray,
    *,
    min_ink_ratio: float = 0.012,
) -> FoulTerminalObservation | None:
    if image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
        return None
    red_mask, blue_mask = colored_ink_masks(image)
    mask = red_mask | blue_mask
    ink_ratio = float(mask.mean())
    if ink_ratio < min_ink_ratio:
        return None
    color = _dominant_color(red_mask, blue_mask)
    row_coverage = mask.mean(axis=1)
    column_coverage = mask.mean(axis=0)
    horizontal = float(row_coverage.max())
    vertical = float(column_coverage.max())
    if horizontal >= 0.72:
        return FoulTerminalObservation(
            FoulTerminalKind.CLOSURE_STROKE,
            color,
            round(min(1.0, 0.55 + horizontal * 0.45), 4),
        )
    height = mask.shape[0]
    top = float(mask[: max(1, height // 3)].mean(axis=1).max())
    middle_start = max(0, height // 3)
    middle_end = max(middle_start + 1, 2 * height // 3)
    top_rows = mask[: max(1, height // 3)].mean(axis=1)
    middle_rows = mask[middle_start:middle_end].mean(axis=1)
    middle = float(middle_rows.max())
    strong_top_rows = int(np.count_nonzero(top_rows >= 0.52))
    strong_middle_rows = int(np.count_nonzero(middle_rows >= 0.42))
    top_run = max(
        (_longest_true_run(row) / max(mask.shape[1], 1) for row in mask[: max(1, height // 3)]),
        default=0.0,
    )
    middle_run = max(
        (_longest_true_run(row) / max(mask.shape[1], 1) for row in mask[middle_start:middle_end]),
        default=0.0,
    )
    f_score = min(vertical / 0.60, 1.0) * 0.45 + min(top / 0.50, 1.0) * 0.30 + min(middle / 0.38, 1.0) * 0.25
    yy, xx = np.nonzero(mask)
    bbox = mask[yy.min() : yy.max() + 1, xx.min() : xx.max() + 1]
    right_quarter = bbox[:, max(0, 3 * bbox.shape[1] // 4) :]
    right_side_rows = float(right_quarter.any(axis=1).mean()) if right_quarter.size else 0.0
    right_eighth = bbox[:, max(0, 7 * bbox.shape[1] // 8) :]
    right_edge_rows = float(right_eighth.any(axis=1).mean()) if right_eighth.size else 0.0
    # ``F`` is terminal and must not steal ordinary foul symbols (especially
    # handwritten P/P1-P3). Keep this deliberately conservative: uncertain
    # cells continue through the normal foul recognizer and human review.
    if (
        f_score >= 0.92
        and vertical >= 0.72
        and top >= 0.52
        and middle >= 0.42
        and strong_top_rows >= 3
        and strong_middle_rows >= 3
        and top_run >= 0.48
        and middle_run >= 0.34
        and ink_ratio <= 0.38
        and right_side_rows <= 0.48
        and right_edge_rows <= 0.30
    ):
        return FoulTerminalObservation(
            FoulTerminalKind.DISQUALIFICATION,
            color,
            round(min(1.0, f_score), 4),
        )
    return None


def _longest_true_run(row: np.ndarray) -> int:
    values = np.flatnonzero(row)
    if len(values) == 0:
        return 0
    best = current = 1
    previous = int(values[0])
    for value in values[1:]:
        current_value = int(value)
        if current_value == previous + 1:
            current += 1
            best = max(best, current)
        else:
            current = 1
        previous = current_value
    return best


def _row_slot_bounds(
    width: int,
    *,
    slots: int,
    x_lines: tuple[int, ...] | None,
) -> list[tuple[int, int]]:
    if x_lines is not None and len(x_lines) == slots + 1:
        return [
            (max(0, x_lines[index]), min(width, x_lines[index + 1]))
            for index in range(slots)
        ]
    slot_width = width / slots
    return [
        (round(index * slot_width), round((index + 1) * slot_width))
        for index in range(slots)
    ]


def analyze_player_foul_row(
    row_image: np.ndarray,
    *,
    team: str,
    jersey: int,
    grid: PlayerFoulGridSpec = PLAYER_FOUL_GRID,
    recognizer: FoulSymbolRecognizer | None = None,
    x_lines: tuple[int, ...] | None = None,
    terminals: list[FoulTerminal] | None = None,
    grid_aligned: bool = True,
) -> list[FoulEvent]:
    separator = detect_half_separator(row_image, slots=grid.slots)
    height, width = row_image.shape[:2]
    slot_bounds = _row_slot_bounds(width, slots=grid.slots, x_lines=x_lines)
    events: list[FoulEvent] = []

    for index, (left, right) in enumerate(slot_bounds):
        margin_x = round((right - left) * grid.inner_margin)
        margin_y = round(height * grid.inner_margin)
        raw_cell = row_image[
            margin_y : max(margin_y + 1, height - margin_y),
            left + margin_x : max(left + margin_x + 1, right - margin_x),
        ]
        terminal = classify_foul_terminal(raw_cell)
        if terminal is not None:
            if terminals is not None:
                terminals.append(
                    FoulTerminal(
                        team=team.upper(),
                        jersey=jersey,
                        slot=index + 1,
                        kind=terminal.kind,
                        raw_symbol=(
                            "F"
                            if terminal.kind is FoulTerminalKind.DISQUALIFICATION
                            else "-"
                        ),
                        ink_color=terminal.ink_color.value,
                        confidence=terminal.confidence,
                        status=(
                            DecisionStatus.ACCEPTED
                            if grid_aligned and terminal.confidence >= 0.85
                            else DecisionStatus.REVIEW
                        ),
                    )
                )
            break

        cell = clean_player_foul_cell(raw_cell)
        red_mask, blue_mask = colored_ink_masks(cell)
        mask = red_mask | blue_mask
        if float(mask.mean()) < grid.min_ink_ratio:
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
            if raw_symbol is not None and raw_symbol.strip().upper() == "F":
                if terminals is not None:
                    terminals.append(
                        FoulTerminal(
                            team=team.upper(),
                            jersey=jersey,
                            slot=index + 1,
                            kind=FoulTerminalKind.DISQUALIFICATION,
                            raw_symbol="F",
                            ink_color=color.value,
                            confidence=symbol_confidence,
                            status=symbol_status,
                        )
                    )
                break
            parsed = parse_foul_symbol_details(raw_symbol)
            kind, free_throws = parsed.kind, parsed.free_throws
        else:
            parsed = parse_foul_symbol_details(None)

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
            and period_confidence is not None
            and period_confidence >= 0.60
            and grid_aligned
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
                cancelled_penalty=parsed.cancelled_penalty,
                counts_as_team_foul=parsed.counts_as_team_foul,
                fighting=parsed.fighting,
                confidence=confidence,
                status=(
                    DecisionStatus.ACCEPTED
                    if accepted
                    else DecisionStatus.REVIEW
                ),
            )
        )
    return events


def clean_player_foul_cell(
    image: np.ndarray,
    *,
    padding: int = 1,
    edge_fraction: float = 0.10,
) -> np.ndarray:
    """Remove linhas impressas coloridas que atravessam a celula de falta.

    Algumas digitalizacoes tornam as linhas azuis/roxas da grade indistintas
    da caneta para o filtro de cor. Linhas da grade atravessam quase toda a
    largura/altura da celula; simbolos manuscritos ocupam apenas parte dela.
    """
    rgb = _as_rgb(image).copy()
    if rgb.size == 0:
        return rgb

    edge_width = max(1, round(rgb.shape[1] * edge_fraction))
    rgb[:, :edge_width, :] = 255
    rgb[:, rgb.shape[1] - edge_width :, :] = 255

    red_mask, blue_mask = colored_ink_masks(rgb)
    colored_mask = red_mask | blue_mask
    row_lines = np.flatnonzero(
        colored_mask.mean(axis=1) >= 0.60
    )

    rows_to_clear = _expanded_indexes(row_lines, rgb.shape[0], padding)
    if rows_to_clear.size:
        rgb[rows_to_clear, :, :] = 255
    return rgb


def _expanded_indexes(indexes: np.ndarray, size: int, padding: int) -> np.ndarray:
    if indexes.size == 0:
        return indexes
    offsets = range(-max(padding, 0), max(padding, 0) + 1)
    expanded = np.concatenate([indexes + offset for offset in offsets])
    return np.unique(expanded[(expanded >= 0) & (expanded < size)])


def extract_player_fouls(
    normalized_image: np.ndarray,
    *,
    team: str,
    jerseys: list[int],
    template: TemplateSpec = FECABA_V1,
    grid: PlayerFoulGridSpec = PLAYER_FOUL_GRID,
    recognizer: FoulSymbolRecognizer | None = None,
) -> list[FoulEvent]:
    events, _ = extract_player_foul_data(
        normalized_image,
        team=team,
        jerseys=jerseys,
        template=template,
        grid=grid,
        recognizer=recognizer,
    )
    return events


def extract_player_foul_data(
    normalized_image: np.ndarray,
    *,
    team: str,
    jerseys: list[int],
    template: TemplateSpec = FECABA_V1,
    grid: PlayerFoulGridSpec = PLAYER_FOUL_GRID,
    recognizer: FoulSymbolRecognizer | None = None,
) -> tuple[list[FoulEvent], list[FoulTerminal]]:
    side = team.upper()
    if side not in {"A", "B"}:
        raise ValueError("team deve ser A ou B")
    if len(jerseys) > grid.roster_rows:
        raise ValueError(
            f"O template suporta no maximo {grid.roster_rows} linhas de jogadores"
        )

    region_name = f"team_{side.lower()}_player_fouls"
    block = crop_region(normalized_image, template.region(region_name))
    events: list[FoulEvent] = []
    terminals: list[FoulTerminal] = []
    detected = detect_player_foul_grid(block, side=side, grid=grid)

    if detected.valid:
        for index, jersey in enumerate(jerseys):
            top = detected.y_lines[index]
            bottom = detected.y_lines[index + 1]
            row = block[top:bottom]
            events.extend(
                analyze_player_foul_row(
                    row,
                    team=side,
                    jersey=jersey,
                    grid=grid,
                    recognizer=recognizer,
                    x_lines=detected.x_lines,
                    terminals=terminals,
                    grid_aligned=True,
                )
            )
        return events, terminals

    data_top = _player_foul_data_top(block, side=side, grid=grid)
    data_bottom = _player_foul_data_bottom(block, side=side, grid=grid)
    data = block[data_top:data_bottom]
    row_height = data.shape[0] / grid.roster_rows
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
                terminals=terminals,
                grid_aligned=False,
            )
        )
    return events, terminals


def _player_foul_data_top(
    block: np.ndarray,
    *,
    side: str,
    grid: PlayerFoulGridSpec = PLAYER_FOUL_GRID,
) -> int:
    """Retorna o inicio das linhas de jogadores no bloco de faltas.

    No formulario FECABA v1 o bloco da Equipe A inclui uma faixa maior acima
    do cabecalho de faltas (desafio tecnico), enquanto a Equipe B comeca muito
    mais perto do cabecalho. As fracoes ficam no spec para manter essa
    assimetria explicita e compartilhada entre analise e dataset.
    """
    normalized_side = side.upper()
    if normalized_side not in {"A", "B"}:
        raise ValueError("side deve ser A ou B")
    fraction = (
        grid.team_a_header_fraction
        if normalized_side == "A"
        else grid.header_fraction
    )
    return round(block.shape[0] * fraction)


def _player_foul_data_bottom(
    block: np.ndarray,
    *,
    side: str,
    grid: PlayerFoulGridSpec = PLAYER_FOUL_GRID,
) -> int:
    normalized_side = side.upper()
    if normalized_side not in {"A", "B"}:
        raise ValueError("side deve ser A ou B")
    if normalized_side == "A":
        return block.shape[0]
    return round(block.shape[0] * (1.0 - grid.team_b_footer_fraction))


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
