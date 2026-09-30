from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .imaging import crop_region
from .models import DecisionStatus, FoulEvent, FoulKind, TeamFoulIndicator
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
    group_left: float = 0.08
    group_width: float = 0.37
    group_right: float = 0.56
    row_top: float = 0.02
    row_height: float = 0.43
    row_bottom: float = 0.55


@dataclass(frozen=True, slots=True)
class PlayerFoulGridSpec:
    header_fraction: float = 0.085
    roster_rows: int = 12
    slots: int = 5
    inner_margin: float = 0.12


TEAM_FOUL_GRID = TeamFoulGridSpec()
PLAYER_FOUL_GRID = PlayerFoulGridSpec()


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


def extract_team_foul_indicators(
    normalized_image: np.ndarray,
    *,
    team: str,
    template: TemplateSpec = FECABA_V1,
) -> list[TeamFoulIndicator]:
    side = team.upper()
    if side not in {"A", "B"}:
        raise ValueError("team deve ser A ou B")

    region_name = f"team_{side.lower()}_team_fouls"
    block = crop_region(normalized_image, template.region(region_name))
    observations: dict[int, list[TeamFoulCellObservation]] = {
        period: [] for period in range(1, 5)
    }
    for period, _, rect in iter_team_foul_cells():
        observations[period].append(
            classify_team_foul_cell(crop_region(block, rect))
        )

    indicators: list[TeamFoulIndicator] = []
    for period in range(1, 5):
        cells = observations[period]
        x_cells = [cell for cell in cells if cell.kind is TeamFoulCellKind.X]
        ambiguous = any(
            cell.kind is TeamFoulCellKind.AMBIGUOUS for cell in cells
        )
        confidence = min((cell.confidence for cell in cells), default=0.0)
        indicators.append(
            TeamFoulIndicator(
                period=period,
                x_count=len(x_cells),
                confidence=round(float(confidence), 4),
                status=(
                    DecisionStatus.REVIEW
                    if ambiguous
                    else DecisionStatus.ACCEPTED
                ),
            )
        )
    return indicators


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
        events.append(
            FoulEvent(
                team=team.upper(),
                jersey=jersey,
                slot=index + 1,
                period=period,
                kind=FoulKind.UNKNOWN,
                raw_symbol=None,
                ink_color=color.value,
                confidence=separator.confidence if period is not None else None,
                status=(
                    DecisionStatus.ACCEPTED
                    if period is not None
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
