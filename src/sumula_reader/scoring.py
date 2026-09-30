from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np

from .imaging import crop_region
from .models import DecisionStatus, ScoringEvent, ShotType
from .template import FECABA_V1, NormalizedRect, TemplateSpec


class InkColor(str, Enum):
    RED = "red"
    BLUE = "blue"
    UNKNOWN = "unknown"


class ScoreMarkKind(str, Enum):
    EMPTY = "empty"
    FREE_THROW = "free_throw"
    FIELD_GOAL = "field_goal"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True, slots=True)
class ScoringGridSpec:
    panels: int = 4
    rows_per_panel: int = 40
    header_fraction: float = 0.047
    jersey_a_end: float = 0.265
    score_a_end: float = 0.505
    score_b_end: float = 0.745


@dataclass(slots=True)
class ScoreMarkObservation:
    kind: ScoreMarkKind
    ink_color: InkColor
    confidence: float
    ink_ratio: float
    elongation: float


@dataclass(slots=True)
class CircleObservation:
    detected: bool
    confidence: float
    angular_coverage: float


class JerseyRecognizer(Protocol):
    def recognize(
        self,
        image: np.ndarray,
        *,
        team: str,
    ) -> tuple[int | None, float | None]: ...


@dataclass(frozen=True, slots=True)
class ScoringCell:
    team: str
    running_score: int
    score_rect: NormalizedRect
    jersey_rect: NormalizedRect


DEFAULT_GRID = ScoringGridSpec()


def colored_ink_masks(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rgb = _ensure_rgb(image).astype(np.int16, copy=False)
    red = rgb[..., 0]
    green = rgb[..., 1]
    blue = rgb[..., 2]
    saturation = np.maximum.reduce([red, green, blue]) - np.minimum.reduce(
        [red, green, blue]
    )
    colored = saturation >= 28

    red_mask = (
        colored
        & (red >= green + 18)
        & (red >= blue + 15)
        & (red >= 70)
    )
    blue_mask = (
        colored
        & (blue >= red + 15)
        & (blue >= green + 5)
        & (blue >= 65)
    )
    return red_mask, blue_mask


def classify_score_mark(
    image: np.ndarray,
    *,
    min_ink_ratio: float = 0.004,
) -> ScoreMarkObservation:
    red_mask, blue_mask = colored_ink_masks(image)
    mask = red_mask | blue_mask
    ink_ratio = float(mask.mean())
    color = _dominant_color(red_mask, blue_mask)

    if ink_ratio < min_ink_ratio:
        return ScoreMarkObservation(
            kind=ScoreMarkKind.EMPTY,
            ink_color=InkColor.UNKNOWN,
            confidence=min(1.0, 1.0 - ink_ratio / max(min_ink_ratio, 1e-9)),
            ink_ratio=ink_ratio,
            elongation=1.0,
        )

    yy, xx = np.nonzero(mask)
    if len(xx) < 4:
        return ScoreMarkObservation(
            kind=ScoreMarkKind.AMBIGUOUS,
            ink_color=color,
            confidence=0.0,
            ink_ratio=ink_ratio,
            elongation=1.0,
        )

    points = np.column_stack((xx, yy)).astype(np.float64)
    centered = points - points.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(len(points) - 1, 1)
    eigenvalues = np.linalg.eigvalsh(covariance)
    minor = max(float(eigenvalues[0]), 1e-6)
    major = max(float(eigenvalues[-1]), minor)
    elongation = float(np.sqrt(major / minor))

    width = max(int(xx.max() - xx.min() + 1), 1)
    height = max(int(yy.max() - yy.min() + 1), 1)
    fill_ratio = len(xx) / float(width * height)
    span_ratio = max(
        width / max(mask.shape[1], 1),
        height / max(mask.shape[0], 1),
    )

    # O ponto de lance livre tende a ser compacto. A diagonal ocupa uma
    # extensao maior e apresenta forte eixo principal.
    dot_score = (
        (1.0 / max(elongation, 1.0)) * 0.45
        + min(fill_ratio / 0.55, 1.0) * 0.35
        + max(0.0, 1.0 - span_ratio / 0.62) * 0.20
    )
    slash_score = (
        min(elongation / 4.0, 1.0) * 0.60
        + min(span_ratio / 0.70, 1.0) * 0.40
    )

    if dot_score >= 0.58 and dot_score > slash_score + 0.05:
        kind = ScoreMarkKind.FREE_THROW
        confidence = dot_score
    elif slash_score >= 0.58 and slash_score > dot_score:
        kind = ScoreMarkKind.FIELD_GOAL
        confidence = slash_score
    else:
        kind = ScoreMarkKind.AMBIGUOUS
        confidence = max(dot_score, slash_score) * 0.75

    return ScoreMarkObservation(
        kind=kind,
        ink_color=color,
        confidence=round(float(min(confidence, 1.0)), 4),
        ink_ratio=round(ink_ratio, 6),
        elongation=round(elongation, 4),
    )


def detect_jersey_circle(
    image: np.ndarray,
    *,
    min_ink_ratio: float = 0.006,
) -> CircleObservation:
    red_mask, blue_mask = colored_ink_masks(image)
    mask = red_mask | blue_mask
    if float(mask.mean()) < min_ink_ratio:
        return CircleObservation(False, 0.0, 0.0)

    height, width = mask.shape
    yy, xx = np.nonzero(mask)
    if len(xx) < 8:
        return CircleObservation(False, 0.0, 0.0)

    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    nx = (xx - cx) / max(width * 0.5, 1.0)
    ny = (yy - cy) / max(height * 0.5, 1.0)
    radius = np.sqrt(nx * nx + ny * ny)
    ring_points = (radius >= 0.48) & (radius <= 1.05)
    if ring_points.sum() < 6:
        return CircleObservation(False, 0.0, 0.0)

    angles = np.arctan2(ny[ring_points], nx[ring_points])
    bins = np.floor((angles + np.pi) / (2 * np.pi) * 24).astype(int)
    bins = np.clip(bins, 0, 23)
    angular_coverage = len(np.unique(bins)) / 24.0

    # Um numero simples pode tocar parte do anel, mas raramente ocupa a
    # maioria dos setores angulares. Círculos imperfeitos continuam cobrindo
    # grande parte dos setores.
    detected = angular_coverage >= 0.54
    confidence = min(1.0, max(0.0, (angular_coverage - 0.28) / 0.50))
    return CircleObservation(
        detected=detected,
        confidence=round(float(confidence), 4),
        angular_coverage=round(float(angular_coverage), 4),
    )


def iter_scoring_cells(
    grid: ScoringGridSpec = DEFAULT_GRID,
) -> tuple[ScoringCell, ...]:
    cells: list[ScoringCell] = []
    panel_width = 1.0 / grid.panels
    data_height = 1.0 - grid.header_fraction
    row_height = data_height / grid.rows_per_panel

    for panel in range(grid.panels):
        panel_x = panel * panel_width
        for row in range(grid.rows_per_panel):
            score = panel * grid.rows_per_panel + row + 1
            y = grid.header_fraction + row * row_height

            a_jersey = NormalizedRect(
                panel_x,
                y,
                panel_width * grid.jersey_a_end,
                row_height,
            )
            a_score = NormalizedRect(
                panel_x + panel_width * grid.jersey_a_end,
                y,
                panel_width * (grid.score_a_end - grid.jersey_a_end),
                row_height,
            )
            b_score = NormalizedRect(
                panel_x + panel_width * grid.score_a_end,
                y,
                panel_width * (grid.score_b_end - grid.score_a_end),
                row_height,
            )
            b_jersey = NormalizedRect(
                panel_x + panel_width * grid.score_b_end,
                y,
                panel_width * (1.0 - grid.score_b_end),
                row_height,
            )
            cells.append(ScoringCell("A", score, a_score, a_jersey))
            cells.append(ScoringCell("B", score, b_score, b_jersey))
    return tuple(cells)


def extract_scoring_events(
    normalized_image: np.ndarray,
    *,
    template: TemplateSpec = FECABA_V1,
    recognizer: JerseyRecognizer | None = None,
    grid: ScoringGridSpec = DEFAULT_GRID,
) -> list[ScoringEvent]:
    table = crop_region(normalized_image, template.region("scoring_table"))
    events: list[ScoringEvent] = []

    for cell in iter_scoring_cells(grid):
        score_crop = crop_region(table, cell.score_rect)
        mark = classify_score_mark(score_crop)
        if mark.kind is ScoreMarkKind.EMPTY:
            continue

        jersey_crop = crop_region(table, cell.jersey_rect)
        circle = detect_jersey_circle(jersey_crop)
        jersey: int | None = None
        jersey_confidence: float | None = None
        if recognizer is not None:
            jersey, jersey_confidence = recognizer.recognize(
                jersey_crop,
                team=cell.team,
            )

        status = DecisionStatus.ACCEPTED
        if mark.kind is ScoreMarkKind.FREE_THROW:
            shot_type = ShotType.FREE_THROW
            points = 1
        elif mark.kind is ScoreMarkKind.FIELD_GOAL and circle.detected:
            shot_type = ShotType.THREE_POINT
            points = 3
        elif mark.kind is ScoreMarkKind.FIELD_GOAL:
            shot_type = ShotType.TWO_POINT
            points = 2
        else:
            shot_type = ShotType.AMBIGUOUS
            points = 0
            status = DecisionStatus.REVIEW

        confidence = mark.confidence
        if shot_type is ShotType.THREE_POINT:
            confidence = min(confidence, circle.confidence)

        events.append(
            ScoringEvent(
                team=cell.team,
                period=None,
                running_score=cell.running_score,
                jersey=jersey,
                shot_type=shot_type,
                points=points,
                ink_color=mark.ink_color.value,
                confidence=round(float(confidence), 4),
                jersey_confidence=jersey_confidence,
                status=status,
            )
        )

    assign_periods_from_color_runs(events)
    return events


def assign_periods_from_color_runs(events: list[ScoringEvent]) -> None:
    """Atribui Q1..Q4 pelos blocos cronologicos de cor de cada equipe.

    A contagem corrente de cada equipe e monotona, logo a sequencia de
    anotacoes daquela equipe tambem e cronologica. A cada troca de cor inicia
    um novo periodo. Eventos com cor desconhecida ficam sem periodo.
    """
    for team in ("A", "B"):
        team_events = sorted(
            (event for event in events if event.team == team),
            key=lambda event: event.running_score,
        )
        period = 0
        previous_color: str | None = None
        for event in team_events:
            color = event.ink_color
            if color not in (InkColor.RED.value, InkColor.BLUE.value):
                event.period = None
                event.status = DecisionStatus.REVIEW
                continue

            if previous_color is None or color != previous_color:
                period += 1
                previous_color = color

            if period > 4:
                event.period = None
                event.status = DecisionStatus.REVIEW
            else:
                event.period = period


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


def _ensure_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("A imagem deve ter formato HxWx3")
    return array[..., :3].astype(np.uint8, copy=False)
