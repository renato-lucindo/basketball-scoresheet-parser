from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np

from .decision import DecisionEngine
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
    rows_bottom_fraction: float = 0.946
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


@dataclass(slots=True)
class ScoringGridDetection:
    x_lines: tuple[int, ...]
    y_lines: tuple[int, ...]
    confidence: float

    @property
    def valid(self) -> bool:
        return len(self.x_lines) == 17 and len(self.y_lines) == 41


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
    if image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
        return ScoreMarkObservation(
            kind=ScoreMarkKind.EMPTY,
            ink_color=InkColor.UNKNOWN,
            confidence=1.0,
            ink_ratio=0.0,
            elongation=1.0,
        )
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


def detect_scoring_grid(
    image: np.ndarray,
    *,
    grid: ScoringGridSpec = DEFAULT_GRID,
) -> ScoringGridDetection:
    """Localiza a grade impressa da contagem de pontos.

    A folha real pode sofrer pequenas distorcoes mesmo depois da homografia.
    Por isso as celulas de pontuacao sao ancoradas nas linhas pretas do
    formulario, em vez de depender apenas de fracoes fixas do crop.
    """
    rgb = _ensure_rgb(image)
    gray = (
        0.299 * rgb[..., 0].astype(np.float32)
        + 0.587 * rgb[..., 1].astype(np.float32)
        + 0.114 * rgb[..., 2].astype(np.float32)
    )
    dark = gray < 145
    height, width = dark.shape

    x_lines = _detect_vertical_lines(dark, expected=grid.panels * 4 + 1)
    if len(x_lines) != grid.panels * 4 + 1:
        return ScoringGridDetection(tuple(), tuple(), 0.0)

    y_candidates = _horizontal_line_candidates(dark, x_lines)
    y_lines = _fit_row_lattice(
        y_candidates,
        height=height,
        rows=grid.rows_per_panel,
        expected_header_fraction=grid.header_fraction,
        expected_bottom_fraction=grid.rows_bottom_fraction,
    )
    if len(y_lines) != grid.rows_per_panel + 1:
        return ScoringGridDetection(tuple(x_lines), tuple(), 0.0)

    x_span = x_lines[-1] - x_lines[0]
    y_span = y_lines[-1] - y_lines[0]
    coverage = min(
        1.0,
        (x_span / max(width, 1) + y_span / max(height, 1)) / 1.85,
    )
    return ScoringGridDetection(
        x_lines=tuple(x_lines),
        y_lines=tuple(y_lines),
        confidence=round(float(coverage), 4),
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
    decision_engine: DecisionEngine | None = None,
    decision_threshold: float = 0.85,
) -> list[ScoringEvent]:
    table = crop_region(normalized_image, template.region("scoring_table"))
    detected_grid = detect_scoring_grid(table, grid=grid)
    observations: list[ScoringEvent] = []

    for cell, score_crop, jersey_crop, grid_aligned in _iter_scoring_crops(
        table,
        detected_grid,
        grid,
    ):
        mark = classify_score_mark(score_crop)
        jersey_red, jersey_blue = colored_ink_masks(jersey_crop)
        jersey_ink = float((jersey_red | jersey_blue).mean())
        if grid_aligned:
            if jersey_ink < 0.025 or _looks_like_closure_stroke(jersey_crop):
                continue
        elif mark.kind is ScoreMarkKind.EMPTY:
            continue

        ink_color = mark.ink_color
        if grid_aligned and ink_color is InkColor.UNKNOWN:
            ink_color = _dominant_color(jersey_red, jersey_blue)

        circle = detect_jersey_circle(jersey_crop)
        jersey: int | None = None
        jersey_confidence: float | None = None
        if recognizer is not None:
            jersey, jersey_confidence = recognizer.recognize(
                jersey_crop,
                team=cell.team,
            )

        status = DecisionStatus.ACCEPTED
        decision_confidence: float | None = None
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

        if (
            shot_type is ShotType.AMBIGUOUS
            and decision_engine is not None
        ):
            try:
                answer = decision_engine.classify_scoring_mark(
                    team=cell.team,
                    running_score=cell.running_score,
                    evidence={
                        "local_kind": mark.kind.value,
                        "local_confidence": mark.confidence,
                        "ink_color": ink_color.value,
                        "ink_ratio": mark.ink_ratio,
                        "elongation": mark.elongation,
                        "jersey_ink_ratio": round(jersey_ink, 6),
                        "circle_detected": circle.detected,
                        "circle_confidence": circle.confidence,
                        "grid_aligned": grid_aligned,
                    },
                )
            except RuntimeError:
                answer = None

            if answer is not None and answer.confidence >= decision_threshold:
                resolved = {
                    "free_throw": (ShotType.FREE_THROW, 1),
                    "two_point": (ShotType.TWO_POINT, 2),
                    "three_point": (ShotType.THREE_POINT, 3),
                }.get(answer.choice)
                if answer.choice == "closure_stroke":
                    continue
                if resolved is not None:
                    shot_type, points = resolved
                    status = DecisionStatus.ACCEPTED
                    decision_confidence = float(answer.confidence)

        confidence = (
            decision_confidence
            if decision_confidence is not None
            else mark.confidence
        )
        if shot_type is ShotType.THREE_POINT:
            confidence = min(confidence, circle.confidence)

        observations.append(
            ScoringEvent(
                team=cell.team,
                period=None,
                running_score=cell.running_score,
                jersey=jersey,
                shot_type=shot_type,
                points=points,
                ink_color=ink_color.value,
                confidence=round(float(confidence), 4),
                jersey_confidence=jersey_confidence,
                status=status,
            )
        )

    events = select_plausible_scoring_sequences(observations)
    assign_periods_from_color_runs(events)
    return events


def select_plausible_scoring_sequences(
    events: list[ScoringEvent],
) -> list[ScoringEvent]:
    """Usa a progressao 1..160 como validacao, sem reclassificar a marca."""
    selected: list[ScoringEvent] = []
    for team in ("A", "B"):
        candidates = sorted(
            (event for event in events if event.team == team),
            key=lambda event: event.running_score,
        )
        previous_score = 0
        for event in candidates:
            delta = event.running_score - previous_score
            if delta not in {1, 2, 3} or (event.points > 0 and event.points != delta):
                event.status = DecisionStatus.REVIEW
                event.confidence = min(event.confidence or 0.0, 0.69)
            previous_score = max(previous_score, event.running_score)
            selected.append(event)

    return sorted(selected, key=lambda event: (event.team, event.running_score))


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

            if previous_color is None:
                period = 1 if color == InkColor.RED.value else 2
                previous_color = color
            elif color != previous_color:
                possible = (1, 3) if color == InkColor.RED.value else (2, 4)
                later = [candidate for candidate in possible if candidate > period]
                if not later:
                    event.period = None
                    event.status = DecisionStatus.REVIEW
                    continue
                period = later[0]
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


def _colored_ink_ratio(image: np.ndarray) -> float:
    red_mask, blue_mask = colored_ink_masks(image)
    return float((red_mask | blue_mask).mean())


def _looks_like_closure_stroke(image: np.ndarray) -> bool:
    red_mask, blue_mask = colored_ink_masks(image)
    mask = red_mask | blue_mask
    yy, xx = np.nonzero(mask)
    if len(xx) < 4:
        return False
    points = np.column_stack((xx, yy)).astype(np.float64)
    centered = points - points.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(len(points) - 1, 1)
    eigenvalues = np.linalg.eigvalsh(covariance)
    elongation = float(np.sqrt(max(eigenvalues[-1], 1e-6) / max(eigenvalues[0], 1e-6)))
    height, width = mask.shape
    width_span = (xx.max() - xx.min() + 1) / max(width, 1)
    height_span = (yy.max() - yy.min() + 1) / max(height, 1)
    ink_ratio = float(mask.mean())
    return elongation > 3.0 or (
        elongation > 2.0
        and width_span > 0.84
        and height_span < 0.86
        and ink_ratio < 0.16
    )


def _iter_scoring_crops(
    table: np.ndarray,
    detected_grid: ScoringGridDetection,
    grid: ScoringGridSpec,
):
    if detected_grid.valid:
        x_lines = detected_grid.x_lines
        y_lines = detected_grid.y_lines
        for panel in range(grid.panels):
            x = x_lines[panel * 4 : panel * 4 + 5]
            for row in range(grid.rows_per_panel):
                running_score = panel * grid.rows_per_panel + row + 1
                y0, y1 = y_lines[row], y_lines[row + 1]
                margin_y = max(1, round((y1 - y0) * 0.10))
                for team, score_index, jersey_index in (
                    ("A", 1, 0),
                    ("B", 2, 3),
                ):
                    sx0, sx1 = x[score_index], x[score_index + 1]
                    jx0, jx1 = x[jersey_index], x[jersey_index + 1]
                    score_margin = max(1, round((sx1 - sx0) * 0.08))
                    jersey_margin = max(1, round((jx1 - jx0) * 0.05))
                    score_crop = table[
                        y0 + margin_y : y1 - margin_y,
                        sx0 + score_margin : sx1 - score_margin,
                    ]
                    jersey_crop = table[
                        y0 + margin_y : y1 - margin_y,
                        jx0 + jersey_margin : jx1 - jersey_margin,
                    ]
                    yield (
                        ScoringCell(
                            team,
                            running_score,
                            NormalizedRect(0, 0, 0, 0),
                            NormalizedRect(0, 0, 0, 0),
                        ),
                        score_crop,
                        jersey_crop,
                        True,
                    )
        return

    for cell in iter_scoring_cells(grid):
        yield (
            cell,
            crop_region(table, cell.score_rect),
            crop_region(table, cell.jersey_rect),
            False,
        )


def _detect_vertical_lines(dark: np.ndarray, *, expected: int) -> list[int]:
    ratio = dark.mean(axis=0)
    width = len(ratio)
    if expected < 2 or width < expected:
        return []

    best: tuple[tuple[int, float, float], list[int]] | None = None
    merge_distance = max(4, round(width * 0.015))
    for threshold in (0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30):
        groups = _contiguous_groups(np.flatnonzero(ratio >= threshold))
        centers = [round((start + end) / 2) for start, end in groups]
        centers = _merge_vertical_candidates(
            centers,
            ratio,
            max_distance=merge_distance,
        )
        fitted, inferred = _complete_vertical_lattice(
            centers,
            width=width,
            expected=expected,
        )
        if len(fitted) != expected or inferred > 2:
            continue

        # Prefere grades apoiadas no maior numero de linhas realmente vistas.
        # Em empate, privilegia perfis mais escuros e limiares mais altos.
        anchor_strength = float(
            np.mean([ratio[min(max(center, 0), width - 1)] for center in centers])
        )
        rank = (inferred, -anchor_strength, -threshold)
        if best is None or rank < best[0]:
            best = (rank, fitted)

    return best[1] if best is not None else []


def _merge_vertical_candidates(
    centers: list[int],
    ratio: np.ndarray,
    *,
    max_distance: int,
) -> list[int]:
    if not centers:
        return []
    clusters: list[list[int]] = [[centers[0]]]
    for center in centers[1:]:
        if center - clusters[-1][-1] <= max_distance:
            clusters[-1].append(center)
        else:
            clusters.append([center])

    merged: list[int] = []
    for cluster in clusters:
        weights = [max(float(ratio[center]), 1e-6) for center in cluster]
        merged.append(round(float(np.average(cluster, weights=weights))))
    return merged


def _complete_vertical_lattice(
    centers: list[int],
    *,
    width: int,
    expected: int,
) -> tuple[list[int], int]:
    """Completa no maximo duas linhas fracas usando a malha observada.

    As linhas verticais impressas sao muito persistentes, mas bordas podem
    desaparecer no recorte e uma divisoria pode ficar fraca por rasura. A
    reconstrucao so acontece quando as demais linhas fornecem ancoras reais e
    o espacamento resultante continua compativel com a grade.
    """
    if len(centers) < expected - 2 or len(centers) > expected or len(centers) < 2:
        return [], 0

    max_gap = width * 0.09
    completed: list[int] = [centers[0]]
    inferred = 0
    for right in centers[1:]:
        left = completed[-1]
        gap = right - left
        intervals = max(1, int(np.ceil(gap / max_gap)))
        missing = intervals - 1
        if inferred + missing > 2:
            return [], 0
        for step in range(1, intervals):
            completed.append(round(left + gap * step / intervals))
            inferred += 1
        completed.append(right)

    if len(completed) > expected:
        return [], 0

    missing_edges = expected - len(completed)
    if missing_edges:
        gaps = np.diff(completed)
        typical = float(np.median(gaps)) if len(gaps) else width / (expected - 1)
        typical = max(typical, 1.0)
        left_gap = float(completed[0])
        right_gap = float((width - 1) - completed[-1])

        best_edge_fit: tuple[float, int, int] | None = None
        for left_count in range(missing_edges + 1):
            right_count = missing_edges - left_count
            cost = 0.0
            if left_count:
                cost += abs(left_gap / left_count - typical) / typical
            else:
                cost += max(0.0, left_gap / typical - 0.45)
            if right_count:
                cost += abs(right_gap / right_count - typical) / typical
            else:
                cost += max(0.0, right_gap / typical - 0.45)
            candidate = (cost, left_count, right_count)
            if best_edge_fit is None or candidate < best_edge_fit:
                best_edge_fit = candidate

        assert best_edge_fit is not None
        _, left_count, right_count = best_edge_fit
        left_values = [
            round(completed[0] * step / left_count)
            for step in range(left_count)
        ] if left_count else []
        right_values = [
            round(
                completed[-1]
                + ((width - 1) - completed[-1]) * step / right_count
            )
            for step in range(1, right_count + 1)
        ] if right_count else []
        completed = left_values + completed + right_values
        inferred += missing_edges

    if len(completed) != expected:
        return [], 0
    if any(second <= first for first, second in zip(completed, completed[1:])):
        return [], 0
    return completed, inferred


def _horizontal_line_candidates(
    dark: np.ndarray,
    x_lines: list[int],
) -> list[int]:
    raw: list[int] = []
    for left, right in zip(x_lines[:-1], x_lines[1:]):
        margin = max(2, round((right - left) * 0.07))
        if right - left <= margin * 2:
            continue
        profile = dark[:, left + margin : right - margin].mean(axis=1)
        for y in range(2, len(profile) - 2):
            window = profile[y - 2 : y + 3]
            if profile[y] >= 0.50 and profile[y] >= float(window.max()):
                raw.append(y)

    if not raw:
        return []
    raw.sort()
    clusters: list[list[int]] = []
    for y in raw:
        if not clusters or y - clusters[-1][-1] > 4:
            clusters.append([y])
        else:
            clusters[-1].append(y)
    return [round(float(np.median(cluster))) for cluster in clusters]


def _fit_row_lattice(
    candidates: list[int],
    *,
    height: int,
    rows: int,
    expected_header_fraction: float,
    expected_bottom_fraction: float,
) -> list[int]:
    if len(candidates) < rows + 1:
        return []

    expected_start = height * expected_header_fraction
    expected_end = height * expected_bottom_fraction
    starts = [
        candidate
        for candidate in candidates
        if abs(candidate - expected_start) <= height * 0.035
    ]
    ends = [
        candidate
        for candidate in candidates
        if abs(candidate - expected_end) <= height * 0.035
    ]
    if not starts or not ends:
        return []
    # No template FECABA, header_fraction aponta para a linha que separa o
    # cabecalho A/B da linha do placar 1. O titulo e outras linhas proximas
    # pertencem ao mesmo reticulado, portanto escolher o inicio apenas pelo
    # maior numero de matches pode deslocar toda a leitura em uma linha.
    start = min(starts, key=lambda value: abs(value - expected_start))
    end = min(ends, key=lambda value: abs(value - expected_end))
    if end <= start:
        return []
    step = (end - start) / rows
    best_lines: list[int] = []
    for index in range(rows + 1):
        predicted = start + index * step
        nearest = min(candidates, key=lambda value: abs(value - predicted))
        if abs(nearest - predicted) <= 5.5:
            best_lines.append(nearest)
        else:
            best_lines.append(round(predicted))
    # Evita duplicatas quando duas previsoes caem sobre o mesmo pico.
    for first, second in zip(best_lines, best_lines[1:]):
        if second <= first:
            return []
    return best_lines


def _contiguous_groups(values: np.ndarray) -> list[tuple[int, int]]:
    if len(values) == 0:
        return []
    groups: list[tuple[int, int]] = []
    start = previous = int(values[0])
    for value in values[1:]:
        current = int(value)
        if current > previous + 1:
            groups.append((start, previous))
            start = current
        previous = current
    groups.append((start, previous))
    return groups


def _ensure_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("A imagem deve ter formato HxWx3")
    return array[..., :3].astype(np.uint8, copy=False)
