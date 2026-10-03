from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .imaging import crop_region
from .models import (
    DecisionStatus,
    ParticipantMark,
    PlayerResult,
    RosterRowObservation,
)
from .recognition import HandwritingRecognizer, WriterProfile, apply_writer_profile
from .template import FECABA_V1, TemplateSpec


@dataclass(slots=True)
class ParticipationDetection:
    mark: ParticipantMark
    participated: bool
    starter: bool
    participation_confidence: float
    starter_confidence: float
    status: DecisionStatus
    blue_pixel_ratio: float
    red_pixel_ratio: float
    red_ring_ratio: float


@dataclass(frozen=True, slots=True)
class ParticipationGridSpec:
    header_fraction: float = 0.085
    roster_rows: int = 12


PARTICIPATION_GRID = ParticipationGridSpec()


@dataclass(slots=True)
class RosterExtraction:
    jerseys: list[int] = field(default_factory=list)
    status: DecisionStatus = DecisionStatus.UNRESOLVED
    observations: list[RosterRowObservation] = field(default_factory=list)


def _as_rgb_array(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError("The image must have HxWx3 (RGB) shape")
    return arr[..., :3].astype(np.int16, copy=False)


def _color_masks(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    red = rgb[..., 0]
    green = rgb[..., 1]
    blue = rgb[..., 2]

    saturation = np.maximum.reduce([red, green, blue]) - np.minimum.reduce([red, green, blue])
    colored = saturation >= 35

    blue_mask = colored & (blue >= red + 20) & (blue >= green + 10) & (blue >= 70)
    red_mask = colored & (red >= green + 25) & (red >= blue + 20) & (red >= 80)
    return blue_mask, red_mask


def detect_participation(
    image: np.ndarray,
    *,
    min_ink_ratio: float = 0.008,
    starter_ring_ratio: float = 0.015,
) -> ParticipationDetection:
    """Classify a player's participation cell.

    MVP rules:
    - blue or red X: participated;
    - blue X surrounded by a red circle: starter;
    - insufficient ink: did not participate.

    This first version uses only color distribution. Template geometry keeps
    the supplied crop within the participation cell.
    """
    rgb = _as_rgb_array(image)
    blue_mask, red_mask = _color_masks(rgb)

    h, w = blue_mask.shape
    total = float(h * w)
    blue_ratio = float(blue_mask.sum() / total)
    red_ratio = float(red_mask.sum() / total)

    yy, xx = np.ogrid[:h, :w]
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    nx = (xx - cx) / max(w / 2.0, 1.0)
    ny = (yy - cy) / max(h / 2.0, 1.0)
    radius = np.sqrt(nx * nx + ny * ny)
    ring = (radius >= 0.42) & (radius <= 0.92)
    ring_pixels = max(int(ring.sum()), 1)
    red_ring_ratio = float((red_mask & ring).sum() / ring_pixels)

    has_blue = blue_ratio >= min_ink_ratio
    has_red = red_ratio >= min_ink_ratio
    participated = has_blue or has_red
    starter = has_blue and has_red and red_ring_ratio >= starter_ring_ratio

    if starter:
        mark = ParticipantMark.BLUE_X_RED_CIRCLE
    elif has_blue and not has_red:
        mark = ParticipantMark.BLUE_X
    elif has_red and not has_blue:
        mark = ParticipantMark.RED_X
    elif has_blue and has_red:
        mark = ParticipantMark.AMBIGUOUS
    else:
        mark = ParticipantMark.NONE

    participation_confidence = min(1.0, max(blue_ratio, red_ratio) / max(min_ink_ratio * 4.0, 1e-9)) if participated else min(1.0, 1.0 - max(blue_ratio, red_ratio) / max(min_ink_ratio, 1e-9))
    starter_confidence = min(1.0, red_ring_ratio / max(starter_ring_ratio * 2.0, 1e-9)) if has_blue else 0.0

    status = DecisionStatus.ACCEPTED
    if mark is ParticipantMark.AMBIGUOUS:
        status = DecisionStatus.REVIEW

    return ParticipationDetection(
        mark=mark,
        participated=participated,
        starter=starter,
        participation_confidence=round(float(participation_confidence), 4),
        starter_confidence=round(float(starter_confidence), 4),
        status=status,
        blue_pixel_ratio=round(blue_ratio, 6),
        red_pixel_ratio=round(red_ratio, 6),
        red_ring_ratio=round(red_ring_ratio, 6),
    )


def extract_roster(
    normalized_image: np.ndarray,
    *,
    team: str,
    recognizer: HandwritingRecognizer | None,
    template: TemplateSpec = FECABA_V1,
    grid: ParticipationGridSpec = PARTICIPATION_GRID,
    writer_id: str | None = None,
    writer_profile: WriterProfile | None = None,
    min_colored_ink_ratio: float = 0.004,
    acceptance_threshold: float | None = None,
) -> RosterExtraction:
    """Recognize occupied jersey rows without inventing low-confidence players."""

    side = team.upper()
    if side not in {"A", "B"}:
        raise ValueError("team must be A or B")
    if recognizer is None:
        return RosterExtraction()

    block = crop_region(
        normalized_image,
        template.region(f"team_{side.lower()}_jersey"),
    )
    data_top = round(block.shape[0] * grid.header_fraction)
    data = block[data_top:]
    row_height = data.shape[0] / grid.roster_rows
    allowed_labels: Sequence[str] = tuple(str(number) for number in range(100))
    observations: list[RosterRowObservation] = []
    jerseys: list[int] = []
    seen: set[int] = set()

    for index in range(grid.roster_rows):
        top = round(index * row_height)
        bottom = round((index + 1) * row_height)
        interior = _roster_row_interior(data[top:bottom])
        if _colored_ink_ratio(interior) < min_colored_ink_ratio:
            continue

        result = recognizer.recognize(
            interior,
            field_type="jersey",
            allowed_labels=allowed_labels,
            writer_id=writer_id,
        )
        result = apply_writer_profile(result, writer_profile)
        jersey = _recognized_jersey(result.value)
        status = result.status
        if jersey is None or jersey in seen:
            status = DecisionStatus.UNRESOLVED
        elif (
            acceptance_threshold is None
            or result.status is not DecisionStatus.ACCEPTED
            or result.confidence < acceptance_threshold
        ):
            status = DecisionStatus.REVIEW

        observations.append(
            RosterRowObservation(
                row=index + 1,
                jersey=jersey,
                confidence=result.confidence,
                status=status,
            )
        )
        if jersey is not None and status is DecisionStatus.ACCEPTED:
            jerseys.append(jersey)
            seen.add(jersey)

    if not observations:
        status = DecisionStatus.UNRESOLVED
    elif any(item.status is DecisionStatus.UNRESOLVED for item in observations):
        status = DecisionStatus.UNRESOLVED
    elif any(item.status is DecisionStatus.REVIEW for item in observations):
        status = DecisionStatus.REVIEW
    elif not jerseys:
        status = DecisionStatus.UNRESOLVED
    else:
        status = DecisionStatus.ACCEPTED
    return RosterExtraction(jerseys=jerseys, status=status, observations=observations)


def _roster_row_interior(row: np.ndarray) -> np.ndarray:
    if row.size == 0:
        return row
    height, width = row.shape[:2]
    y_margin = max(1, round(height * 0.12))
    x_margin = max(1, round(width * 0.06))
    return row[
        y_margin : max(y_margin + 1, height - y_margin),
        x_margin : max(x_margin + 1, width - x_margin),
    ]


def _colored_ink_ratio(image: np.ndarray) -> float:
    if image.size == 0:
        return 0.0
    rgb = _as_rgb_array(image)
    saturation = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    return float((saturation >= 28).mean())


def _recognized_jersey(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        jersey = int(value)
    except ValueError:
        return None
    return jersey if 0 <= jersey <= 99 else None


def extract_players(
    normalized_image: np.ndarray,
    *,
    team: str,
    jerseys: list[int],
    template: TemplateSpec = FECABA_V1,
    grid: ParticipationGridSpec = PARTICIPATION_GRID,
) -> list[PlayerResult]:
    side = team.upper()
    if side not in {"A", "B"}:
        raise ValueError("team must be A or B")
    if len(jerseys) > grid.roster_rows:
        raise ValueError(
            f"The template supports at most {grid.roster_rows} player rows"
        )

    block = crop_region(
        normalized_image,
        template.region(f"team_{side.lower()}_participation"),
    )
    data_top = round(block.shape[0] * grid.header_fraction)
    data = block[data_top:]
    row_height = data.shape[0] / grid.roster_rows
    players: list[PlayerResult] = []

    for index, jersey in enumerate(jerseys):
        top = round(index * row_height)
        bottom = round((index + 1) * row_height)
        detection = detect_participation(data[top:bottom])
        players.append(
            PlayerResult(
                jersey=jersey,
                participation_mark=detection.mark,
                participated=detection.participated,
                starter=detection.starter,
                participation_confidence=detection.participation_confidence,
                starter_confidence=detection.starter_confidence,
                participation_status=detection.status,
                starter_status=detection.status,
            )
        )
    return players
