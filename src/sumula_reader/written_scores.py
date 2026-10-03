from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .imaging import crop_region
from .models import DecisionStatus, PeriodResult
from .recognition import HandwritingRecognizer, WriterProfile, apply_writer_profile
from .template import FECABA_V1, NormalizedRect, TemplateSpec


@dataclass(frozen=True, slots=True)
class PeriodScoreCell:
    team: str
    period: int
    rect: NormalizedRect


PERIOD_SCORE_CELLS = (
    PeriodScoreCell("A", 1, NormalizedRect(0.245, 0.22, 0.080, 0.30)),
    PeriodScoreCell("B", 1, NormalizedRect(0.345, 0.22, 0.080, 0.30)),
    PeriodScoreCell("A", 2, NormalizedRect(0.690, 0.22, 0.090, 0.30)),
    PeriodScoreCell("B", 2, NormalizedRect(0.820, 0.22, 0.090, 0.30)),
    PeriodScoreCell("A", 3, NormalizedRect(0.245, 0.53, 0.080, 0.32)),
    PeriodScoreCell("B", 3, NormalizedRect(0.345, 0.53, 0.080, 0.32)),
    PeriodScoreCell("A", 4, NormalizedRect(0.690, 0.53, 0.090, 0.32)),
    PeriodScoreCell("B", 4, NormalizedRect(0.820, 0.53, 0.090, 0.32)),
)


def extract_period_score_candidates(
    normalized_image: np.ndarray,
    *,
    recognizer: HandwritingRecognizer | None,
    template: TemplateSpec = FECABA_V1,
    writer_id: str | None = None,
    writer_profile: WriterProfile | None = None,
    acceptance_threshold: float | None = None,
    min_colored_ink_ratio: float = 0.004,
) -> dict[str, list[PeriodResult]]:
    """Read written period-score cells while preserving uncalibrated candidates."""

    results: dict[str, list[PeriodResult]] = {"A": [], "B": []}
    if recognizer is None:
        return results

    block = crop_region(normalized_image, template.region("period_scores"))
    labels: Sequence[str] = tuple(str(number) for number in range(100))
    for cell in PERIOD_SCORE_CELLS:
        crop = crop_region(block, cell.rect)
        if _colored_ink_ratio(crop) < min_colored_ink_ratio:
            results[cell.team].append(
                PeriodResult(number=cell.period, status=DecisionStatus.UNRESOLVED)
            )
            continue

        recognition = recognizer.recognize(
            crop,
            field_type="jersey",
            allowed_labels=labels,
            writer_id=writer_id,
        )
        recognition = apply_writer_profile(recognition, writer_profile)
        candidate = _score_value(recognition.value)
        if candidate is None:
            status = DecisionStatus.UNRESOLVED
        elif (
            acceptance_threshold is not None
            and recognition.status is DecisionStatus.ACCEPTED
            and recognition.confidence >= acceptance_threshold
        ):
            status = DecisionStatus.ACCEPTED
        else:
            status = DecisionStatus.REVIEW
        results[cell.team].append(
            PeriodResult(
                number=cell.period,
                written_score=(candidate if status is DecisionStatus.ACCEPTED else None),
                written_score_candidate=candidate,
                confidence=recognition.confidence,
                status=status,
            )
        )
    return results


def _colored_ink_ratio(image: np.ndarray) -> float:
    if image.size == 0:
        return 0.0
    rgb = np.asarray(image)[..., :3].astype(np.int16, copy=False)
    saturation = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    return float((saturation >= 28).mean())


def _score_value(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        score = int(value)
    except ValueError:
        return None
    return score if 0 <= score <= 99 else None
