from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scoring import colored_ink_masks


@dataclass(slots=True)
class AlterationObservation:
    suspected: bool
    confidence: float
    ink_ratio: float
    red_ratio: float
    blue_ratio: float
    transition_ratio: float
    reasons: tuple[str, ...]


def detect_alteration(
    image: np.ndarray,
    *,
    dense_ink_threshold: float = 0.22,
    mixed_color_threshold: float = 0.018,
    transition_threshold: float = 0.34,
) -> AlterationObservation:
    """Sinaliza celulas potencialmente rasuradas/sobrepostas."""
    red_mask, blue_mask = colored_ink_masks(image)
    mask = red_mask | blue_mask
    ink_ratio = float(mask.mean())
    red_ratio = float(red_mask.mean())
    blue_ratio = float(blue_mask.mean())

    if mask.shape[0] > 1 and mask.shape[1] > 1:
        horizontal = np.not_equal(mask[:, 1:], mask[:, :-1]).mean()
        vertical = np.not_equal(mask[1:, :], mask[:-1, :]).mean()
        transition_ratio = float((horizontal + vertical) / 2.0)
    else:
        transition_ratio = 0.0

    reasons: list[str] = []
    scores: list[float] = []
    if ink_ratio >= dense_ink_threshold:
        reasons.append("dense_ink")
        scores.append(min(1.0, ink_ratio / dense_ink_threshold - 0.25))
    if (
        red_ratio >= mixed_color_threshold
        and blue_ratio >= mixed_color_threshold
    ):
        reasons.append("mixed_red_blue")
        scores.append(
            min(
                1.0,
                min(red_ratio, blue_ratio) / mixed_color_threshold * 0.7,
            )
        )
    if transition_ratio >= transition_threshold:
        reasons.append("high_stroke_complexity")
        scores.append(min(1.0, transition_ratio / transition_threshold * 0.7))

    confidence = max(scores, default=0.0)
    return AlterationObservation(
        suspected=confidence >= 0.65,
        confidence=round(float(confidence), 4),
        ink_ratio=round(ink_ratio, 6),
        red_ratio=round(red_ratio, 6),
        blue_ratio=round(blue_ratio, 6),
        transition_ratio=round(transition_ratio, 6),
        reasons=tuple(reasons),
    )
