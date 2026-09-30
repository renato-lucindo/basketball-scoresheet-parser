from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np

from .models import DecisionStatus


@dataclass(frozen=True, slots=True)
class RecognitionCandidate:
    label: str
    probability: float


@dataclass(slots=True)
class RecognitionResult:
    value: str | None
    confidence: float
    candidates: list[RecognitionCandidate] = field(default_factory=list)
    status: DecisionStatus = DecisionStatus.ACCEPTED


class HandwritingRecognizer(Protocol):
    def recognize(
        self,
        image: np.ndarray,
        *,
        field_type: str,
        allowed_labels: Sequence[str] | None = None,
        writer_id: str | None = None,
    ) -> RecognitionResult: ...


@dataclass(slots=True)
class WriterProfile:
    writer_id: str
    sample_count: int = 0
    label_priors: dict[str, float] = field(default_factory=dict)


def constrain_candidates(
    result: RecognitionResult,
    allowed_labels: Sequence[str],
) -> RecognitionResult:
    allowed = set(allowed_labels)
    filtered = [
        candidate
        for candidate in result.candidates
        if candidate.label in allowed
    ]
    if not filtered:
        return RecognitionResult(
            value=None,
            confidence=0.0,
            candidates=[],
            status=DecisionStatus.REVIEW,
        )

    total = sum(max(candidate.probability, 0.0) for candidate in filtered)
    if total <= 0:
        return RecognitionResult(
            value=None,
            confidence=0.0,
            candidates=filtered,
            status=DecisionStatus.REVIEW,
        )

    normalized = sorted(
        (
            RecognitionCandidate(
                candidate.label,
                max(candidate.probability, 0.0) / total,
            )
            for candidate in filtered
        ),
        key=lambda candidate: candidate.probability,
        reverse=True,
    )
    best = normalized[0]
    return RecognitionResult(
        value=best.label,
        confidence=round(float(best.probability), 4),
        candidates=normalized,
        status=(
            DecisionStatus.ACCEPTED
            if best.probability >= 0.70
            else DecisionStatus.REVIEW
        ),
    )


def apply_writer_profile(
    result: RecognitionResult,
    profile: WriterProfile | None,
    *,
    max_weight: float = 0.30,
    saturation_samples: int = 100,
) -> RecognitionResult:
    """Repondera candidatos usando historico anonimizado do apontador.

    O perfil nunca substitui o modelo global. Seu peso cresce com a quantidade
    de exemplos rotulados e e limitado por max_weight.
    """
    if profile is None or profile.sample_count <= 0 or not result.candidates:
        return result

    weight = min(
        max_weight,
        max_weight * profile.sample_count / max(saturation_samples, 1),
    )
    adjusted: list[RecognitionCandidate] = []
    for candidate in result.candidates:
        prior = max(profile.label_priors.get(candidate.label, 0.0), 1e-4)
        probability = candidate.probability * ((1.0 - weight) + weight * prior)
        adjusted.append(RecognitionCandidate(candidate.label, probability))

    total = sum(candidate.probability for candidate in adjusted)
    if total <= 0:
        return result
    adjusted = sorted(
        (
            RecognitionCandidate(candidate.label, candidate.probability / total)
            for candidate in adjusted
        ),
        key=lambda candidate: candidate.probability,
        reverse=True,
    )
    best = adjusted[0]
    return RecognitionResult(
        value=best.label,
        confidence=round(float(best.probability), 4),
        candidates=adjusted,
        status=(
            DecisionStatus.ACCEPTED
            if best.probability >= 0.70
            else DecisionStatus.REVIEW
        ),
    )


class JerseyRecognitionAdapter:
    def __init__(
        self,
        recognizer: HandwritingRecognizer,
        rosters: dict[str, Sequence[int]],
        *,
        writer_id: str | None = None,
        profile: WriterProfile | None = None,
    ) -> None:
        self.recognizer = recognizer
        self.rosters = {
            side.upper(): tuple(numbers)
            for side, numbers in rosters.items()
        }
        self.writer_id = writer_id
        self.profile = profile

    def recognize(
        self,
        image: np.ndarray,
        *,
        team: str,
    ) -> tuple[int | None, float | None]:
        allowed = [str(number) for number in self.rosters.get(team.upper(), ())]
        result = self.recognizer.recognize(
            image,
            field_type="jersey",
            allowed_labels=allowed or None,
            writer_id=self.writer_id,
        )
        if allowed:
            result = constrain_candidates(result, allowed)
        result = apply_writer_profile(result, self.profile)
        if result.value is None:
            return None, result.confidence
        try:
            return int(result.value), result.confidence
        except ValueError:
            return None, result.confidence
