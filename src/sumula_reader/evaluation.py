from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    expected: str
    predicted: str | None
    confidence: float
    document_id: str
    writer_known: bool = False

    @property
    def correct(self) -> bool:
        return self.predicted == self.expected


@dataclass(frozen=True, slots=True)
class ThresholdMetrics:
    threshold: float
    total: int
    accepted: int
    reviewed: int
    accepted_errors: int
    global_accuracy: float
    automation_rate: float
    accepted_error_rate: float


def evaluate_threshold(
    records: Iterable[PredictionRecord],
    threshold: float,
) -> ThresholdMetrics:
    samples = list(records)
    accepted = [
        record
        for record in samples
        if record.predicted is not None and record.confidence >= threshold
    ]
    accepted_errors = sum(not record.correct for record in accepted)
    correct = sum(record.correct for record in samples)
    total = len(samples)
    accepted_count = len(accepted)
    return ThresholdMetrics(
        threshold=threshold,
        total=total,
        accepted=accepted_count,
        reviewed=total - accepted_count,
        accepted_errors=accepted_errors,
        global_accuracy=(correct / total if total else 0.0),
        automation_rate=(accepted_count / total if total else 0.0),
        accepted_error_rate=(
            accepted_errors / accepted_count if accepted_count else 0.0
        ),
    )


def choose_acceptance_threshold(
    records: Iterable[PredictionRecord],
    *,
    max_accepted_error_rate: float = 0.01,
    min_automation_rate: float = 0.10,
) -> ThresholdMetrics | None:
    """Escolhe o menor limiar que respeita o gate de erro aceito.

    Entre limiares validos, prioriza maior automacao; em empate, menor erro.
    """
    samples = list(records)
    if not samples:
        return None
    thresholds = sorted(
        {0.0, 1.0, *(record.confidence for record in samples)}
    )
    candidates: list[ThresholdMetrics] = []
    for threshold in thresholds:
        metrics = evaluate_threshold(samples, threshold)
        if (
            metrics.automation_rate >= min_automation_rate
            and metrics.accepted_error_rate <= max_accepted_error_rate
        ):
            candidates.append(metrics)
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            item.automation_rate,
            -item.accepted_error_rate,
            -item.threshold,
        ),
    )


def compare_known_unknown_writers(
    records: Iterable[PredictionRecord],
    threshold: float,
) -> dict[str, ThresholdMetrics]:
    samples = list(records)
    return {
        "known_writer": evaluate_threshold(
            (record for record in samples if record.writer_known),
            threshold,
        ),
        "unknown_writer": evaluate_threshold(
            (record for record in samples if not record.writer_known),
            threshold,
        ),
    }
