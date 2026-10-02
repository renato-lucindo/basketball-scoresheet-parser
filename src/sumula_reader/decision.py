from __future__ import annotations

from typing import Any, Mapping, Protocol


class DecisionAnswer(Protocol):
    choice: str
    confidence: float


class DecisionEngine(Protocol):
    def classify_team_foul(
        self,
        *,
        period: int,
        slot: int,
        evidence: Mapping[str, Any],
        context: Mapping[str, Any] | None = None,
    ) -> DecisionAnswer: ...

    def classify_scoring_mark(
        self,
        *,
        team: str,
        running_score: int,
        evidence: Mapping[str, Any],
        context: Mapping[str, Any] | None = None,
    ) -> DecisionAnswer: ...
