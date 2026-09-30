from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class DecisionStatus(StrEnum):
    ACCEPTED = "accepted"
    REVIEW = "review"
    UNRESOLVED = "unresolved"


class PeriodType(StrEnum):
    REGULAR = "regular"
    OVERTIME = "overtime"


class ShotType(StrEnum):
    FREE_THROW = "free_throw"
    TWO_POINT = "two_point"
    THREE_POINT = "three_point"
    AMBIGUOUS = "ambiguous"


class FoulKind(StrEnum):
    PERSONAL = "personal"
    TECHNICAL = "technical"
    UNSPORTSMANLIKE = "unsportsmanlike"
    DISQUALIFYING = "disqualifying"
    UNKNOWN = "unknown"


class ParticipantMark(StrEnum):
    NONE = "none"
    BLUE_X = "blue_x"
    RED_X = "red_x"
    BLUE_X_RED_CIRCLE = "blue_x_red_circle"
    AMBIGUOUS = "ambiguous"


@dataclass(slots=True)
class Evidence:
    observed: Any = None
    interpreted: Any = None
    confidence: float | None = None
    status: DecisionStatus = DecisionStatus.ACCEPTED
    requires_review: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DocumentMetadata:
    template: str = "fecaba_v1"
    game_date: str | None = None
    competition: str | None = None
    category: str | None = None
    game_id: str | None = None
    apontador_raw: str | None = None
    writer_id: str | None = None
    writer_known: bool = False


@dataclass(slots=True)
class ScoringEvent:
    team: str
    period: int
    running_score: int
    jersey: int | None
    shot_type: ShotType
    points: int
    confidence: float | None = None
    status: DecisionStatus = DecisionStatus.ACCEPTED


@dataclass(slots=True)
class FoulEvent:
    team: str
    jersey: int
    slot: int
    period: int | None
    kind: FoulKind
    free_throws: int | None = None
    raw_symbol: str | None = None
    ink_color: str | None = None
    confidence: float | None = None
    status: DecisionStatus = DecisionStatus.ACCEPTED


@dataclass(slots=True)
class TeamFoulIndicator:
    period: int
    x_count: int
    count_is_capped: bool = False
    minimum_team_fouls: int | None = None
    confidence: float | None = None
    status: DecisionStatus = DecisionStatus.ACCEPTED

    def __post_init__(self) -> None:
        if not 0 <= self.x_count <= 4:
            raise ValueError("x_count deve estar entre 0 e 4")
        if self.minimum_team_fouls is None:
            self.minimum_team_fouls = self.x_count
        if self.x_count == 4:
            self.count_is_capped = True
            self.minimum_team_fouls = 4


@dataclass(slots=True)
class PlayerResult:
    jersey: int
    registered: bool = True
    participation_mark: ParticipantMark = ParticipantMark.NONE
    participated: bool = False
    starter: bool = False
    participation_confidence: float | None = None
    starter_confidence: float | None = None
    fouls: list[FoulEvent] = field(default_factory=list)
    points: int = 0
    free_throws_made: int = 0
    two_points_made: int = 0
    three_points_made: int = 0

    def __post_init__(self) -> None:
        if self.starter and not self.participated:
            raise ValueError("starter=true exige participated=true")


@dataclass(slots=True)
class PeriodResult:
    number: int
    period_type: PeriodType = PeriodType.REGULAR
    score: int | None = None
    written_score: int | None = None
    confidence: float | None = None
    status: DecisionStatus = DecisionStatus.ACCEPTED


@dataclass(slots=True)
class TeamResult:
    side: str
    name: str | None = None
    players: list[PlayerResult] = field(default_factory=list)
    periods: list[PeriodResult] = field(default_factory=list)
    scoring_events: list[ScoringEvent] = field(default_factory=list)
    team_fouls: list[TeamFoulIndicator] = field(default_factory=list)
    calculated_score: int = 0
    written_final_score: int | None = None
    status: DecisionStatus = DecisionStatus.ACCEPTED
    warnings: list[str] = field(default_factory=list)

    @property
    def starters(self) -> list[int]:
        return [player.jersey for player in self.players if player.starter]

    @property
    def participants(self) -> list[int]:
        return [player.jersey for player in self.players if player.participated]


@dataclass(slots=True)
class DocumentResult:
    schema_version: str = "0.1"
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)
    teams: dict[str, TeamResult] = field(default_factory=dict)
    status: DecisionStatus = DecisionStatus.ACCEPTED
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value
