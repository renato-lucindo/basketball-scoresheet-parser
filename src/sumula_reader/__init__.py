from .models import (
    DecisionStatus,
    DocumentResult,
    FoulEvent,
    FoulKind,
    ParticipantMark,
    PeriodResult,
    PeriodType,
    PlayerResult,
    ScoringEvent,
    ShotType,
    TeamFoulIndicator,
    TeamResult,
)
from .template import FECABA_V1, get_template
from .scoring import (
    InkColor,
    ScoreMarkKind,
    classify_score_mark,
    detect_jersey_circle,
    extract_scoring_events,
)
from .fouls import (
    TeamFoulCellKind,
    analyze_player_foul_row,
    classify_team_foul_cell,
    detect_half_separator,
    extract_player_fouls,
    extract_team_foul_indicators,
)

__all__ = [
    "DecisionStatus",
    "DocumentResult",
    "FoulEvent",
    "FoulKind",
    "ParticipantMark",
    "PeriodResult",
    "PeriodType",
    "PlayerResult",
    "ScoringEvent",
    "ShotType",
    "TeamFoulIndicator",
    "TeamResult",
    "FECABA_V1",
    "get_template",
    "InkColor",
    "ScoreMarkKind",
    "classify_score_mark",
    "detect_jersey_circle",
    "extract_scoring_events",
    "TeamFoulCellKind",
    "analyze_player_foul_row",
    "classify_team_foul_cell",
    "detect_half_separator",
    "extract_player_fouls",
    "extract_team_foul_indicators",
]

__version__ = "0.1.0"
