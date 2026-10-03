from __future__ import annotations

from .models import DecisionStatus, TeamResult


def validate_team(team: TeamResult) -> TeamResult:
    warnings: list[str] = []

    starters = team.starters
    if len(starters) != 5:
        warnings.append(f"Detected {len(starters)} starters; expected 5")

    for player in team.players:
        if player.starter and not player.participated:
            warnings.append(f"Jersey {player.jersey}: starter without participation")
        if (player.points > 0 or player.fouls) and not player.participated:
            warnings.append(
                f"Jersey {player.jersey}: has a game event without a participation mark"
            )

    event_score = sum(event.points for event in team.scoring_events)
    if team.scoring_events and event_score != team.calculated_score:
        warnings.append(
            f"Event score ({event_score}) differs from calculated score ({team.calculated_score})"
        )

    if team.written_final_score is not None and team.calculated_score != team.written_final_score:
        warnings.append(
            f"Calculated score ({team.calculated_score}) differs from written score ({team.written_final_score})"
        )

    for period in team.periods:
        if period.written_score is not None and period.score != period.written_score:
            warnings.append(
                f"Q{period.number}: calculated period score ({period.score}) differs from written score ({period.written_score})"
            )

    team.warnings = warnings
    team.status = DecisionStatus.REVIEW if warnings else DecisionStatus.ACCEPTED
    return team
