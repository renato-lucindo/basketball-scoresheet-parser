from __future__ import annotations

from .models import CoreFieldStatus, DecisionStatus, DocumentResult, TeamResult


def assess_core_fields(document: DocumentResult) -> DocumentResult:
    """Record whether every M2 core field is available or needs attention."""

    fields: list[CoreFieldStatus] = []
    for side in ("A", "B"):
        team = document.teams.get(side)
        if team is None:
            fields.extend(_missing_team_fields(side))
            continue
        team_fields = _team_fields(team)
        fields.extend(team_fields)
        if any(field.status is DecisionStatus.UNRESOLVED for field in team_fields):
            team.status = DecisionStatus.UNRESOLVED
        elif any(field.status is DecisionStatus.REVIEW for field in team_fields):
            team.status = DecisionStatus.REVIEW

    document.core_fields = fields
    unresolved = [field for field in fields if field.status is DecisionStatus.UNRESOLVED]
    needs_review = [field for field in fields if field.status is DecisionStatus.REVIEW]
    if unresolved:
        document.status = DecisionStatus.UNRESOLVED
    elif needs_review or document.warnings:
        document.status = DecisionStatus.REVIEW
    else:
        document.status = DecisionStatus.ACCEPTED
    return document


def _team_fields(team: TeamResult) -> list[CoreFieldStatus]:
    side = team.side
    players_available = bool(team.players)
    regular_periods = {period.number: period for period in team.periods if period.number <= 4}
    written_periods_available = all(
        number in regular_periods and regular_periods[number].written_score is not None
        for number in range(1, 5)
    )
    team_foul_periods = {indicator.period for indicator in team.team_fouls}

    scoring_status = _combined_status(event.status for event in team.scoring_events)
    if not team.scoring_events:
        zero_score_game = (
            team.written_final_score == 0
            and written_periods_available
            and all(regular_periods[number].written_score == 0 for number in range(1, 5))
        )
        scoring_status = (
            DecisionStatus.ACCEPTED if zero_score_game else DecisionStatus.UNRESOLVED
        )

    foul_status = _combined_status(
        foul.status
        for player in team.players
        for foul in player.fouls
    )

    return [
        _field(f"teams.{side}.name", team.name is not None, "team name was not extracted"),
        _field(f"teams.{side}.players", players_available, "player roster is empty"),
        _field(
            f"teams.{side}.participation",
            players_available,
            "participation cannot be represented without players",
        ),
        CoreFieldStatus(
            path=f"teams.{side}.starters",
            status=(DecisionStatus.ACCEPTED if len(team.starters) == 5 else DecisionStatus.REVIEW),
            reason=(None if len(team.starters) == 5 else f"expected 5 starters, found {len(team.starters)}"),
        ),
        _field(
            f"teams.{side}.period_scoring",
            written_periods_available,
            "written scores for regular periods were not extracted",
        ),
        CoreFieldStatus(
            path=f"teams.{side}.scoring_events",
            status=scoring_status,
            reason=(None if scoring_status is DecisionStatus.ACCEPTED else "one or more scoring events are unavailable or require review"),
        ),
        CoreFieldStatus(
            path=f"teams.{side}.individual_fouls",
            status=(foul_status if players_available else DecisionStatus.UNRESOLVED),
            reason=(None if players_available and foul_status is DecisionStatus.ACCEPTED else "individual foul data is unavailable or requires review"),
        ),
        _field(
            f"teams.{side}.team_fouls",
            all(number in team_foul_periods for number in range(1, 5)),
            "team-foul indicators for all regular periods were not extracted",
        ),
        _field(
            f"teams.{side}.final_score",
            team.written_final_score is not None,
            "written final score was not extracted",
        ),
    ]


def _field(path: str, available: bool, missing_reason: str) -> CoreFieldStatus:
    return CoreFieldStatus(
        path=path,
        status=DecisionStatus.ACCEPTED if available else DecisionStatus.UNRESOLVED,
        reason=None if available else missing_reason,
    )


def _combined_status(statuses) -> DecisionStatus:
    values = list(statuses)
    if any(status is DecisionStatus.UNRESOLVED for status in values):
        return DecisionStatus.UNRESOLVED
    if any(status is DecisionStatus.REVIEW for status in values):
        return DecisionStatus.REVIEW
    return DecisionStatus.ACCEPTED


def _missing_team_fields(side: str) -> list[CoreFieldStatus]:
    return [
        CoreFieldStatus(
            path=f"teams.{side}.{name}",
            status=DecisionStatus.UNRESOLVED,
            reason="team result is missing",
        )
        for name in (
            "name",
            "players",
            "participation",
            "starters",
            "period_scoring",
            "scoring_events",
            "individual_fouls",
            "team_fouls",
            "final_score",
        )
    ]
