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
    period_candidates_available = all(
        number in regular_periods
        and regular_periods[number].written_score_candidate is not None
        for number in range(1, 5)
    )
    if written_periods_available:
        period_scoring_status = _combined_status(
            regular_periods[number].status for number in range(1, 5)
        )
    elif period_candidates_available:
        period_scoring_status = _combined_status(
            regular_periods[number].status for number in range(1, 5)
        )
        if period_scoring_status is DecisionStatus.ACCEPTED:
            period_scoring_status = DecisionStatus.REVIEW
    else:
        period_scoring_status = DecisionStatus.UNRESOLVED
    team_foul_periods = {indicator.period for indicator in team.team_fouls}
    participation_status = _combined_status(
        player.participation_status for player in team.players
    )
    starter_status = _combined_status(player.starter_status for player in team.players)
    if not players_available:
        participation_status = DecisionStatus.UNRESOLVED
        starter_status = DecisionStatus.UNRESOLVED
    elif len(team.starters) != 5 and starter_status is DecisionStatus.ACCEPTED:
        starter_status = DecisionStatus.REVIEW

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
    if team.written_final_score is not None:
        final_score_status = team.written_final_score_status
    elif team.written_final_score_candidate is not None:
        final_score_status = team.written_final_score_status
        if final_score_status is DecisionStatus.ACCEPTED:
            final_score_status = DecisionStatus.REVIEW
    else:
        final_score_status = DecisionStatus.UNRESOLVED

    return [
        _field(f"teams.{side}.name", team.name is not None, "team name was not extracted"),
        CoreFieldStatus(
            path=f"teams.{side}.players",
            status=(team.roster_status if players_available else DecisionStatus.UNRESOLVED),
            reason=(None if players_available and team.roster_status is DecisionStatus.ACCEPTED else "player roster is empty or contains unresolved rows"),
        ),
        CoreFieldStatus(
            path=f"teams.{side}.participation",
            status=participation_status,
            reason=(None if participation_status is DecisionStatus.ACCEPTED else "one or more participation marks are unavailable or require review"),
        ),
        CoreFieldStatus(
            path=f"teams.{side}.starters",
            status=starter_status,
            reason=(None if starter_status is DecisionStatus.ACCEPTED else f"starter evidence requires review; detected {len(team.starters)} starters"),
        ),
        CoreFieldStatus(
            path=f"teams.{side}.period_scoring",
            status=period_scoring_status,
            reason=(None if period_scoring_status is DecisionStatus.ACCEPTED else "written period scores are unavailable or require review"),
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
        CoreFieldStatus(
            path=f"teams.{side}.final_score",
            status=final_score_status,
            reason=(None if final_score_status is DecisionStatus.ACCEPTED else "written final score is unavailable or requires review"),
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
