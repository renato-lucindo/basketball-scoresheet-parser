from __future__ import annotations

from collections import defaultdict

from .models import (
    DecisionStatus,
    DocumentResult,
    PeriodResult,
    PeriodType,
    ShotType,
    TeamResult,
)
from .validation import validate_team


def reconcile_team(team: TeamResult) -> TeamResult:
    warnings: list[str] = []
    events = sorted(team.scoring_events, key=lambda event: event.running_score)
    source_periods = {period.number: period for period in team.periods}

    team.warnings = []
    for player in team.players:
        player.points = 0
        player.free_throws_made = 0
        player.two_points_made = 0
        player.three_points_made = 0

    previous_score = 0
    period_points: dict[int, int] = defaultdict(int)
    players_by_jersey = {player.jersey: player for player in team.players}

    for event in events:
        if event.points <= 0 or event.shot_type is ShotType.AMBIGUOUS:
            event.status = DecisionStatus.REVIEW
            warnings.append(
                f"Score {event.running_score}: ambiguous shot type"
            )
            continue

        delta = event.running_score - previous_score
        if delta != event.points:
            event.status = DecisionStatus.REVIEW
            warnings.append(
                f"Score {event.running_score}: jump of {delta} is incompatible "
                f"with a {event.points}-point event"
            )
        previous_score = max(previous_score, event.running_score)

        if event.period is not None:
            period_points[event.period] += event.points
        else:
            event.status = DecisionStatus.UNRESOLVED
            warnings.append(
                f"Score {event.running_score}: period is unresolved"
            )

        if event.jersey is None:
            event.status = DecisionStatus.UNRESOLVED
            warnings.append(
                f"Score {event.running_score}: jersey is unresolved"
            )
            continue
        player = players_by_jersey.get(event.jersey)
        if player is None:
            warnings.append(
                f"Score {event.running_score}: jersey {event.jersey} "
                "is not in the supplied roster"
            )
            event.status = DecisionStatus.REVIEW
            continue
        player.points += event.points
        if event.shot_type is ShotType.FREE_THROW:
            player.free_throws_made += 1
        elif event.shot_type is ShotType.TWO_POINT:
            player.two_points_made += 1
        elif event.shot_type is ShotType.THREE_POINT:
            player.three_points_made += 1

    team.calculated_score = sum(event.points for event in events if event.points > 0)
    team.periods = [
        PeriodResult(
            number=period,
            period_type=(
                PeriodType.REGULAR if period <= 4 else PeriodType.OVERTIME
            ),
            score=period_points.get(period, 0),
            written_score=(
                source_periods[period].written_score
                if period in source_periods
                else None
            ),
            written_score_candidate=(
                source_periods[period].written_score_candidate
                if period in source_periods
                else None
            ),
            confidence=(
                source_periods[period].confidence
                if period in source_periods
                else None
            ),
            status=(
                source_periods[period].status
                if period in source_periods
                else DecisionStatus.UNRESOLVED
            ),
        )
        for period in range(1, max(4, max(period_points, default=4)) + 1)
    ]

    derived_team_fouls: dict[int, int] = defaultdict(int)
    for player in team.players:
        for foul in player.fouls:
            if (
                foul.period is not None
                and foul.counts_as_team_foul
                and not foul.cancelled_penalty
                and not foul.fighting
            ):
                derived_team_fouls[foul.period] += 1
    for indicator in team.team_fouls:
        derived = derived_team_fouls.get(indicator.period, 0)
        matches = (
            derived >= 4
            if indicator.x_count == 4
            else derived == indicator.x_count
        )
        if not matches:
            warnings.append(
                f"Q{indicator.period}: derived team fouls ({derived}) "
                f"differ from the X boxes ({indicator.x_count})"
            )

    validate_team(team)
    team.warnings = warnings + team.warnings
    if team.warnings:
        team.status = DecisionStatus.REVIEW
    return team


def reconcile_document(document: DocumentResult) -> DocumentResult:
    warnings: list[str] = []
    for side, team in document.teams.items():
        reconcile_team(team)
        for warning in team.warnings:
            warnings.append(f"Team {side}: {warning}")

    document.warnings = warnings
    document.status = (
        DecisionStatus.REVIEW if warnings else DecisionStatus.ACCEPTED
    )
    return document
