from __future__ import annotations

from .models import DecisionStatus, TeamResult


def validate_team(team: TeamResult) -> TeamResult:
    warnings: list[str] = []

    starters = team.starters
    if len(starters) != 5:
        warnings.append(f"Foram detectados {len(starters)} titulares; esperado: 5")

    for player in team.players:
        if player.starter and not player.participated:
            warnings.append(f"Camisa {player.jersey}: titular sem participacao")
        if (player.points > 0 or player.fouls) and not player.participated:
            warnings.append(
                f"Camisa {player.jersey}: possui evento de jogo, mas nao ha X de participacao"
            )

    event_score = sum(event.points for event in team.scoring_events)
    if team.scoring_events and event_score != team.calculated_score:
        warnings.append(
            f"Pontuacao por eventos ({event_score}) difere do placar calculado ({team.calculated_score})"
        )

    if team.written_final_score is not None and team.calculated_score != team.written_final_score:
        warnings.append(
            f"Placar calculado ({team.calculated_score}) difere do placar escrito ({team.written_final_score})"
        )

    team.warnings = warnings
    team.status = DecisionStatus.REVIEW if warnings else DecisionStatus.ACCEPTED
    return team
