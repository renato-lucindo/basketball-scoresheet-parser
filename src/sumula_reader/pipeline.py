from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .completeness import assess_core_fields
from .decision import DecisionEngine
from .fouls import extract_player_foul_data, extract_team_foul_indicators
from .imaging import load_document, normalize_document
from .models import DocumentMetadata, DocumentResult, TeamResult
from .participation import extract_players
from .recognition import (
    FoulRecognitionAdapter,
    HandwritingRecognizer,
    JerseyRecognitionAdapter,
    WriterProfile,
)
from .reconcile import reconcile_document
from .scoring import extract_scoring_events
from .template import FECABA_V1, TemplateSpec


@dataclass(slots=True)
class AnalysisContext:
    rosters: dict[str, list[int]] = field(default_factory=dict)
    writer_id: str | None = None
    writer_known: bool = False
    writer_profile: WriterProfile | None = None


def analyze_image(
    normalized_image: np.ndarray,
    *,
    context: AnalysisContext,
    template: TemplateSpec = FECABA_V1,
    handwriting: HandwritingRecognizer | None = None,
    decision_engine: DecisionEngine | None = None,
) -> DocumentResult:
    rosters = {
        side.upper(): numbers
        for side, numbers in context.rosters.items()
    }
    jersey_recognizer = (
        JerseyRecognitionAdapter(
            handwriting,
            rosters,
            writer_id=context.writer_id,
            profile=context.writer_profile,
        )
        if handwriting is not None
        else None
    )
    foul_recognizer = (
        FoulRecognitionAdapter(
            handwriting,
            writer_id=context.writer_id,
            profile=context.writer_profile,
        )
        if handwriting is not None
        else None
    )
    scoring_events = extract_scoring_events(
        normalized_image,
        template=template,
        recognizer=jersey_recognizer,
        decision_engine=decision_engine,
    )

    teams: dict[str, TeamResult] = {}
    for side in ("A", "B"):
        jerseys = list(rosters.get(side, ()))
        players = extract_players(
            normalized_image,
            team=side,
            jerseys=jerseys,
            template=template,
        )
        fouls, foul_terminals = extract_player_foul_data(
            normalized_image,
            team=side,
            jerseys=jerseys,
            template=template,
            recognizer=foul_recognizer,
        )
        players_by_jersey = {player.jersey: player for player in players}
        for foul in fouls:
            player = players_by_jersey.get(foul.jersey)
            if player is not None:
                player.fouls.append(foul)
        for terminal in foul_terminals:
            player = players_by_jersey.get(terminal.jersey)
            if player is not None:
                player.foul_terminals.append(terminal)

        teams[side] = TeamResult(
            side=side,
            players=players,
            scoring_events=[
                event for event in scoring_events if event.team == side
            ],
            team_fouls=extract_team_foul_indicators(
                normalized_image,
                team=side,
                template=template,
                decision_engine=decision_engine,
            ),
        )

    result = DocumentResult(
        metadata=DocumentMetadata(
            template=template.template_id,
            writer_id=context.writer_id,
            writer_known=context.writer_known,
        ),
        teams=teams,
    )
    return assess_core_fields(reconcile_document(result))


def analyze_path(
    path: str | Path,
    *,
    context: AnalysisContext,
    template: TemplateSpec = FECABA_V1,
    handwriting: HandwritingRecognizer | None = None,
    decision_engine: DecisionEngine | None = None,
    dpi: int = 250,
) -> DocumentResult:
    image = load_document(path, dpi=dpi)
    normalized = normalize_document(image, template)
    return analyze_image(
        normalized.image,
        context=context,
        template=template,
        handwriting=handwriting,
        decision_engine=decision_engine,
    )
