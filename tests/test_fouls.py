import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from sumula_reader.fouls import (
    TeamFoulCellObservation,
    TeamFoulCellKind,
    _reconcile_team_foul_cells,
    analyze_player_foul_row,
    classify_team_foul_cell,
    detect_half_separator,
    extract_team_foul_indicators,
    parse_foul_symbol,
)
from sumula_reader.models import DecisionStatus, FoulKind
from sumula_reader.recognition import RecognitionCandidate, RecognitionResult
from sumula_reader.scoring import InkColor


RED = (190, 45, 45)
BLUE = (35, 65, 185)


def blank(width=100, height=70):
    return np.full((height, width, 3), 255, dtype=np.uint8)


class FixedFoulRecognizer:
    def __init__(self, label="P2", confidence=0.9):
        self.label = label
        self.confidence = confidence

    def recognize(self, image):
        return RecognitionResult(
            value=self.label,
            confidence=self.confidence,
            candidates=[RecognitionCandidate(self.label, self.confidence)],
            status=DecisionStatus.ACCEPTED,
        )


class FixedDecisionEngine:
    def __init__(self, choice="x", confidence=0.95):
        self.choice = choice
        self.confidence = confidence
        self.calls = []

    def classify_team_foul(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choice=self.choice,
            confidence=self.confidence,
        )


class FoulTests(unittest.TestCase):
    def test_team_foul_x(self):
        image = Image.fromarray(blank())
        draw = ImageDraw.Draw(image)
        draw.line((20, 12, 80, 58), fill=RED, width=5)
        draw.line((80, 12, 20, 58), fill=RED, width=5)
        result = classify_team_foul_cell(np.asarray(image))
        self.assertEqual(result.kind, TeamFoulCellKind.X)

    def test_team_foul_unused_double_horizontal(self):
        image = Image.fromarray(blank())
        draw = ImageDraw.Draw(image)
        draw.line((12, 27, 88, 27), fill=BLUE, width=4)
        draw.line((12, 42, 88, 42), fill=BLUE, width=4)
        result = classify_team_foul_cell(np.asarray(image))
        self.assertEqual(result.kind, TeamFoulCellKind.UNUSED)

    def test_team_foul_unused_horizontal_correction_over_x(self):
        image = Image.fromarray(blank())
        draw = ImageDraw.Draw(image)
        draw.line((20, 12, 80, 58), fill=RED, width=5)
        draw.line((80, 12, 20, 58), fill=RED, width=5)
        draw.line((6, 27, 94, 27), fill=RED, width=5)
        draw.line((6, 42, 94, 42), fill=RED, width=5)
        result = classify_team_foul_cell(np.asarray(image))
        self.assertEqual(result.kind, TeamFoulCellKind.UNUSED)

    def test_team_foul_sequence_infers_weak_cell_before_later_x(self):
        cells = [
            TeamFoulCellObservation(TeamFoulCellKind.X, InkColor.BLUE, 0.9, 0.2, 1.2),
            TeamFoulCellObservation(TeamFoulCellKind.X, InkColor.BLUE, 0.9, 0.2, 1.2),
            TeamFoulCellObservation(TeamFoulCellKind.AMBIGUOUS, InkColor.BLUE, 0.4, 0.03, 2.3),
            TeamFoulCellObservation(TeamFoulCellKind.X, InkColor.BLUE, 0.8, 0.15, 1.4),
        ]
        count, review = _reconcile_team_foul_cells(cells)
        self.assertEqual(count, 4)
        self.assertTrue(review)

    def test_team_foul_sequence_accepts_x_then_unused(self):
        cells = [
            TeamFoulCellObservation(TeamFoulCellKind.X, InkColor.RED, 0.9, 0.2, 1.2),
            TeamFoulCellObservation(TeamFoulCellKind.X, InkColor.RED, 0.9, 0.2, 1.2),
            TeamFoulCellObservation(TeamFoulCellKind.UNUSED, InkColor.RED, 0.9, 0.1, 4.0),
            TeamFoulCellObservation(TeamFoulCellKind.UNUSED, InkColor.RED, 0.9, 0.1, 4.0),
        ]
        count, review = _reconcile_team_foul_cells(cells)
        self.assertEqual(count, 2)
        self.assertFalse(review)

    def test_jev_resolves_only_ambiguous_team_foul_cell(self):
        x = TeamFoulCellObservation(
            TeamFoulCellKind.X,
            InkColor.BLUE,
            0.9,
            0.2,
            1.2,
        )
        ambiguous = TeamFoulCellObservation(
            TeamFoulCellKind.AMBIGUOUS,
            InkColor.BLUE,
            0.4,
            0.03,
            2.3,
        )
        empty = TeamFoulCellObservation(
            TeamFoulCellKind.EMPTY,
            InkColor.UNKNOWN,
            1.0,
            0.0,
            1.0,
        )
        observations = [x, x, ambiguous, x] + [empty] * 12
        engine = FixedDecisionEngine()

        with patch(
            "sumula_reader.fouls.classify_team_foul_cell",
            side_effect=observations,
        ):
            indicators = extract_team_foul_indicators(
                blank(100, 100),
                team="A",
                decision_engine=engine,
            )

        self.assertEqual(indicators[0].x_count, 4)
        self.assertEqual(indicators[0].status, DecisionStatus.ACCEPTED)
        self.assertEqual(len(engine.calls), 1)
        self.assertEqual(engine.calls[0]["period"], 1)
        self.assertEqual(engine.calls[0]["slot"], 3)

    def test_half_separator_ignores_thickness(self):
        for thickness in (2, 8):
            image = Image.fromarray(blank(250, 50))
            draw = ImageDraw.Draw(image)
            x = 100  # limite entre slot 2 e slot 3
            draw.line((x, 0, x, 49), fill=BLUE, width=thickness)
            result = detect_half_separator(np.asarray(image))
            self.assertEqual(result.first_half_slots, 2)

    def test_player_fouls_map_half_and_color_to_period(self):
        image = Image.fromarray(blank(250, 50))
        draw = ImageDraw.Draw(image)
        draw.line((100, 0, 100, 49), fill=BLUE, width=4)

        # Slot 1 vermelho -> Q1
        draw.text((15, 12), "P", fill=RED)
        # Slot 2 azul -> Q2
        draw.text((65, 12), "P", fill=BLUE)
        # Slot 3 vermelho -> Q3
        draw.text((115, 12), "P", fill=RED)
        # Slot 4 azul -> Q4
        draw.text((165, 12), "P", fill=BLUE)

        fouls = analyze_player_foul_row(
            np.asarray(image),
            team="A",
            jersey=11,
        )
        self.assertEqual([foul.slot for foul in fouls], [1, 2, 3, 4])
        self.assertEqual([foul.period for foul in fouls], [1, 2, 3, 4])
        self.assertTrue(all(foul.status is DecisionStatus.REVIEW for foul in fouls))

    def test_player_foul_recognizer_sets_kind_and_free_throws(self):
        image = Image.fromarray(blank(250, 50))
        draw = ImageDraw.Draw(image)
        draw.line((100, 0, 100, 49), fill=BLUE, width=4)
        draw.text((15, 12), "P", fill=RED)

        fouls = analyze_player_foul_row(
            np.asarray(image),
            team="A",
            jersey=11,
            recognizer=FixedFoulRecognizer("P2", 0.9),
        )
        self.assertEqual(len(fouls), 1)
        self.assertEqual(fouls[0].kind, FoulKind.PERSONAL)
        self.assertEqual(fouls[0].free_throws, 2)
        self.assertEqual(fouls[0].raw_symbol, "P2")
        self.assertEqual(fouls[0].status, DecisionStatus.ACCEPTED)

    def test_parse_foul_symbol(self):
        self.assertEqual(parse_foul_symbol("P2"), (FoulKind.PERSONAL, 2))
        self.assertEqual(
            parse_foul_symbol(" u1 "),
            (FoulKind.UNSPORTSMANLIKE, 1),
        )
        self.assertEqual(parse_foul_symbol("?"), (FoulKind.UNKNOWN, None))


if __name__ == "__main__":
    unittest.main()
