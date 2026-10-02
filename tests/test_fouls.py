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
    classify_foul_terminal,
    classify_team_foul_cell,
    clean_player_foul_cell,
    detect_half_separator,
    detect_player_foul_grid,
    extract_team_foul_indicators,
    parse_foul_symbol,
    parse_foul_symbol_details,
)
from sumula_reader.models import DecisionStatus, FoulKind, FoulTerminalKind
from sumula_reader.recognition import RecognitionCandidate, RecognitionResult
from sumula_reader.scoring import InkColor, colored_ink_masks


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

    def test_player_foul_cleaner_removes_long_colored_grid_line(self):
        image = Image.fromarray(blank(100, 70))
        draw = ImageDraw.Draw(image)
        draw.line((0, 35, 99, 35), fill=BLUE, width=3)
        cleaned = clean_player_foul_cell(np.asarray(image))
        _, blue_mask = colored_ink_masks(cleaned)
        self.assertEqual(int(blue_mask.sum()), 0)

    def test_player_foul_cleaner_removes_grid_corner(self):
        image = Image.fromarray(blank(100, 70))
        draw = ImageDraw.Draw(image)
        draw.line((2, 0, 2, 69), fill=BLUE, width=4)
        draw.line((0, 35, 99, 35), fill=BLUE, width=3)
        cleaned = clean_player_foul_cell(np.asarray(image))
        _, blue_mask = colored_ink_masks(cleaned)
        self.assertEqual(int(blue_mask.sum()), 0)

    def test_player_foul_cleaner_preserves_handwritten_symbol(self):
        image = Image.fromarray(blank(100, 70))
        draw = ImageDraw.Draw(image)
        draw.line((30, 12, 30, 58), fill=BLUE, width=5)
        draw.arc((30, 12, 65, 38), 270, 90, fill=BLUE, width=5)
        cleaned = clean_player_foul_cell(np.asarray(image))
        _, blue_mask = colored_ink_masks(cleaned)
        self.assertGreater(int(blue_mask.sum()), 50)

    def test_player_foul_row_ignores_colored_horizontal_grid_line(self):
        image = Image.fromarray(blank(250, 50))
        draw = ImageDraw.Draw(image)
        draw.line((0, 25, 249, 25), fill=BLUE, width=3)
        events = analyze_player_foul_row(
            np.asarray(image),
            team="A",
            jersey=7,
        )
        self.assertEqual(events, [])

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
        draw.rectangle((15, 10, 24, 32), fill=RED)
        # Slot 2 azul -> Q2
        draw.rectangle((65, 10, 74, 32), fill=BLUE)
        # Slot 3 vermelho -> Q3
        draw.rectangle((115, 10, 124, 32), fill=RED)
        # Slot 4 azul -> Q4
        draw.rectangle((165, 10, 174, 32), fill=BLUE)

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
        draw.rectangle((15, 10, 24, 32), fill=RED)

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

    def test_horizontal_terminal_stops_later_player_foul_cells(self):
        image = Image.fromarray(blank(250, 50))
        draw = ImageDraw.Draw(image)
        draw.line((100, 0, 100, 49), fill=BLUE, width=4)
        draw.rectangle((15, 10, 24, 32), fill=RED)
        draw.line((58, 25, 92, 25), fill=BLUE, width=4)
        draw.rectangle((115, 10, 124, 32), fill=RED)
        terminals = []

        fouls = analyze_player_foul_row(
            np.asarray(image),
            team="A",
            jersey=11,
            recognizer=FixedFoulRecognizer("P2", 0.9),
            terminals=terminals,
        )

        self.assertEqual([foul.slot for foul in fouls], [1])
        self.assertEqual(len(terminals), 1)
        self.assertEqual(terminals[0].kind, FoulTerminalKind.CLOSURE_STROKE)

    def test_p_like_symbol_is_not_misclassified_as_terminal_f(self):
        image = Image.fromarray(blank(44, 24))
        draw = ImageDraw.Draw(image)
        draw.line((8, 3, 8, 20), fill=RED, width=3)
        draw.arc((7, 3, 26, 15), 270, 90, fill=RED, width=3)
        draw.line((25, 7, 25, 11), fill=RED, width=3)

        self.assertIsNone(classify_foul_terminal(np.asarray(image)))

    def test_clear_f_shape_is_terminal_disqualification(self):
        image = Image.fromarray(blank(44, 24))
        draw = ImageDraw.Draw(image)
        draw.line((8, 2, 8, 21), fill=BLUE, width=4)
        draw.line((8, 3, 34, 3), fill=BLUE, width=4)
        draw.line((8, 11, 28, 11), fill=BLUE, width=4)

        terminal = classify_foul_terminal(np.asarray(image))
        self.assertIsNotNone(terminal)
        self.assertEqual(terminal.kind, FoulTerminalKind.DISQUALIFICATION)

    def test_player_foul_grid_detects_five_real_cells_and_twelve_rows(self):
        image = Image.fromarray(blank(250, 600))
        draw = ImageDraw.Draw(image)
        top = round(600 * 0.085)
        bottom = round(600 * 0.92)
        for x in (0, 50, 100, 150, 200, 249):
            draw.line((x, top, x, bottom), fill=(0, 0, 0), width=2)
        for index in range(13):
            y = round(top + (bottom - top) * index / 12)
            draw.line((0, y, 249, y), fill=(0, 0, 0), width=2)

        detected = detect_player_foul_grid(np.asarray(image), side="B")

        self.assertTrue(detected.valid)
        self.assertEqual(len(detected.x_lines), 6)
        self.assertEqual(len(detected.y_lines), 13)

    def test_parse_foul_symbol(self):
        self.assertEqual(parse_foul_symbol("P2"), (FoulKind.PERSONAL, 2))
        self.assertEqual(
            parse_foul_symbol(" u1 "),
            (FoulKind.UNSPORTSMANLIKE, 1),
        )
        self.assertEqual(
            parse_foul_symbol("U3"),
            (FoulKind.UNSPORTSMANLIKE, 3),
        )
        self.assertEqual(
            parse_foul_symbol("D2"),
            (FoulKind.DISQUALIFYING, 2),
        )
        cancelled = parse_foul_symbol_details("Pc")
        self.assertEqual(cancelled.kind, FoulKind.PERSONAL)
        self.assertTrue(cancelled.cancelled_penalty)
        self.assertFalse(cancelled.counts_as_team_foul)
        fighting = parse_foul_symbol_details("GD")
        self.assertTrue(fighting.fighting)
        self.assertFalse(fighting.counts_as_team_foul)
        self.assertEqual(parse_foul_symbol("?"), (FoulKind.UNKNOWN, None))


if __name__ == "__main__":
    unittest.main()
