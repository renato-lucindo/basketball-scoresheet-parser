import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from sumula_reader.models import DecisionStatus, ScoringEvent, ShotType
from sumula_reader.scoring import (
    InkColor,
    ScoreMarkObservation,
    ScoreMarkKind,
    ScoringCell,
    _detect_vertical_lines,
    _looks_like_closure_stroke,
    assign_periods_from_color_runs,
    classify_score_mark,
    detect_jersey_circle,
    extract_scoring_events,
    iter_scoring_cells,
    select_plausible_scoring_sequences,
)
from sumula_reader.template import FECABA_V1, NormalizedRect


RED = (190, 45, 45)
BLUE = (35, 65, 185)


def blank(width=80, height=80):
    return np.full((height, width, 3), 255, dtype=np.uint8)


class FixedScoringDecisionEngine:
    def __init__(self, choice="two_point", confidence=0.95):
        self.choice = choice
        self.confidence = confidence
        self.calls = []

    def classify_scoring_mark(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choice=self.choice,
            confidence=self.confidence,
        )


class ScoringTests(unittest.TestCase):
    def test_vertical_grid_recovers_missing_outer_lines(self):
        width = 1200
        dark = np.zeros((600, width), dtype=bool)
        lines = [0, 70, 132, 194, 274, 354, 416, 478, 558, 638, 719, 801, 881, 961, 1043, 1126, 1199]
        for x in lines[1:-1]:
            dark[:, max(0, x - 1) : min(width, x + 2)] = True

        detected = _detect_vertical_lines(dark, expected=17)

        self.assertEqual(len(detected), 17)
        self.assertLessEqual(abs(detected[0] - lines[0]), 2)
        self.assertLessEqual(abs(detected[-1] - lines[-1]), 2)

    def test_vertical_grid_merges_split_lines_and_ignores_short_noise(self):
        width = 1208
        dark = np.zeros((600, width), dtype=bool)
        lines = [3, 71, 132, 194, 272, 350, 412, 473, 551, 630, 711, 794, 874, 953, 1037, 1121, 1203]
        for x in lines:
            dark[:, max(0, x - 1) : min(width, x + 2)] = True
        # Split one line into two nearby vertical columns; both must
        # continuar representando uma unica divisoria da grade.
        dark[:, 69:75] = False
        dark[:, 68:70] = True
        dark[:, 73:75] = True
        # Risco manuscrito curto: forte localmente, mas insuficiente para ser
        # promovido a linha estrutural da tabela.
        dark[:210, 225:228] = True

        detected = _detect_vertical_lines(dark, expected=17)

        self.assertEqual(len(detected), 17)
        self.assertTrue(all(abs(a - b) <= 4 for a, b in zip(detected, lines)))

    def test_vertical_grid_recovers_two_missing_internal_dividers(self):
        width = 1213
        dark = np.zeros((600, width), dtype=bool)
        lines = [9, 68, 127, 195, 266, 344, 406, 467, 546, 625, 708, 791, 872, 954, 1038, 1124, 1207]
        missing = {1, 3}
        for index, x in enumerate(lines):
            if index in missing:
                continue
            dark[:, max(0, x - 1) : min(width, x + 2)] = True

        detected = _detect_vertical_lines(dark, expected=17)

        self.assertEqual(len(detected), 17)
        self.assertTrue(all(abs(a - b) <= 12 for a, b in zip(detected, lines)))

    def test_free_throw_dot(self):
        image = Image.fromarray(blank())
        draw = ImageDraw.Draw(image)
        draw.ellipse((31, 31, 48, 48), fill=RED)
        result = classify_score_mark(np.asarray(image))
        self.assertEqual(result.kind, ScoreMarkKind.FREE_THROW)
        self.assertEqual(result.ink_color, InkColor.RED)

    def test_field_goal_diagonal(self):
        image = Image.fromarray(blank())
        draw = ImageDraw.Draw(image)
        draw.line((18, 62, 62, 18), fill=BLUE, width=6)
        result = classify_score_mark(np.asarray(image))
        self.assertEqual(result.kind, ScoreMarkKind.FIELD_GOAL)
        self.assertEqual(result.ink_color, InkColor.BLUE)

    def test_jersey_circle(self):
        image = Image.fromarray(blank(100, 70))
        draw = ImageDraw.Draw(image)
        draw.text((42, 25), "7", fill=BLUE)
        draw.ellipse((12, 5, 88, 65), outline=BLUE, width=4)
        result = detect_jersey_circle(np.asarray(image))
        self.assertTrue(result.detected)
        self.assertGreater(result.angular_coverage, 0.5)

    def test_plain_jersey_is_not_circle(self):
        image = Image.fromarray(blank(100, 70))
        draw = ImageDraw.Draw(image)
        draw.line((45, 18, 45, 55), fill=BLUE, width=5)
        draw.line((45, 18, 58, 25), fill=BLUE, width=5)
        result = detect_jersey_circle(np.asarray(image))
        self.assertFalse(result.detected)

    def test_long_closure_stroke_is_not_a_jersey(self):
        image = Image.fromarray(blank(90, 50))
        draw = ImageDraw.Draw(image)
        draw.line((4, 44, 84, 5), fill=BLUE, width=4)
        self.assertTrue(_looks_like_closure_stroke(np.asarray(image)))

    def test_compact_two_digit_jersey_is_not_closure_stroke(self):
        image = Image.fromarray(blank(90, 50))
        draw = ImageDraw.Draw(image)
        draw.text((25, 12), "16", fill=BLUE)
        self.assertFalse(_looks_like_closure_stroke(np.asarray(image)))

    def test_periods_follow_color_runs(self):
        events = [
            ScoringEvent("A", None, 2, 7, ShotType.TWO_POINT, 2, "red"),
            ScoringEvent("A", None, 3, 7, ShotType.FREE_THROW, 1, "red"),
            ScoringEvent("A", None, 5, 9, ShotType.TWO_POINT, 2, "blue"),
            ScoringEvent("A", None, 7, 9, ShotType.TWO_POINT, 2, "red"),
            ScoringEvent("A", None, 9, 11, ShotType.TWO_POINT, 2, "blue"),
        ]
        assign_periods_from_color_runs(events)
        self.assertEqual([event.period for event in events], [1, 1, 2, 3, 4])

    def test_first_blue_scoring_run_maps_to_second_quarter(self):
        events = [
            ScoringEvent("A", None, 2, 7, ShotType.TWO_POINT, 2, "blue"),
        ]
        assign_periods_from_color_runs(events)
        self.assertEqual(events[0].period, 2)

    def test_running_score_validates_without_reclassifying_visual_mark(self):
        event = ScoringEvent(
            "A",
            None,
            3,
            7,
            ShotType.TWO_POINT,
            2,
            "red",
            confidence=0.95,
        )
        selected = select_plausible_scoring_sequences([event])
        self.assertEqual(selected[0].shot_type, ShotType.TWO_POINT)
        self.assertEqual(selected[0].points, 2)
        self.assertEqual(selected[0].status, DecisionStatus.REVIEW)

    def test_jev_resolves_only_ambiguous_scoring_mark(self):
        cell = ScoringCell(
            "A",
            2,
            NormalizedRect(0, 0, 0, 0),
            NormalizedRect(0, 0, 0, 0),
        )
        score_crop = blank(40, 40)
        jersey_crop = np.full((40, 40, 3), BLUE, dtype=np.uint8)
        mark = ScoreMarkObservation(
            kind=ScoreMarkKind.AMBIGUOUS,
            ink_color=InkColor.BLUE,
            confidence=0.4,
            ink_ratio=0.12,
            elongation=2.1,
        )
        engine = FixedScoringDecisionEngine()

        with patch(
            "sumula_reader.scoring._iter_scoring_crops",
            return_value=iter([(cell, score_crop, jersey_crop, True)]),
        ):
            with patch(
                "sumula_reader.scoring.classify_score_mark",
                return_value=mark,
            ):
                with patch(
                    "sumula_reader.scoring._looks_like_closure_stroke",
                    return_value=False,
                ):
                    with patch(
                        "sumula_reader.scoring.detect_jersey_circle",
                        return_value=SimpleNamespace(
                            detected=False,
                            confidence=1.0,
                        ),
                    ):
                        events = extract_scoring_events(
                            blank(100, 100),
                            decision_engine=engine,
                        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].shot_type, ShotType.TWO_POINT)
        self.assertEqual(events[0].points, 2)
        self.assertEqual(events[0].confidence, 0.95)
        self.assertEqual(len(engine.calls), 1)

    def test_extract_scoring_events_from_canonical_grid(self):
        page = Image.new(
            "RGB",
            (FECABA_V1.canonical_width, FECABA_V1.canonical_height),
            "white",
        )
        table_rect = FECABA_V1.region("scoring_table")
        table_box = table_rect.pixels(*page.size)
        table_width = table_box[2] - table_box[0]
        table_height = table_box[3] - table_box[1]
        draw = ImageDraw.Draw(page)

        cells = {
            (cell.team, cell.running_score): cell
            for cell in iter_scoring_cells()
        }

        def box_on_page(rect):
            left, top, right, bottom = rect.pixels(table_width, table_height)
            return (
                table_box[0] + left,
                table_box[1] + top,
                table_box[0] + right,
                table_box[1] + bottom,
            )

        # Team A reaches 2 with a two-point basket, then 3 with a free throw.
        a2 = box_on_page(cells[("A", 2)].score_rect)
        draw.line(
            (a2[0] + 3, a2[3] - 3, a2[2] - 3, a2[1] + 3),
            fill=RED,
            width=6,
        )
        a3 = box_on_page(cells[("A", 3)].score_rect)
        cx = (a3[0] + a3[2]) // 2
        cy = (a3[1] + a3[3]) // 2
        draw.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), fill=RED)

        events = extract_scoring_events(np.asarray(page))
        team_a = [event for event in events if event.team == "A"]
        self.assertEqual(
            [(event.running_score, event.shot_type, event.period) for event in team_a],
            [
                (2, ShotType.TWO_POINT, 1),
                (3, ShotType.FREE_THROW, 1),
            ],
        )


if __name__ == "__main__":
    unittest.main()
