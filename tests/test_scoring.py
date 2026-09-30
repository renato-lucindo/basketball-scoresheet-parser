import unittest

import numpy as np
from PIL import Image, ImageDraw

from sumula_reader.models import ScoringEvent, ShotType
from sumula_reader.scoring import (
    InkColor,
    ScoreMarkKind,
    assign_periods_from_color_runs,
    classify_score_mark,
    detect_jersey_circle,
    extract_scoring_events,
    iter_scoring_cells,
)
from sumula_reader.template import FECABA_V1


RED = (190, 45, 45)
BLUE = (35, 65, 185)


def blank(width=80, height=80):
    return np.full((height, width, 3), 255, dtype=np.uint8)


class ScoringTests(unittest.TestCase):
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

        # A chega a 2 com cesta de dois e depois a 3 com lance livre.
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
