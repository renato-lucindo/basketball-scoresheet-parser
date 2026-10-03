import unittest

import numpy as np
from PIL import Image, ImageDraw

from sumula_reader.models import DecisionStatus
from sumula_reader.pipeline import AnalysisContext, analyze_image
from sumula_reader.recognition import RecognitionCandidate, RecognitionResult
from sumula_reader.scoring import iter_scoring_cells
from sumula_reader.template import FECABA_V1


RED = (200, 35, 35)
BLUE = (20, 60, 190)


class EndToEndTests(unittest.TestCase):
    def test_synthetic_fecaba_page_produces_complete_result_structure(self):
        page = Image.new(
            "RGB",
            (FECABA_V1.canonical_width, FECABA_V1.canonical_height),
            "white",
        )
        draw = ImageDraw.Draw(page)
        self._draw_score_marks(draw, page.size)
        self._draw_starters(draw, page.size, "A")
        self._draw_starters(draw, page.size, "B")

        context = AnalysisContext(
            rosters={"A": [4, 5, 6, 7, 8], "B": [9, 10, 11, 12, 13]},
            team_names={"A": "Team A", "B": "Team B"},
            period_scores={"A": [3, 0, 0, 0], "B": [0, 0, 0, 0]},
            final_scores={"A": 3, "B": 0},
        )

        result = analyze_image(
            np.asarray(page),
            context=context,
            handwriting=_JerseyFourRecognizer(),
        )

        payload = result.to_dict()
        self.assertEqual(payload["schema_version"], "0.1")
        self.assertEqual(set(payload["teams"]), {"A", "B"})
        self.assertEqual(len(payload["core_fields"]), 18)
        self.assertEqual(len(payload["teams"]["A"]["players"]), 5)
        self.assertEqual(payload["teams"]["A"]["written_final_score"], 3)
        self.assertEqual(
            [period["written_score"] for period in payload["teams"]["A"]["periods"]],
            [3, 0, 0, 0],
        )
        self.assertEqual(payload["teams"]["A"]["calculated_score"], 3)
        self.assertEqual(len(payload["teams"]["A"]["scoring_events"]), 2)
        self.assertEqual(payload["status"], "accepted")
        self.assertTrue(
            all(field["status"] == "accepted" for field in payload["core_fields"])
        )

    @staticmethod
    def _draw_score_marks(draw, page_size):
        table_box = FECABA_V1.region("scoring_table").pixels(*page_size)
        table_width = table_box[2] - table_box[0]
        table_height = table_box[3] - table_box[1]
        cells = {
            (cell.team, cell.running_score): cell
            for cell in iter_scoring_cells()
        }

        def on_page(rect):
            left, top, right, bottom = rect.pixels(table_width, table_height)
            return (
                table_box[0] + left,
                table_box[1] + top,
                table_box[0] + right,
                table_box[1] + bottom,
            )

        two_points = on_page(cells[("A", 2)].score_rect)
        draw.line(
            (
                two_points[0] + 3,
                two_points[3] - 3,
                two_points[2] - 3,
                two_points[1] + 3,
            ),
            fill=RED,
            width=6,
        )
        free_throw = on_page(cells[("A", 3)].score_rect)
        cx = (free_throw[0] + free_throw[2]) // 2
        cy = (free_throw[1] + free_throw[3]) // 2
        draw.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), fill=RED)

    @staticmethod
    def _draw_starters(draw, page_size, side):
        box = FECABA_V1.region(f"team_{side.lower()}_participation").pixels(*page_size)
        width = box[2] - box[0]
        height = box[3] - box[1]
        data_top = box[1] + round(height * 0.085)
        row_height = (height - round(height * 0.085)) / 12
        for row in range(5):
            top = round(data_top + row * row_height)
            bottom = round(data_top + (row + 1) * row_height)
            left, right = box[0], box[0] + width
            margin = max(2, round(min(right - left, bottom - top) * 0.18))
            draw.line((left + margin, top + margin, right - margin, bottom - margin), fill=BLUE, width=3)
            draw.line((right - margin, top + margin, left + margin, bottom - margin), fill=BLUE, width=3)
            draw.ellipse((left + 1, top + 1, right - 1, bottom - 1), outline=RED, width=3)


class _JerseyFourRecognizer:
    def recognize(self, image, **kwargs):
        return RecognitionResult(
            value="4",
            confidence=1.0,
            candidates=[RecognitionCandidate("4", 1.0)],
            status=DecisionStatus.ACCEPTED,
        )


if __name__ == "__main__":
    unittest.main()
