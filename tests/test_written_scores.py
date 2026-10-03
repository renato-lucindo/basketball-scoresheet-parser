import unittest

import numpy as np

from sumula_reader.models import DecisionStatus
from sumula_reader.recognition import RecognitionResult
from sumula_reader.template import NormalizedRect, TemplateSpec
from sumula_reader.written_scores import PERIOD_SCORE_CELLS, extract_period_score_candidates


class WrittenScoreTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((100, 1000, 3), 255, dtype=np.uint8)
        for cell in PERIOD_SCORE_CELLS:
            left, top, right, bottom = cell.rect.pixels(1000, 100)
            self.image[top + 2 : bottom - 2, left + 2 : right - 2] = (20, 60, 190)
        self.template = TemplateSpec(
            template_id="test",
            canonical_width=1000,
            canonical_height=100,
            regions={"period_scores": NormalizedRect(0, 0, 1, 1)},
        )

    def test_uncalibrated_scores_remain_review_candidates(self):
        recognizer = _SequenceRecognizer([16, 11, 21, 9, 27, 2, 26, 23])

        results = extract_period_score_candidates(
            self.image,
            recognizer=recognizer,
            template=self.template,
        )

        self.assertEqual(
            [period.written_score_candidate for period in results["A"]],
            [16, 21, 27, 26],
        )
        self.assertTrue(all(period.written_score is None for period in results["A"]))
        self.assertTrue(
            all(period.status is DecisionStatus.REVIEW for period in results["A"])
        )

    def test_calibrated_threshold_can_accept_scores(self):
        recognizer = _SequenceRecognizer([16, 11, 21, 9, 27, 2, 26, 23])

        results = extract_period_score_candidates(
            self.image,
            recognizer=recognizer,
            template=self.template,
            acceptance_threshold=0.90,
        )

        self.assertEqual(
            [period.written_score for period in results["B"]],
            [11, 9, 2, 23],
        )
        self.assertTrue(
            all(period.status is DecisionStatus.ACCEPTED for period in results["B"])
        )


class _SequenceRecognizer:
    def __init__(self, values):
        self.values = iter(values)

    def recognize(self, image, **kwargs):
        return RecognitionResult(str(next(self.values)), 0.95)


if __name__ == "__main__":
    unittest.main()
