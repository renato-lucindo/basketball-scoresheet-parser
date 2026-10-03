import unittest
from unittest.mock import patch

import numpy as np

from sumula_reader.models import DecisionStatus, ParticipantMark
from sumula_reader.participation import (
    ParticipationGridSpec,
    detect_participation,
    extract_roster,
)
from sumula_reader.recognition import RecognitionResult


def blank(size=80):
    return np.full((size, size, 3), 255, dtype=np.uint8)


def draw_x(image, color, thickness=5):
    size = image.shape[0]
    for offset in range(-thickness // 2, thickness // 2 + 1):
        idx = np.arange(18, size - 18)
        y1 = np.clip(idx + offset, 0, size - 1)
        y2 = np.clip((size - 1 - idx) + offset, 0, size - 1)
        image[y1, idx] = color
        image[y2, idx] = color


def draw_circle(image, color, radius=29, thickness=4):
    size = image.shape[0]
    cy = cx = (size - 1) / 2
    yy, xx = np.ogrid[:size, :size]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = (dist >= radius - thickness / 2) & (dist <= radius + thickness / 2)
    image[mask] = color


class ParticipationTests(unittest.TestCase):
    def test_empty_cell(self):
        result = detect_participation(blank())
        self.assertFalse(result.participated)
        self.assertFalse(result.starter)
        self.assertEqual(result.mark, ParticipantMark.NONE)

    def test_blue_x_participated(self):
        image = blank()
        draw_x(image, (20, 60, 190))
        result = detect_participation(image)
        self.assertTrue(result.participated)
        self.assertFalse(result.starter)
        self.assertEqual(result.mark, ParticipantMark.BLUE_X)

    def test_red_x_participated(self):
        image = blank()
        draw_x(image, (200, 35, 35))
        result = detect_participation(image)
        self.assertTrue(result.participated)
        self.assertFalse(result.starter)
        self.assertEqual(result.mark, ParticipantMark.RED_X)

    def test_blue_x_with_red_circle_is_starter(self):
        image = blank()
        draw_x(image, (20, 60, 190))
        draw_circle(image, (200, 35, 35))
        result = detect_participation(image)
        self.assertTrue(result.participated)
        self.assertTrue(result.starter)
        self.assertEqual(result.mark, ParticipantMark.BLUE_X_RED_CIRCLE)

    def test_roster_recognition_keeps_only_accepted_rows(self):
        block = np.full((30, 30, 3), 255, dtype=np.uint8)
        block[2:8, 8:22] = (20, 60, 190)
        block[12:18, 8:22] = (20, 60, 190)

        class Recognizer:
            def __init__(self):
                self.results = iter(
                    [
                        RecognitionResult("4", 0.95),
                        RecognitionResult(
                            "7",
                            0.55,
                            status=DecisionStatus.REVIEW,
                        ),
                    ]
                )

            def recognize(self, image, **kwargs):
                return next(self.results)

        with patch("sumula_reader.participation.crop_region", return_value=block):
            result = extract_roster(
                np.zeros((1, 1, 3), dtype=np.uint8),
                team="A",
                recognizer=Recognizer(),
                grid=ParticipationGridSpec(header_fraction=0.0, roster_rows=3),
                acceptance_threshold=0.90,
            )

        self.assertEqual(result.jerseys, [4])
        self.assertEqual(result.status, DecisionStatus.REVIEW)
        self.assertEqual([item.row for item in result.observations], [1, 2])
        self.assertEqual(result.observations[1].jersey, 7)

    def test_duplicate_recognized_jersey_is_unresolved(self):
        block = np.full((20, 30, 3), 255, dtype=np.uint8)
        block[2:8, 8:22] = (20, 60, 190)
        block[12:18, 8:22] = (20, 60, 190)

        class Recognizer:
            def recognize(self, image, **kwargs):
                return RecognitionResult("4", 0.95)

        with patch("sumula_reader.participation.crop_region", return_value=block):
            result = extract_roster(
                np.zeros((1, 1, 3), dtype=np.uint8),
                team="B",
                recognizer=Recognizer(),
                grid=ParticipationGridSpec(header_fraction=0.0, roster_rows=2),
                acceptance_threshold=0.90,
            )

        self.assertEqual(result.jerseys, [4])
        self.assertEqual(result.status, DecisionStatus.UNRESOLVED)
        self.assertEqual(result.observations[1].status, DecisionStatus.UNRESOLVED)

    def test_uncalibrated_roster_candidate_is_never_accepted(self):
        block = np.full((10, 30, 3), 255, dtype=np.uint8)
        block[2:8, 8:22] = (20, 60, 190)

        class Recognizer:
            def recognize(self, image, **kwargs):
                return RecognitionResult("4", 1.0)

        with patch("sumula_reader.participation.crop_region", return_value=block):
            result = extract_roster(
                np.zeros((1, 1, 3), dtype=np.uint8),
                team="A",
                recognizer=Recognizer(),
                grid=ParticipationGridSpec(header_fraction=0.0, roster_rows=1),
            )

        self.assertEqual(result.jerseys, [])
        self.assertEqual(result.status, DecisionStatus.REVIEW)
        self.assertEqual(result.observations[0].status, DecisionStatus.REVIEW)


if __name__ == "__main__":
    unittest.main()
