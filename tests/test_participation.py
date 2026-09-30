import unittest

import numpy as np

from sumula_reader.models import ParticipantMark
from sumula_reader.participation import detect_participation


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


if __name__ == "__main__":
    unittest.main()
