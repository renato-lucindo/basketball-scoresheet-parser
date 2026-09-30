import unittest

import numpy as np
from PIL import Image, ImageDraw

from sumula_reader.fouls import (
    TeamFoulCellKind,
    analyze_player_foul_row,
    classify_team_foul_cell,
    detect_half_separator,
)


RED = (190, 45, 45)
BLUE = (35, 65, 185)


def blank(width=100, height=70):
    return np.full((height, width, 3), 255, dtype=np.uint8)


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


if __name__ == "__main__":
    unittest.main()
