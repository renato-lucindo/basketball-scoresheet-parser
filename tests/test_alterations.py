import unittest

import numpy as np
from PIL import Image, ImageDraw

from sumula_reader.alterations import detect_alteration


RED = (190, 45, 45)
BLUE = (35, 65, 185)


class AlterationTests(unittest.TestCase):
    def test_clean_single_color_mark_is_not_rasure(self):
        image = Image.new("RGB", (80, 80), "white")
        draw = ImageDraw.Draw(image)
        draw.line((15, 65, 65, 15), fill=BLUE, width=5)
        result = detect_alteration(np.asarray(image))
        self.assertFalse(result.suspected)

    def test_mixed_overwrite_is_suspected(self):
        image = Image.new("RGB", (80, 80), "white")
        draw = ImageDraw.Draw(image)
        for y in range(12, 69, 8):
            draw.line((10, y, 70, 70 - y // 2), fill=RED, width=5)
        for x in range(12, 69, 8):
            draw.line((x, 10, 70 - x // 2, 70), fill=BLUE, width=5)
        result = detect_alteration(np.asarray(image))
        self.assertTrue(result.suspected)
        self.assertIn("mixed_red_blue", result.reasons)


if __name__ == "__main__":
    unittest.main()
