import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from sumula_reader.imaging import (
    NormalizedDocument,
    _find_document_corners_numpy,
    crop_region,
    order_corners,
    save_debug_bundle,
)
from sumula_reader.template import FECABA_V1, NormalizedRect


class ImagingTests(unittest.TestCase):
    def test_order_corners(self):
        points = np.array(
            [[100, 100], [10, 10], [10, 100], [100, 10]],
            dtype=np.float32,
        )
        ordered = order_corners(points)
        np.testing.assert_array_equal(
            ordered,
            np.array(
                [[10, 10], [100, 10], [100, 100], [10, 100]],
                dtype=np.float32,
            ),
        )

    def test_crop_region_uses_normalized_coordinates(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        crop = crop_region(image, NormalizedRect(0.25, 0.20, 0.50, 0.40))
        self.assertEqual(crop.shape, (40, 100, 3))

    def test_debug_bundle_writes_overlay_and_all_regions(self):
        image = np.full(
            (FECABA_V1.canonical_height, FECABA_V1.canonical_width, 3),
            255,
            dtype=np.uint8,
        )
        document = NormalizedDocument(
            image=image,
            corners=None,
            used_perspective_warp=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            files = save_debug_bundle(document, FECABA_V1, tmp)
            self.assertTrue((Path(tmp) / "normalized.png").exists())
            self.assertTrue((Path(tmp) / "regions-overlay.png").exists())
            self.assertTrue((Path(tmp) / "crops" / "scoring_table.png").exists())
            self.assertEqual(len(files), len(FECABA_V1.regions) + 2)

    def test_numpy_border_detector_finds_rectangular_sheet(self):
        image = np.full((700, 500, 3), 255, dtype=np.uint8)
        image[40:44, 30:470] = 0
        image[650:654, 30:470] = 0
        image[40:654, 30:34] = 0
        image[40:654, 466:470] = 0
        corners = _find_document_corners_numpy(image)
        self.assertIsNotNone(corners)
        assert corners is not None
        np.testing.assert_allclose(
            corners,
            np.array([[30, 40], [469, 40], [469, 653], [30, 653]]),
            atol=5,
        )


if __name__ == "__main__":
    unittest.main()
