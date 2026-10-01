from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from sumula_reader.torch_recognition import (
    FoulSymbolCNN,
    JerseySequenceCNN,
    TorchHandwritingRecognizer,
    prepare_handwriting_image,
)
from sumula_reader.recognition import FOUL_LABELS


class TorchRecognitionTests(unittest.TestCase):
    def test_models_return_expected_shapes(self):
        images = torch.zeros((2, 1, 32, 64))
        self.assertEqual(tuple(JerseySequenceCNN.build()(images).shape), (2, 2, 11))
        self.assertEqual(
            tuple(FoulSymbolCNN.build()(images).shape),
            (2, len(FOUL_LABELS)),
        )

    def test_preprocessing_centers_colored_ink(self):
        image = np.full((30, 20, 3), 255, dtype=np.uint8)
        image[5:25, 8:12] = (0, 0, 255)
        tensor = prepare_handwriting_image(image)
        self.assertEqual(tuple(tensor.shape), (1, 32, 64))
        self.assertGreater(float(tensor.max()), 0.5)

    def test_recognizer_restricts_candidates_to_roster(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = JerseySequenceCNN.build()
            torch.save(
                {
                    "labels": [str(value) for value in range(100)],
                    "state_dict": model.state_dict(),
                    "threshold": 0.0,
                },
                root / "jersey.pt",
            )
            recognizer = TorchHandwritingRecognizer(root)
            result = recognizer.recognize(
                np.full((30, 20, 3), 255, dtype=np.uint8),
                field_type="jersey",
                allowed_labels=["4", "11", "25"],
            )
            self.assertEqual(
                {candidate.label for candidate in result.candidates},
                {"4", "11", "25"},
            )


if __name__ == "__main__":
    unittest.main()
