from pathlib import Path
import json
import tempfile
import unittest

import numpy as np
import torch

from sumula_reader.torch_recognition import (
    FoulSymbolCNN,
    JerseySequenceCNN,
    TorchHandwritingRecognizer,
    _manifest_items,
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

    def test_training_manifest_uses_only_reviewed_grid_anchored_scoring_events(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "manifest.reviewed.jsonl"
            items = [
                {
                    "field_type": "scoring_event",
                    "split": "train",
                    "label": "7",
                    "review_state": "accepted",
                    "crop_path": "accepted.png",
                },
                {
                    "field_type": "scoring_event",
                    "split": "train",
                    "label": "8",
                    "review_state": "pending",
                    "crop_path": "pending.png",
                },
                {
                    "field_type": "jersey",
                    "split": "train",
                    "label": "9",
                    "review_state": "accepted",
                    "crop_path": "legacy.png",
                },
            ]
            manifest.write_text(
                "".join(json.dumps(item) + "\n" for item in items),
                encoding="utf-8",
            )

            selected = _manifest_items(manifest, "scoring_event", "train")

            self.assertEqual([item["label"] for item in selected], ["7"])


if __name__ == "__main__":
    unittest.main()
