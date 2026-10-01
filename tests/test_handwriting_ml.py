from pathlib import Path
import unittest

import torch

from sumula_reader.cli import build_parser
from sumula_reader.handwriting_ml import DigitCNN, DigitTrainingConfig, train_emnist_digits


class HandwritingMLTests(unittest.TestCase):
    def test_digit_model_returns_one_logit_per_digit(self):
        model = DigitCNN.build()
        output = model(torch.zeros((3, 1, 28, 28)))
        self.assertEqual(tuple(output.shape), (3, 10))

    def test_training_command_has_reproducible_defaults(self):
        args = build_parser().parse_args(["train-digits"])
        self.assertEqual(args.epochs, 5)
        self.assertEqual(args.seed, 20260930)
        self.assertEqual(args.output, Path("models/digit-cnn-v1.pt"))

    def test_training_rejects_zero_epochs_before_downloading(self):
        config = DigitTrainingConfig(
            data_dir=Path("unused"),
            output=Path("unused.pt"),
            epochs=0,
        )
        with self.assertRaisesRegex(ValueError, "epochs"):
            train_emnist_digits(config)


if __name__ == "__main__":
    unittest.main()
