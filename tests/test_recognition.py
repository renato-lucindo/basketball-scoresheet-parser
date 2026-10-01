import unittest

import numpy as np

from sumula_reader.models import DecisionStatus
from sumula_reader.recognition import (
    FoulRecognitionAdapter,
    JerseyRecognitionAdapter,
    RecognitionCandidate,
    RecognitionResult,
    WriterProfile,
    apply_writer_profile,
    constrain_candidates,
)


class FakeRecognizer:
    def __init__(self, candidates):
        self.candidates = candidates

    def recognize(self, image, **kwargs):
        return RecognitionResult(
            value=self.candidates[0].label,
            confidence=self.candidates[0].probability,
            candidates=list(self.candidates),
        )


class RecognitionTests(unittest.TestCase):
    def test_roster_constraint_removes_impossible_jersey(self):
        result = RecognitionResult(
            value="19",
            confidence=0.55,
            candidates=[
                RecognitionCandidate("19", 0.55),
                RecognitionCandidate("14", 0.40),
                RecognitionCandidate("11", 0.05),
            ],
        )
        constrained = constrain_candidates(result, ["14", "11"])
        self.assertEqual(constrained.value, "14")
        self.assertGreater(constrained.confidence, 0.8)

    def test_writer_profile_reorders_close_candidates(self):
        result = RecognitionResult(
            value="19",
            confidence=0.51,
            candidates=[
                RecognitionCandidate("19", 0.51),
                RecognitionCandidate("14", 0.49),
            ],
        )
        profile = WriterProfile(
            writer_id="writer_07",
            sample_count=100,
            label_priors={"19": 0.05, "14": 0.95},
        )
        adjusted = apply_writer_profile(result, profile)
        self.assertEqual(adjusted.value, "14")

    def test_jersey_adapter_returns_integer(self):
        recognizer = FakeRecognizer(
            [
                RecognitionCandidate("19", 0.6),
                RecognitionCandidate("14", 0.4),
            ]
        )
        adapter = JerseyRecognitionAdapter(
            recognizer,
            {"A": [4, 7, 14]},
        )
        value, confidence = adapter.recognize(
            np.zeros((20, 20, 3), dtype=np.uint8),
            team="A",
        )
        self.assertEqual(value, 14)
        self.assertEqual(confidence, 1.0)

    def test_foul_adapter_limits_labels_and_keeps_writer_context(self):
        recognizer = FakeRecognizer(
            [
                RecognitionCandidate("P2", 0.75),
                RecognitionCandidate("A", 0.20),
                RecognitionCandidate("U1", 0.05),
            ]
        )
        adapter = FoulRecognitionAdapter(
            recognizer,
            writer_id="writer_07",
        )
        result = adapter.recognize(
            np.zeros((20, 20, 3), dtype=np.uint8),
        )
        self.assertEqual(result.value, "P2")
        self.assertGreater(result.confidence, 0.9)
        self.assertEqual(
            [candidate.label for candidate in result.candidates],
            ["P2", "U1"],
        )

    def test_no_allowed_candidate_requires_review(self):
        result = RecognitionResult(
            value="99",
            confidence=1.0,
            candidates=[RecognitionCandidate("99", 1.0)],
        )
        constrained = constrain_candidates(result, ["4", "7"])
        self.assertIsNone(constrained.value)
        self.assertEqual(constrained.status, DecisionStatus.REVIEW)


if __name__ == "__main__":
    unittest.main()
