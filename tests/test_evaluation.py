import unittest

from sumula_reader.evaluation import (
    PredictionRecord,
    choose_acceptance_threshold,
    compare_known_unknown_writers,
    evaluate_threshold,
)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            PredictionRecord("14", "14", 0.99, "g1", True),
            PredictionRecord("7", "7", 0.95, "g1", True),
            PredictionRecord("11", "17", 0.60, "g2", False),
            PredictionRecord("5", "5", 0.80, "g3", False),
        ]

    def test_threshold_metrics_separate_review_from_accept(self):
        metrics = evaluate_threshold(self.records, 0.80)
        self.assertEqual(metrics.accepted, 3)
        self.assertEqual(metrics.reviewed, 1)
        self.assertEqual(metrics.accepted_errors, 0)
        self.assertEqual(metrics.accepted_error_rate, 0.0)

    def test_choose_threshold_respects_error_gate(self):
        metrics = choose_acceptance_threshold(
            self.records,
            max_accepted_error_rate=0.0,
            min_automation_rate=0.50,
        )
        self.assertIsNotNone(metrics)
        assert metrics is not None
        self.assertGreaterEqual(metrics.threshold, 0.80)
        self.assertEqual(metrics.accepted_errors, 0)

    def test_known_unknown_writer_metrics_are_separate(self):
        groups = compare_known_unknown_writers(self.records, 0.0)
        self.assertEqual(groups["known_writer"].global_accuracy, 1.0)
        self.assertEqual(groups["unknown_writer"].global_accuracy, 0.5)


if __name__ == "__main__":
    unittest.main()
