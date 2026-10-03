import unittest

from sumula_reader.cli import _parse_period_scores, _parse_roster, build_parser


class CliTests(unittest.TestCase):
    def test_parse_roster(self):
        self.assertEqual(_parse_roster("4, 7,11"), [4, 7, 11])

    def test_duplicate_roster_number_is_rejected(self):
        with self.assertRaises(SystemExit):
            _parse_roster("4,7,4")

    def test_analyze_command_is_registered(self):
        args = build_parser().parse_args(
            [
                "analyze",
                "jogo.pdf",
                "--roster-a",
                "4,5,6,7,8",
                "--roster-b",
                "9,10,11,12,13",
            ]
        )
        self.assertEqual(args.command, "analyze")

    def test_analyze_allows_automatic_roster_recognition(self):
        args = build_parser().parse_args(["analyze", "game.pdf"])
        self.assertIsNone(args.roster_a)
        self.assertIsNone(args.roster_b)

    def test_parse_period_scores(self):
        self.assertEqual(_parse_period_scores("16, 21,27,26"), [16, 21, 27, 26])

    def test_period_scores_require_four_non_negative_values(self):
        for value in ("1,2,3", "1,2,3,-1", "1,2,three,4"):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                _parse_period_scores(value)

    def test_analyze_accepts_written_score_context(self):
        args = build_parser().parse_args(
            [
                "analyze",
                "game.pdf",
                "--roster-a",
                "4,5,6,7,8",
                "--roster-b",
                "9,10,11,12,13",
                "--team-a-name",
                "CETAF B",
                "--team-b-name",
                "Saldanha",
                "--period-scores-a",
                "16,21,27,26",
                "--period-scores-b",
                "11,9,2,23",
                "--final-score-a",
                "90",
                "--final-score-b",
                "45",
            ]
        )
        self.assertEqual(args.period_scores_a, "16,21,27,26")
        self.assertEqual(args.final_score_b, 45)

    def test_analyze_accepts_optional_jev_model(self):
        args = build_parser().parse_args(
            [
                "analyze",
                "jogo.pdf",
                "--roster-a",
                "4,5,6,7,8",
                "--roster-b",
                "9,10,11,12,13",
                "--jev",
                "--jev-model",
                "jev-preview",
            ]
        )
        self.assertTrue(args.jev)
        self.assertEqual(args.jev_model, "jev-preview")

    def test_review_commands_are_registered(self):
        prepare = build_parser().parse_args(
            ["dataset-review-prepare", "--stage", "scoring"]
        )
        imported = build_parser().parse_args(
            ["dataset-review-import", "export.json", "--stage", "jerseys"]
        )
        status = build_parser().parse_args(["dataset-review-status"])
        self.assertEqual(prepare.command, "dataset-review-prepare")
        self.assertEqual(prepare.stage, "scoring")
        self.assertEqual(imported.command, "dataset-review-import")
        self.assertEqual(status.command, "dataset-review-status")


if __name__ == "__main__":
    unittest.main()
