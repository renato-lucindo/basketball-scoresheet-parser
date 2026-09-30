import unittest

from sumula_reader.cli import _parse_roster, build_parser


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


if __name__ == "__main__":
    unittest.main()
