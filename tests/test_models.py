import unittest

from sumula_reader.models import (
    DocumentResult,
    ParticipantMark,
    PlayerResult,
    TeamFoulIndicator,
    TeamResult,
)
from sumula_reader.validation import validate_team


class ModelTests(unittest.TestCase):
    def test_starter_requires_participation(self):
        with self.assertRaises(ValueError):
            PlayerResult(jersey=7, starter=True, participated=False)

    def test_four_team_fouls_is_capped(self):
        indicator = TeamFoulIndicator(period=2, x_count=4)
        self.assertTrue(indicator.count_is_capped)
        self.assertEqual(indicator.minimum_team_fouls, 4)

    def test_document_serializes_enums(self):
        player = PlayerResult(
            jersey=14,
            participated=True,
            starter=True,
            participation_mark=ParticipantMark.BLUE_X_RED_CIRCLE,
        )
        doc = DocumentResult(teams={"A": TeamResult(side="A", players=[player])})
        payload = doc.to_dict()
        self.assertEqual(
            payload["teams"]["A"]["players"][0]["participation_mark"],
            "blue_x_red_circle",
        )

    def test_team_requires_five_starters_for_clean_validation(self):
        players = [
            PlayerResult(jersey=n, participated=True, starter=(n < 5))
            for n in range(6)
        ]
        team = validate_team(TeamResult(side="A", players=players))
        self.assertEqual(len(team.starters), 5)
        self.assertEqual(team.warnings, [])


if __name__ == "__main__":
    unittest.main()
