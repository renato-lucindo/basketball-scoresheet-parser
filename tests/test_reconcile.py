import unittest

from sumula_reader.models import (
    DecisionStatus,
    PlayerResult,
    ScoringEvent,
    ShotType,
    TeamResult,
)
from sumula_reader.reconcile import reconcile_team


class ReconcileTests(unittest.TestCase):
    def test_consistent_running_score_updates_player_totals(self):
        players = [
            PlayerResult(jersey=4, participated=True, starter=True),
            PlayerResult(jersey=5, participated=True, starter=True),
            PlayerResult(jersey=6, participated=True, starter=True),
            PlayerResult(jersey=7, participated=True, starter=True),
            PlayerResult(jersey=8, participated=True, starter=True),
        ]
        team = TeamResult(
            side="A",
            players=players,
            scoring_events=[
                ScoringEvent("A", 1, 2, 4, ShotType.TWO_POINT, 2, "red"),
                ScoringEvent("A", 1, 3, 4, ShotType.FREE_THROW, 1, "red"),
                ScoringEvent("A", 2, 6, 5, ShotType.THREE_POINT, 3, "blue"),
            ],
        )
        reconcile_team(team)
        self.assertEqual(team.calculated_score, 6)
        self.assertEqual([period.score for period in team.periods[:2]], [3, 3])
        self.assertEqual(players[0].points, 3)
        self.assertEqual(players[1].three_points_made, 1)
        self.assertEqual(team.status, DecisionStatus.ACCEPTED)

    def test_running_score_gap_requires_review(self):
        players = [
            PlayerResult(jersey=n, participated=True, starter=True)
            for n in range(4, 9)
        ]
        team = TeamResult(
            side="A",
            players=players,
            scoring_events=[
                ScoringEvent("A", 1, 2, 4, ShotType.TWO_POINT, 2, "red"),
                ScoringEvent("A", 1, 5, 5, ShotType.TWO_POINT, 2, "red"),
            ],
        )
        reconcile_team(team)
        self.assertEqual(team.status, DecisionStatus.REVIEW)
        self.assertTrue(
            any("salto 3" in warning for warning in team.warnings)
        )

    def test_unknown_jersey_requires_review(self):
        players = [
            PlayerResult(jersey=n, participated=True, starter=True)
            for n in range(4, 9)
        ]
        team = TeamResult(
            side="A",
            players=players,
            scoring_events=[
                ScoringEvent("A", 1, 2, None, ShotType.TWO_POINT, 2, "red"),
            ],
        )
        reconcile_team(team)
        self.assertEqual(team.status, DecisionStatus.REVIEW)
        self.assertTrue(
            any("camisa nao reconhecida" in warning for warning in team.warnings)
        )

    def test_reconciliation_is_idempotent(self):
        players = [
            PlayerResult(jersey=n, participated=True, starter=True)
            for n in range(4, 9)
        ]
        team = TeamResult(
            side="A",
            players=players,
            scoring_events=[
                ScoringEvent("A", 1, 2, 4, ShotType.TWO_POINT, 2, "red"),
            ],
        )
        reconcile_team(team)
        reconcile_team(team)
        self.assertEqual(players[0].points, 2)
        self.assertEqual(team.calculated_score, 2)


if __name__ == "__main__":
    unittest.main()
