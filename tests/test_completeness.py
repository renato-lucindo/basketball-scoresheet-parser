import unittest

from sumula_reader.completeness import assess_core_fields
from sumula_reader.models import (
    DecisionStatus,
    DocumentResult,
    ParticipantMark,
    PeriodResult,
    PlayerResult,
    TeamFoulIndicator,
    TeamResult,
)


class CompletenessTests(unittest.TestCase):
    def test_missing_team_is_explicitly_unresolved(self):
        document = assess_core_fields(DocumentResult())

        self.assertEqual(document.status, DecisionStatus.UNRESOLVED)
        self.assertEqual(len(document.core_fields), 18)
        self.assertTrue(
            all(field.status is DecisionStatus.UNRESOLVED for field in document.core_fields)
        )

    def test_complete_core_record_is_accepted(self):
        document = DocumentResult(
            teams={
                side: TeamResult(
                    side=side,
                    name=f"Team {side}",
                    players=[
                        PlayerResult(
                            jersey=number,
                            participation_mark=ParticipantMark.BLUE_X_RED_CIRCLE,
                            participated=True,
                            starter=True,
                        )
                        for number in range(4, 9)
                    ],
                    periods=[
                        PeriodResult(number=number, score=0, written_score=0)
                        for number in range(1, 5)
                    ],
                    scoring_events=[],
                    team_fouls=[
                        TeamFoulIndicator(period=number, x_count=0)
                        for number in range(1, 5)
                    ],
                    written_final_score=0,
                )
                for side in ("A", "B")
            }
        )

        assess_core_fields(document)

        states = {field.path: field.status for field in document.core_fields}
        self.assertEqual(states["teams.A.scoring_events"], DecisionStatus.ACCEPTED)
        self.assertEqual(document.status, DecisionStatus.ACCEPTED)

    def test_partial_core_record_preserves_review_and_unresolved_states(self):
        team = TeamResult(
            side="A",
            name="Team A",
            players=[
                PlayerResult(jersey=number, participated=True, starter=number < 9)
                for number in range(4, 10)
            ],
        )
        document = assess_core_fields(DocumentResult(teams={"A": team}))
        states = {field.path: field.status for field in document.core_fields}

        self.assertEqual(states["teams.A.starters"], DecisionStatus.ACCEPTED)
        self.assertEqual(states["teams.A.final_score"], DecisionStatus.UNRESOLVED)
        self.assertEqual(states["teams.B.name"], DecisionStatus.UNRESOLVED)
        self.assertEqual(team.status, DecisionStatus.UNRESOLVED)

    def test_ambiguous_participation_propagates_review(self):
        players = [
            PlayerResult(jersey=number, participated=True, starter=True)
            for number in range(4, 9)
        ]
        players[0].participation_status = DecisionStatus.REVIEW
        team = TeamResult(side="A", name="Team A", players=players)

        document = assess_core_fields(DocumentResult(teams={"A": team}))
        states = {field.path: field.status for field in document.core_fields}

        self.assertEqual(states["teams.A.participation"], DecisionStatus.REVIEW)


if __name__ == "__main__":
    unittest.main()
