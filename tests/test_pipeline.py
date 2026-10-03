import unittest
from unittest.mock import patch

import numpy as np

from sumula_reader.models import PlayerResult, TeamFoulIndicator
from sumula_reader.pipeline import AnalysisContext, analyze_image


class PipelineTests(unittest.TestCase):
    def test_analysis_context_populates_written_game_fields(self):
        context = AnalysisContext(
            rosters={"A": [4, 5, 6, 7, 8], "B": [9, 10, 11, 12, 13]},
            team_names={"A": "CETAF B", "B": "Saldanha"},
            period_scores={"A": [0, 0, 0, 0], "B": [0, 0, 0, 0]},
            final_scores={"A": 0, "B": 0},
        )

        def players(_image, *, team, jerseys, template):
            return [
                PlayerResult(jersey=jersey, participated=True, starter=True)
                for jersey in jerseys
            ]

        def team_fouls(_image, *, team, template, decision_engine):
            return [TeamFoulIndicator(period=period, x_count=0) for period in range(1, 5)]

        with (
            patch("sumula_reader.pipeline.extract_scoring_events", return_value=[]),
            patch("sumula_reader.pipeline.extract_players", side_effect=players),
            patch("sumula_reader.pipeline.extract_player_foul_data", return_value=([], [])),
            patch("sumula_reader.pipeline.extract_team_foul_indicators", side_effect=team_fouls),
        ):
            result = analyze_image(np.zeros((10, 10, 3), dtype=np.uint8), context=context)

        self.assertEqual(result.teams["A"].name, "CETAF B")
        self.assertEqual(result.teams["B"].written_final_score, 0)
        self.assertEqual(
            [period.written_score for period in result.teams["A"].periods],
            [0, 0, 0, 0],
        )
        states = {field.path: field.status.value for field in result.core_fields}
        self.assertEqual(states["teams.A.name"], "accepted")
        self.assertEqual(states["teams.A.period_scoring"], "accepted")
        self.assertEqual(states["teams.A.final_score"], "accepted")


if __name__ == "__main__":
    unittest.main()
