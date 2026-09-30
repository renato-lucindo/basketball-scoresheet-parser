import unittest

from sumula_reader.template import FECABA_V1, get_template


class TemplateTests(unittest.TestCase):
    def test_fecaba_core_regions_are_configured(self):
        expected = {
            "team_a_roster",
            "team_a_jersey",
            "team_a_participation",
            "team_a_player_fouls",
            "team_a_team_fouls",
            "team_b_roster",
            "team_b_jersey",
            "team_b_participation",
            "team_b_player_fouls",
            "team_b_team_fouls",
            "scoring_table",
            "period_scores",
            "final_score",
            "apontador",
        }
        self.assertTrue(expected.issubset(FECABA_V1.regions))

    def test_get_template(self):
        self.assertIs(get_template("fecaba_v1"), FECABA_V1)


if __name__ == "__main__":
    unittest.main()
