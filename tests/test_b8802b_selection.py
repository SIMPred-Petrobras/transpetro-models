"""Regression checks for sentinel dates accidentally rewarding false alarms."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from automl import _select_best_row
from transpetro_modelos.config import EQUIPMENT_CONFIGS
from transpetro_modelos.training.automl import rank_results


class SentinelSelectionTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            dict(model="alarma_mais", prefailure_alert_rate=.9, composite_score=.89,
                 normal_alert_rate=.001, val_fp_rate_heldout=.02),
            dict(model="alarma_menos", prefailure_alert_rate=.1, composite_score=.09,
                 normal_alert_rate=.002, val_fp_rate_heldout=.005),
        ]

    def test_sentinel_does_not_reward_pseudo_detection(self):
        self.assertEqual(rank_results(self.rows, failure_observed=False).iloc[0].model, "alarma_menos")
        self.assertEqual(_select_best_row(self.rows, .03, "insample", False)["model"], "alarma_menos")

    def test_real_failure_keeps_existing_objective(self):
        self.assertEqual(_select_best_row(self.rows, .03, "heldout")["model"], "alarma_mais")

    def test_missing_holdout_cannot_fall_back_to_training(self):
        for r in self.rows:
            r["val_fp_rate_heldout"] = None
        with self.assertRaises(ValueError):
            rank_results(self.rows, failure_observed=False)

    def test_no_candidate_under_budget(self):
        self.assertIsNone(_select_best_row(self.rows, .001, "heldout", False))

    def test_sentinel_metadata_inherited_by_thermal_variants(self):
        for eq in ["B-8802B-2025", "B-8802B-2025-fe", "B-8802B-2025-dT1", "B-8802B-2025-dT2"]:
            self.assertFalse(EQUIPMENT_CONFIGS[eq].failure_observed)
        self.assertTrue(EQUIPMENT_CONFIGS["B-8802B"].failure_observed)


if __name__ == "__main__":
    unittest.main()
