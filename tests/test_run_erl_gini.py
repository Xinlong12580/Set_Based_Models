"""Check the Gini runner using statistics and synthetic events only."""

from contextlib import redirect_stdout
import importlib.util
import io
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is optional")
class ERLGiniTests(unittest.TestCase):
    def setUp(self):
        from casual_tests.stale import run_ERL_Gini
        from set_based_models import RuleStatistics

        self.runner = run_ERL_Gini
        self.statistics = RuleStatistics

    def stats(self, accepted_signal, total_signal, accepted_background, total_background):
        return self.statistics(
            accepted_signal, total_signal, accepted_background, total_background,
            float(accepted_signal), float(total_signal),
            float(accepted_background), float(total_background), 1, 1,
        )

    def test_electron_example_ranks_above_broad_rule(self):
        electron = self.stats(47, 1100, 1, 2000)
        broad = self.stats(368, 1100, 480, 2000)
        self.assertLess(
            electron.true_positive_rate - electron.false_positive_rate,
            broad.true_positive_rate - broad.false_positive_rate,
        )
        self.assertAlmostEqual(self.runner.gini_rule_score(electron), 0.012260615398533185)
        self.assertAlmostEqual(self.runner.gini_rule_score(broad), 0.0047148625825053)
        self.assertGreater(
            self.runner.gini_rule_score(electron), self.runner.gini_rule_score(broad),
        )

    def test_degenerate_and_perfect_partitions(self):
        for stats in (
            self.stats(0, 10, 0, 20),  # Reject everything.
            self.stats(10, 10, 20, 20),  # Accept everything.
            self.stats(5, 10, 10, 20),  # Equal acceptance rates.
            self.stats(0, 0, 0, 0),  # No weight.
            self.stats(5, 10, 0, 0),  # One class only.
        ):
            with self.subTest(stats=stats):
                self.assertAlmostEqual(self.runner.gini_rule_score(stats), 0.0)
        self.assertAlmostEqual(
            self.runner.gini_rule_score(self.stats(10, 10, 0, 10)), 0.5,
        )

    def test_event_weights_determine_gain(self):
        # Counts have equal acceptance rates, but weights perfectly separate
        # classes: accepted background and rejected signal have zero weight.
        stats = self.statistics(5, 10, 5, 10, 10.0, 10.0, 0.0, 30.0, 2, 1)
        self.assertAlmostEqual(self.runner.gini_rule_score(stats), 0.375)

    def test_support_and_complement_symmetry(self):
        small = self.stats(1, 100, 0, 100)
        larger = self.stats(20, 100, 0, 100)
        complement = self.stats(80, 100, 100, 100)
        self.assertGreater(
            self.runner.gini_rule_score(larger), self.runner.gini_rule_score(small),
        )
        self.assertAlmostEqual(
            self.runner.gini_rule_score(larger),
            self.runner.gini_rule_score(complement),
        )

    def test_main_uses_gini_during_synthetic_search(self):
        import numpy as np
        import torch

        signal = np.zeros((20, 23, 4))
        background = np.zeros((40, 23, 4))
        signal[:, 0, 0] = background[:, 0, 0] = 10.0
        signal[:4, 1] = [40.0, 0.0, 0.0, 2.0]  # Four events with an electron.
        datasets = {
            self.runner.ROOT / "data/anomalous.pt": (
                signal, signal[..., 0] > 0, torch.ones(20, dtype=torch.int64),
            ),
            self.runner.ROOT / "data/reference.pt": (
                background, background[..., 0] > 0, torch.ones(40, dtype=torch.int64),
            ),
        }
        models = []
        constructor = self.runner.ExistentialRuleList

        def make_model(*args, **kwargs):
            model = constructor(*args, **kwargs)
            models.append(model)
            return model

        output = io.StringIO()
        with (
            patch.object(self.runner, "load_obj_dataset", side_effect=datasets.__getitem__),
            patch.object(self.runner, "ExistentialRuleList", side_effect=make_model),
            redirect_stdout(output),
        ):
            self.runner.main([])

        model, = models
        self.assertIs(model.scoring_function, self.runner.gini_rule_score)
        self.assertTrue(model.candidate_rules_)
        self.assertAlmostEqual(model.candidate_rules_[0].generation_score, 4.0 / 63.0)
        for rule in model.candidate_rules_:
            for step in rule.steps:
                self.assertAlmostEqual(
                    step.generation_score, self.runner.gini_rule_score(step.statistics),
                )
        self.assertIn("mixture-enriched", output.getvalue())
        self.assertIn("Top pruned rules", output.getvalue())


if __name__ == "__main__":
    unittest.main()
