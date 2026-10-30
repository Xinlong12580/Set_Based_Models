"""EDT can be absent without affecting core/ERL or the study workflow."""

import importlib.util
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]


class OptionalEDTTests(unittest.TestCase):
    def run_without_edt(self, code):
        with tempfile.TemporaryDirectory(prefix="erl_without_edt_") as temporary:
            root = Path(temporary)
            # Copy no extension source. Block installed EDT too, if present.
            shutil.copytree(ROOT / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__"))
            for package in ("casual_tests", "scripts"):
                (root / package).mkdir()
                for source in (ROOT / package).glob("*.py"):
                    shutil.copy2(source, root / package / source.name)
            shutil.copytree(ROOT / "raw_event_conversion", root / "raw_event_conversion",
                            ignore=shutil.ignore_patterns("__pycache__"))
            prefix = '''
import importlib.abc
import sys
class NoEDT(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "set_based_edt" or fullname.startswith("set_based_edt."):
            raise ModuleNotFoundError("EDT is deliberately absent", name=fullname)
sys.meta_path.insert(0, NoEDT())
'''
            result = subprocess.run(
                [sys.executable, "-B", "-c", textwrap.dedent(prefix) + textwrap.dedent(code)],
                cwd=root, env={**os.environ, "PYTHONPATH": str(root / "src"),
                               "PYTHONDONTWRITEBYTECODE": "1"},
                capture_output=True, text=True, timeout=60,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_core_import_star_gains_and_fit_do_not_load_edt(self):
        self.run_without_edt('''
            import numpy as np
            import set_based_models as models
            from set_based_models import *
            from set_based_models.core import gini_gain, PartitionStatistics, SplitStatistics
            assert "ExistentialDecisionTree" not in models.__all__
            root = PartitionStatistics(2, 2, 2., 2.)
            split = SplitStatistics(root, PartitionStatistics(2, 0, 2., 0.),
                                    PartitionStatistics(0, 2, 0., 2.), root)
            assert gini_gain(split) == .5
            model = ExistentialRuleList(
                (), event_observables=(ObservableSpec("x", "continuous"),),
                max_rule_depth=1, scoring_function=TPRFPRRuleScore(),
            ).fit(signal_event_features=[[2.], [3.]], background_event_features=[[0.], [1.]])
            assert model.get_rule(0).statistics.true_positive_rate == 1.
            assert model.get_rule(0).statistics.false_positive_rate == 0.
            assert not any(name.startswith(("set_based_edt", "set_based_models.decision_tree",
                                            "set_based_dt")) for name in sys.modules)
            try:
                from set_based_models import ExistentialDecisionTree
            except ModuleNotFoundError as error:
                assert error.name == "set_based_edt"
                assert "pip install -e extensions/edt" in str(error)
            else:
                raise AssertionError("EDT unexpectedly available")
        ''')

    @unittest.skipUnless(all(importlib.util.find_spec(name) for name in
                            ("torch", "awkward", "matplotlib")), "Study dependencies are optional")
    def test_study_workflow_and_all_input_modes_work_without_edt(self):
        self.run_without_edt(f'''
            import importlib
            import json
            from pathlib import Path
            import torch
            from unittest.mock import patch
            for path in Path("casual_tests").glob("run_ERL*.py"):
                importlib.import_module("casual_tests." + path.stem)
            from casual_tests import train_ERL as runner
            from raw_event_conversion import (build_object_tensor, build_pair_tensor,
                                             object_feature_schema, pair_feature_schema)
            from raw_event_conversion.objects import OBJECT_COLUMNS
            from raw_event_conversion.pairs import PAIR_COLUMNS
            from raw_event_conversion.tests.test_pairs import records
            from scripts.make_reference_anomalous import make_datasets
            from scripts.plot_acceptance_report import load_report, select_paths
            assert load_report({str(ROOT / 'examples/reports/edt_synthetic.json')!r})["model"] == "EDT"
            datasets = []
            for signal in (False, True):
                raw = records([{{"photons": [{{"pt": 60}}, {{"pt": 40, "phi": 3.}}]}}
                               if signal else {{"jets": [{{"pt": 10}}]}} for _ in range(12)])
                pairs, mask = build_pair_tensor(raw)
                datasets.append({{
                    "pf": torch.zeros((12, 1, 1)), "pf_columns": ["pt"],
                    "obj": build_object_tensor(raw), "obj_columns": list(OBJECT_COLUMNS),
                    "obj_feature_schema": object_feature_schema(),
                    "pair_obj": pairs, "pair_slot_mask": mask,
                    "pair_obj_columns": list(PAIR_COLUMNS), "pair_obj_feature_schema": pair_feature_schema(),
                    "event_features": torch.full((12, 1), 2 if signal else 0),
                    "event_feature_columns": ["n_photons"],
                    "label": torch.full((12,), 2 if signal else 1, dtype=torch.int64),
                }})
            reference, mixture = make_datasets(*datasets, reference_count=4,
                                              background_count=4, signal_count=2, seed=42)
            torch.save(reference, "reference.pt")
            torch.save(mixture, "anomalous.pt")
            original = runner.ExistentialRuleList
            def small_model(*args, **kwargs):
                return original(*args, **{{**kwargs, "max_rule_depth": 2, "beam_width": 3,
                                          "max_thresholds": 3, "max_candidate_rules": 15}})
            with patch.object(runner, "ExistentialRuleList", side_effect=small_model):
                for mode, options in (("events", ["--event-only"]), ("objects", ["--object-only"]),
                                      ("combined", []), ("pairs", ["--pair-tensor", "--object-only"])):
                    report = mode + ".txt"
                    runner.main(["--signal", "anomalous.pt", "--background", "reference.pt",
                                 "--report", report, "--print-top", "1", *options])
                    parsed = load_report(report)
                    assert parsed["results"]["rules"], mode
                    assert select_paths(parsed, rules="all", direction="mixture", top=1)[1], mode
            assert not any(name.startswith(("set_based_edt", "set_based_models.decision_tree",
                                            "set_based_dt", "casual_tests._edt_reporting",
                                            "casual_tests.stale"))
                           for name in sys.modules)
        ''')

    @unittest.skipUnless(importlib.util.find_spec("set_based_edt"), "EDT is an optional extension")
    def test_extension_and_compatibility_imports_share_types(self):
        from set_based_edt import ExistentialDecisionTree, Node, PartitionStatistics, gini_gain
        from set_based_dt import ExistentialDecisionTree as legacy
        from set_based_models import ExistentialDecisionTree as previous
        from set_based_models.core import PartitionStatistics as shared, gini_gain as shared_gain
        from set_based_models.decision_tree import ExistentialDecisionTree as namespace

        self.assertIs(ExistentialDecisionTree, legacy)
        self.assertIs(ExistentialDecisionTree, previous)
        self.assertIs(ExistentialDecisionTree, namespace)
        self.assertIs(PartitionStatistics, shared)
        self.assertIs(gini_gain, shared_gain)
        # Historical pickle global references remain resolvable through the aliases.
        self.assertIs(pickle.loads(b"cset_based_models.decision_tree.nodes\nNode\n."), Node)
        self.assertIs(pickle.loads(b"cset_based_models.decision_tree.training\nPartitionStatistics\n."), shared)


if __name__ == "__main__":
    unittest.main()
