"""Integration checks for the event-aware example runners."""

from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is optional")
class EventFeatureRunnerTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import torch

        self.np = np
        self.torch = torch
        self.signal = np.zeros((40, 23, 4), dtype=np.float64)
        self.background = np.zeros((60, 23, 4), dtype=np.float64)
        self.signal[:, 0, 0] = 10.0
        self.background[:, 0, 0] = 10.0
        self.signal_events = np.column_stack(
            (np.zeros(40), np.zeros(40), np.full(40, 2), np.ones(40))
        )
        self.background_events = np.column_stack(
            (np.zeros(60), np.zeros(60), np.zeros(60), np.ones(60))
        )
        self.columns = ("n_electrons", "n_muons", "n_photons", "n_jets")

    def loaded(self, array, events):
        return (
            array,
            array[..., 0] > 0,
            events,
            self.torch.ones(len(array), dtype=self.torch.int64),
            self.columns,
        )

    @unittest.skipUnless(importlib.util.find_spec("set_based_edt"), "EDT is an optional extension")
    def test_edt_learns_and_prints_event_cut(self):
        from casual_tests.stale import run_EDT_with_event_features as runner

        datasets = {
            (runner.ROOT / "data/with_event_multiplicities/anomalous.pt").resolve():
                self.loaded(self.signal, self.signal_events),
            (runner.ROOT / "data/with_event_multiplicities/reference.pt").resolve():
                self.loaded(self.background, self.background_events),
        }
        output = io.StringIO()
        with patch.object(runner, "load_dataset", side_effect=datasets.__getitem__), redirect_stdout(output):
            runner.main([])
        text = output.getvalue()
        self.assertIn("Event observables: n_electrons, n_muons, n_photons, n_jets", text)
        self.assertIn("event: n_photons", text)

    def test_erl_learns_and_prints_event_cut(self):
        from casual_tests.stale import run_ERL_Gini_with_event_features as runner

        datasets = {
            (runner.ROOT / "data/with_event_multiplicities/anomalous.pt").resolve():
                self.loaded(self.signal, self.signal_events),
            (runner.ROOT / "data/with_event_multiplicities/reference.pt").resolve():
                self.loaded(self.background, self.background_events),
        }
        output = io.StringIO()
        with patch.object(runner, "load_dataset", side_effect=datasets.__getitem__), redirect_stdout(output):
            runner.main([])
        text = output.getvalue()
        self.assertIn("Generated", text)
        self.assertIn("event: n_photons", text)
        self.assertIn("Top pruned rules", text)

    def test_loader_reads_augmented_format(self):
        from casual_tests.stale import run_EDT_with_event_features as runner

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "augmented.pt"
            obj = self.torch.zeros((2, 23, 4), dtype=self.torch.float32)
            obj[:, 0, 0] = 20
            dataset = {
                "obj": obj,
                "label": self.torch.tensor([0, 1]),
                "obj_columns": ["pt", "eta", "phi", "type_id"],
                "event_features": self.torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]]),
                "event_feature_columns": list(self.columns),
            }
            self.torch.save(dataset, path)
            loaded_obj, mask, events, labels, columns = runner.load_dataset(path)
        self.assertEqual(loaded_obj.shape, (2, 23, 4))
        self.assertTrue(mask[:, 0].all())
        self.assertEqual(events.shape, (2, 4))
        self.assertEqual(labels.tolist(), [0, 1])
        self.assertEqual(columns, self.columns)


if __name__ == "__main__":
    unittest.main()
