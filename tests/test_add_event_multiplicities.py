"""Tests for adding event-level object multiplicities to prepared datasets."""

from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is optional")
class EventMultiplicityTests(unittest.TestCase):
    def setUp(self):
        import torch
        from scripts.add_event_multiplicities import (
            EVENT_FEATURE_COLUMNS,
            add_event_multiplicities,
            main,
        )

        self.torch = torch
        self.columns = EVENT_FEATURE_COLUMNS
        self.convert = add_event_multiplicities
        self.main = main

    def sample(self):
        torch = self.torch
        obj = torch.zeros((3, 7, 4), dtype=torch.float32)
        # event 0: two jets, one muon, one electron, and one photon
        obj[0, :5] = torch.tensor([
            [30.0, 0.1, 0.2, 0.0], [20.0, -0.1, -0.2, 0.0],
            [12.0, 0.3, 1.0, 1.0], [25.0, 0.2, 2.0, 2.0],
            [35.0, 0.4, -2.0, 3.0],
        ])
        # event 1: two photons plus MET; MET must not be counted.
        obj[1, :3] = torch.tensor([
            [45.0, 0.2, 1.0, 3.0], [32.0, -0.5, -1.0, 3.0],
            [50.0, 0.0, 2.5, 4.0],
        ])
        # event 2 stays fully padded: padding has type_id=0 but is not a jet.
        return {
            "obj": obj,
            "label": torch.tensor([1, 5, 1]),
            "obj_columns": ["pt", "eta", "phi", "type_id"],
            "pf": torch.arange(18).reshape(3, 3, 2),
            "pf_columns": ["pt", "eta"],
            "source_index": torch.tensor([7, 2, 9]),
            "description": "preserve me",
        }

    def test_counts_and_preserves_original_data(self):
        source = self.sample()
        result = self.convert(source)
        self.assertEqual(result["event_feature_columns"], self.columns)
        self.assertEqual(result["event_features"].dtype, self.torch.int64)
        self.assertEqual(
            result["event_features"].tolist(),
            [[1, 1, 1, 2], [0, 0, 2, 0], [0, 0, 0, 0]],
        )
        for key, value in source.items():
            if isinstance(value, self.torch.Tensor):
                self.assertTrue(self.torch.equal(result[key], value))
            else:
                self.assertEqual(result[key], value)
        self.assertNotIn("event_features", source)

    def test_column_lookup_does_not_assume_feature_positions(self):
        source = self.sample()
        order = [3, 2, 0, 1]
        source["obj"] = source["obj"][..., order]
        source["obj_columns"] = [source["obj_columns"][index] for index in order]
        result = self.convert(source)
        self.assertEqual(
            result["event_features"].tolist(),
            [[1, 1, 1, 2], [0, 0, 2, 0], [0, 0, 0, 0]],
        )

    def test_invalid_inputs_are_rejected(self):
        source = self.sample()
        cases = {
            "existing event tensor": {**source, "event_features": self.torch.zeros(3, 1)},
            "misaligned label": {**source, "label": source["label"][:2]},
            "unknown type": {**source, "obj": source["obj"].clone()},
            "non-integer type": {**source, "obj": source["obj"].clone()},
        }
        cases["unknown type"]["obj"][0, 0, 3] = 8
        cases["non-integer type"]["obj"][0, 0, 3] = 1.5
        for name, bad in cases.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.convert(bad, source_name=name)

    def test_cli_writes_new_aligned_files_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = [root / "reference-input.pt", root / "anomalous-input.pt"]
            for path in inputs:
                self.torch.save(self.sample(), path)
            output_dir = root / "prepared"
            args = [
                "--reference", str(inputs[0]), "--anomalous", str(inputs[1]),
                "--output-dir", str(output_dir),
            ]
            with redirect_stdout(io.StringIO()):
                self.main(args)
            for filename in ("reference.pt", "anomalous.pt"):
                result = self.torch.load(output_dir / filename, weights_only=True)
                self.assertEqual(tuple(result["event_features"].shape), (3, 4))
                self.assertTrue(self.torch.equal(result["obj"], self.sample()["obj"]))
            error = io.StringIO()
            with redirect_stderr(error), self.assertRaises(SystemExit) as exc:
                self.main(args)
            self.assertEqual(exc.exception.code, 2)
            self.assertIn("Refusing to overwrite", error.getvalue())


if __name__ == "__main__":
    unittest.main()
