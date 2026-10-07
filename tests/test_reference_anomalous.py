"""Tests for the optional PyTorch dataset preparation script."""

import importlib.util
from contextlib import redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is optional")
class ReferenceAnomalousTests(unittest.TestCase):
    def setUp(self):
        import torch
        from scripts.make_reference_anomalous import main, make_datasets

        self.torch = torch
        self.make_datasets = make_datasets
        self.main = main

        def sample(n, label):
            rows = torch.arange(n, dtype=torch.float32) + label * 100
            return {
                "pf": rows[:, None, None].expand(n, 3, 2).clone(),
                "obj": rows[:, None, None].expand(n, 2, 2).clone(),
                "label": torch.full((n,), label, dtype=torch.int64),
                "event_features": rows[:, None].clone(),
                "pf_columns": ["pt", "eta"],
                "obj_columns": ["pt", "eta"],
            }

        self.minbias, self.zz = sample(20, 1), sample(10, 5)
        self.options = dict(reference_count=8, background_count=5, signal_count=3, seed=42)

    def test_disjoint_counts_and_alignment(self):
        reference, anomalous = self.make_datasets(self.minbias, self.zz, **self.options)
        self.assertEqual(len(reference["label"]), 8)
        self.assertEqual(anomalous["label"].tolist().count(1), 5)
        self.assertEqual(anomalous["label"].tolist().count(5), 3)
        used = set(reference["source_index"].tolist())
        other = set(anomalous["source_index"][anomalous["source_id"] == 0].tolist())
        self.assertFalse(used & other)
        self.assertEqual(len(used), 8)
        self.assertEqual(len(other), 5)
        for result in (reference, anomalous):
            ids = list(zip(result["source_id"].tolist(), result["source_index"].tolist()))
            self.assertEqual(len(ids), len(set(ids)))
            for row, (source_id, source_index) in enumerate(ids):
                source = (self.minbias, self.zz)[source_id]
                for key in ("pf", "obj", "label", "event_features"):
                    self.assertTrue(self.torch.equal(result[key][row], source[key][source_index]))
            self.assertEqual(result["pf_columns"], self.minbias["pf_columns"])
        self.assertNotEqual(anomalous["source_id"].tolist(), [0] * 5 + [1] * 3)

    def test_reproducible_without_changing_global_rng(self):
        before = self.torch.random.get_rng_state().clone()
        first = self.make_datasets(self.minbias, self.zz, **self.options)
        second = self.make_datasets(self.minbias, self.zz, **self.options)
        self.assertTrue(self.torch.equal(before, self.torch.random.get_rng_state()))
        for left, right in zip(first, second):
            for key in left:
                if isinstance(left[key], self.torch.Tensor):
                    self.assertTrue(self.torch.equal(left[key], right[key]))
                else:
                    self.assertEqual(left[key], right[key])

    def test_invalid_inputs(self):
        for name, value in (("reference_count", 0), ("background_count", -1),
                            ("reference_count", 16), ("signal_count", 11),
                            ("seed", -1)):
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                self.make_datasets(self.minbias, self.zz, **{**self.options, name: value})
        bad_schema = {**self.zz, "pf_columns": ["eta", "pt"]}
        with self.assertRaisesRegex(ValueError, "metadata"):
            self.make_datasets(self.minbias, bad_schema, **self.options)
        bad_shape = {**self.zz, "obj": self.zz["obj"][:, :1]}
        with self.assertRaisesRegex(ValueError, "shapes"):
            self.make_datasets(self.minbias, bad_shape, **self.options)
        bad_alignment = {**self.zz, "event_features": self.zz["event_features"][:-1]}
        with self.assertRaisesRegex(ValueError, "aligned"):
            self.make_datasets(self.minbias, bad_alignment, **self.options)

    def test_cli_round_trip_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = [root / "minbias.pt", root / "zz.pt"]
            for sample, path in zip((self.minbias, self.zz), inputs):
                self.torch.save(sample, path)
            args = ["--minbias", str(inputs[0]), "--zz", str(inputs[1]),
                    "--output-dir", str(root / "output"), "--reference-count", "8",
                    "--background-count", "5", "--signal-count", "3"]
            self.main(args)
            for name in ("reference", "anomalous"):
                saved = self.torch.load(root / "output" / f"{name}.pt", weights_only=True)
                self.assertEqual(len(saved["label"]), 8)
                self.assertEqual(saved["source_files"], [str(path) for path in inputs])
            old = (root / "output/reference.pt").read_bytes()
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                self.main(args + ["--no-overwrite"])
            self.assertEqual(exc.exception.code, 2)
            self.assertEqual((root / "output/reference.pt").read_bytes(), old)

    def cli_inputs(self, root):
        inputs = [root / "background.pt", root / "signal.pt"]
        for sample, path in zip((self.minbias, self.zz), inputs):
            self.torch.save(sample, path)
        args = ["--background", str(inputs[0]), "--signal", str(inputs[1]),
                "--output-dir", str(root / "output"), "--reference-count", "8",
                "--background-count", "5", "--signal-count", "3"]
        return inputs, args

    def test_cli_default_overwrite_warns_for_both_files_and_updates_outputs(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            inputs, args = self.cli_inputs(root)
            self.main(args)
            before = (root / "output/reference.pt").read_bytes()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                self.main(args + ["--seed", "43"])
            self.assertEqual(len(caught), 2)
            self.assertTrue(all("Overwriting existing file:" in str(item.message) for item in caught))
            self.assertNotEqual((root / "output/reference.pt").read_bytes(), before)
            for name in ("reference", "anomalous"):
                data = self.torch.load(root / "output" / f"{name}.pt", weights_only=True)
                self.assertEqual(data["sampling_seed"], 43)
                self.assertEqual(data["source_files"], [str(path) for path in inputs])

    def test_failed_second_serialization_preserves_both_existing_outputs(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            _, args = self.cli_inputs(root)
            self.main(args)
            outputs = [root / "output" / name for name in ("reference.pt", "anomalous.pt")]
            before = [path.read_bytes() for path in outputs]
            original_save = self.torch.save
            calls = 0

            def fail_second(*values, **options):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("save failed")
                return original_save(*values, **options)

            with warnings.catch_warnings(), patch("scripts.make_reference_anomalous.torch.save", side_effect=fail_second):
                warnings.simplefilter("ignore", UserWarning)
                with self.assertRaisesRegex(RuntimeError, "save failed"):
                    self.main(args + ["--seed", "43"])
            self.assertEqual([path.read_bytes() for path in outputs], before)
            self.assertEqual(list((root / "output").glob(".reference_anomalous_*")), [])

    def test_cli_protects_inputs_output_aliases_and_non_files(self):
        for mode in ("input_path", "hardlink", "symlink", "dangling_symlink", "directory"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                inputs, args = self.cli_inputs(root)
                output = root / "output/reference.pt"
                output.parent.mkdir()
                if mode == "input_path":
                    self.torch.save(self.minbias, output)
                    args[1] = str(output)
                elif mode == "hardlink":
                    os.link(inputs[0], output)
                elif mode == "symlink":
                    output.symlink_to(inputs[0])
                elif mode == "dangling_symlink":
                    output.symlink_to(root / "missing.pt")
                else:
                    output.mkdir()
                before = [path.read_bytes() for path in inputs]
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    self.main(args)
                self.assertEqual([path.read_bytes() for path in inputs], before)
                self.assertFalse((root / "output/anomalous.pt").exists())

    def test_generic_cli_preserves_arbitrary_signal_labels_and_provenance(self):
        # Cover another legacy Other=5 process, top, an unknown process, and
        # equal input labels: source identity must never be inferred from label.
        for name, label in (("ggHgammagamma", 5), ("TT_leptonic", 2),
                            ("other_signal", 42), ("same_label", 1)):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                signal = {**self.zz, "label": self.torch.full_like(self.zz["label"], label)}
                inputs = [root / "background.pt", root / f"{name}.pt"]
                for sample, path in zip((self.minbias, signal), inputs):
                    self.torch.save(sample, path)
                with redirect_stdout(io.StringIO()):
                    self.main([
                        "--background", str(inputs[0]), "--signal", str(inputs[1]),
                        "--output-dir", str(root / "output"),
                        "--reference-count", "8", "--background-count", "5",
                        "--signal-count", "3", "--seed", "42",
                    ])
                reference = self.torch.load(root / "output/reference.pt", weights_only=True)
                anomalous = self.torch.load(root / "output/anomalous.pt", weights_only=True)
                is_background = anomalous["source_id"] == 0
                is_signal = anomalous["source_id"] == 1
                self.assertEqual(int(is_background.sum()), 5)
                self.assertEqual(int(is_signal.sum()), 3)
                self.assertTrue(bool((anomalous["label"][is_signal] == label).all()))
                self.assertTrue(bool((anomalous["label"][is_background] == 1).all()))
                self.assertFalse(
                    set(reference["source_index"].tolist())
                    & set(anomalous["source_index"][is_background].tolist())
                )
                for result in (reference, anomalous):
                    self.assertEqual(result["source_files"], [str(path) for path in inputs])
                    for row, (source_id, source_index) in enumerate(zip(
                        result["source_id"].tolist(), result["source_index"].tolist(),
                    )):
                        source = (self.minbias, signal)[source_id]
                        for key in ("pf", "obj", "label", "event_features"):
                            self.assertTrue(self.torch.equal(
                                result[key][row], source[key][source_index],
                            ))

    def test_generic_cli_rejects_same_file_before_creating_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "sample.pt"
            self.torch.save(self.minbias, source)
            error = io.StringIO()
            with redirect_stderr(error), self.assertRaises(SystemExit) as exc:
                self.main([
                    "--background", str(source), "--signal", str(source),
                    "--output-dir", str(root / "output"),
                ])
            self.assertEqual(exc.exception.code, 2)
            self.assertIn("Background and signal inputs must be different files", error.getvalue())
            self.assertFalse((root / "output").exists())


if __name__ == "__main__":
    unittest.main()
