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
import weakref


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

    def chunk_inputs(self, root):
        """Unequal chunks with real row identities and several tensor layouts."""
        files = [[], []]
        samples = []
        for role, original, boundaries in ((0, self.minbias, (0, 7, 20)),
                                            (1, self.zz, (0, 4, 10))):
            count = len(original["label"])
            sample = {**original,
                      "pair_obj": original["obj"].clone(),
                      "pair_slot_mask": self.torch.ones((count, 2), dtype=self.torch.bool),
                      "raw_source_file": self.torch.arange(count) // 3,
                      "raw_source_row": self.torch.arange(count) % 3,
                      "pair_obj_columns": ["pt_1", "pt_2"]}
            samples.append(sample)
            name = "background" if role == 0 else "signal"
            for index, (start, stop) in enumerate(zip(boundaries, boundaries[1:])):
                path = root / f"{name}_{index:05d}.pt"
                self.torch.save({key: value[start:stop] if isinstance(value, self.torch.Tensor) else value
                                 for key, value in sample.items()}, path)
                files[role].append(path)
        args = ["--background", str(root / "background_*.pt"),
                "--signal", str(root / "signal_*.pt"),
                "--output-dir", str(root / "output"),
                "--reference-count", "8", "--background-count", "5", "--signal-count", "3"]
        return files, samples, args

    def load_results(self, directory):
        return tuple(self.torch.load(directory / f"{name}.pt", weights_only=True)
                     for name in ("reference", "anomalous"))

    def assert_datasets_equal(self, first, second):
        for left, right in zip(first, second):
            self.assertEqual(left.keys(), right.keys())
            for key, value in left.items():
                if isinstance(value, self.torch.Tensor):
                    self.assertTrue(self.torch.equal(value, right[key]), key)
                else:
                    self.assertEqual(value, right[key], key)

    def test_quoted_wildcards_preserve_all_tensors_sampling_and_file_row_provenance(self):
        with tempfile.TemporaryDirectory(prefix="mixture with spaces ") as temporary:
            root = Path(temporary)
            groups, originals, args = self.chunk_inputs(root)
            before_rng = self.torch.random.get_rng_state().clone()
            with redirect_stdout(io.StringIO()):
                self.main(args)
            self.assertTrue(self.torch.equal(before_rng, self.torch.random.get_rng_state()))
            results = self.load_results(root / "output")
            baseline = self.make_datasets(*originals, **self.options)
            inputs = groups[0] + groups[1]
            source_chunks = [self.torch.load(path, weights_only=True) for path in inputs]
            for result, expected in zip(results, baseline):
                self.assertEqual(result["source_files"], [str(path) for path in inputs])
                for key in originals[0]:
                    if isinstance(expected[key], self.torch.Tensor):
                        self.assertTrue(self.torch.equal(result[key], expected[key]), key)
                    else:
                        self.assertEqual(result[key], expected[key], key)
                self.assertTrue(self.torch.equal(result["source_id"], expected["source_id"]))
                ids = list(zip(result["source_file_index"].tolist(), result["source_index"].tolist()))
                self.assertEqual(len(ids), len(set(ids)))
                for row, (file_index, source_row) in enumerate(ids):
                    self.assertEqual(int(result["source_id"][row]), int(file_index >= len(groups[0])))
                    for key, value in source_chunks[file_index].items():
                        if isinstance(value, self.torch.Tensor):
                            self.assertTrue(self.torch.equal(result[key][row], value[source_row]), key)
            reference_ids = set(zip(results[0]["source_file_index"].tolist(),
                                    results[0]["source_index"].tolist()))
            background_mask = results[1]["source_id"] == 0
            mixture_ids = set(zip(results[1]["source_file_index"][background_mask].tolist(),
                                  results[1]["source_index"][background_mask].tolist()))
            self.assertFalse(reference_ids & mixture_ids)

    def test_explicit_file_lists_match_quoted_patterns_and_are_reproducible(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            groups, _, args = self.chunk_inputs(root)
            self.main(args)
            expected = self.load_results(root / "output")
            expanded = ["--background", *map(str, groups[0]), "--signal", *map(str, groups[1]),
                        *args[4:]]
            expanded[expanded.index("--output-dir") + 1] = str(root / "explicit")
            self.main(expanded)
            self.assert_datasets_equal(expected, self.load_results(root / "explicit"))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                self.main(expanded)
            self.assert_datasets_equal(expected, self.load_results(root / "explicit"))

    def test_single_file_cli_matches_existing_python_api_exactly(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            inputs, args = self.cli_inputs(root)
            self.main(args)
            expected = self.make_datasets(self.minbias, self.zz, **self.options)
            for sample in expected:
                sample["source_files"] = [str(path) for path in inputs]
            self.assert_datasets_equal(expected, self.load_results(root / "output"))

    def test_patterns_sort_matches_and_deduplicate_paths_symlinks_and_hardlinks(self):
        from scripts.make_reference_anomalous import _expand_inputs
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            groups, _, _ = self.chunk_inputs(root)
            link = root / "alias.pt"
            hardlink = root / "hardlink.pt"
            link.symlink_to(groups[0][0])
            os.link(groups[0][1], hardlink)
            self.assertEqual(
                _expand_inputs([root / "background_*.pt", groups[0][0], link, hardlink]), groups[0])
            self.assertEqual(_expand_inputs([groups[0][1], root / "background_*.pt"]),
                             [groups[0][1], groups[0][0]])

    def test_unmatched_patterns_and_missing_literals_fail_before_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, args = self.cli_inputs(root)
            for expression in ("missing_*.pt", "missing.pt"):
                with self.subTest(expression=expression), redirect_stderr(io.StringIO()), \
                        patch("scripts.make_reference_anomalous.torch.load") as load:
                    with self.assertRaises(SystemExit):
                        self.main([*args, "--background", args[1], str(root / expression)])
                    load.assert_not_called()
                    self.assertFalse((root / "output").exists())

    def test_pool_overlap_detects_nonfirst_files_and_file_aliases(self):
        for alias in ("same_path", "symlink", "hardlink"):
            with self.subTest(alias=alias), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                groups, _, args = self.chunk_inputs(root)
                shared = groups[0][1]
                if alias != "same_path":
                    shared = root / "alias.pt"
                    if alias == "symlink":
                        shared.symlink_to(groups[0][1])
                    else:
                        os.link(groups[0][1], shared)
                with redirect_stderr(io.StringIO()), \
                        patch("scripts.make_reference_anomalous.torch.load") as load:
                    with self.assertRaises(SystemExit):
                        self.main([*args, "--signal", *map(str, groups[1]), str(shared)])
                    load.assert_not_called()
                self.assertFalse((root / "output").exists())

    def test_later_schema_error_preserves_existing_outputs(self):
        for error in ("metadata", "dtype", "shape", "keys", "alignment", "provenance"):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                groups, _, args = self.chunk_inputs(root)
                bad = self.torch.load(groups[1][1], weights_only=True)
                if error == "metadata":
                    bad["obj_columns"] = ["eta", "pt"]
                elif error == "dtype":
                    bad["pair_obj"] = bad["pair_obj"].to(self.torch.float64)
                elif error == "shape":
                    bad["pair_obj"] = bad["pair_obj"][:, :1]
                elif error == "keys":
                    del bad["pair_obj"]
                elif error == "alignment":
                    bad["event_features"] = bad["event_features"][:-1]
                else:
                    bad["source_file_index"] = self.torch.zeros(len(bad["label"]), dtype=self.torch.int64)
                self.torch.save(bad, groups[1][1])
                output = root / "output"
                output.mkdir()
                targets = [output / f"{name}.pt" for name in ("reference", "anomalous")]
                for path in targets:
                    path.write_bytes(b"preserve previous output")
                with redirect_stderr(io.StringIO()), warnings.catch_warnings(), \
                        patch("scripts.make_reference_anomalous.torch.save") as save:
                    warnings.simplefilter("ignore", UserWarning)
                    with self.assertRaises(SystemExit):
                        self.main(args)
                    save.assert_not_called()
                self.assertTrue(all(path.read_bytes() == b"preserve previous output" for path in targets))

    def test_pool_size_checks_happen_before_publication(self):
        for flag, value in (("--reference-count", "16"), ("--signal-count", "11")):
            with self.subTest(flag=flag), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                _, _, args = self.chunk_inputs(root)
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    self.main([*args, flag, value])
                self.assertFalse((root / "output").exists())

    def test_multiple_inputs_keep_output_protection_and_no_overwrite(self):
        for mode in ("input_alias", "no_overwrite"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                groups, _, args = self.chunk_inputs(root)
                output = root / "output"
                output.mkdir()
                target = output / "reference.pt"
                if mode == "input_alias":
                    os.link(groups[0][1], target)
                else:
                    target.write_bytes(b"previous reference")
                    args.append("--no-overwrite")
                before = target.read_bytes()
                with redirect_stderr(io.StringIO()), \
                        patch("scripts.make_reference_anomalous.torch.load") as load:
                    with self.assertRaises(SystemExit):
                        self.main(args)
                    load.assert_not_called()
                self.assertEqual(target.read_bytes(), before)
                self.assertFalse((output / "anomalous.pt").exists())

    def test_empty_chunks_are_allowed_and_not_selected(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            _, originals, args = self.chunk_inputs(root)
            empty = root / "empty.pt"
            self.torch.save({key: value[:0] if isinstance(value, self.torch.Tensor) else value
                             for key, value in originals[0].items()}, empty)
            self.main([*args, "--background", str(empty), args[1]])
            for sample in self.load_results(root / "output"):
                self.assertFalse(bool((sample["source_file_index"] == 0).any()))

    def test_only_one_source_file_is_held_and_unselected_files_are_not_reloaded(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            groups, _, args = self.chunk_inputs(root)
            references, calls = [], []
            original_load = self.torch.load

            def load(path, **options):
                self.assertTrue(all(reference() is None for reference in references))
                sample = original_load(path, **options)
                references.extend(weakref.ref(value) for value in sample.values()
                                  if isinstance(value, self.torch.Tensor))
                calls.append(path)
                return sample

            with patch("scripts.make_reference_anomalous.torch.load", new=load):
                self.main([*args, "--reference-count", "1", "--background-count", "1", "--signal-count", "1"])
            inputs = groups[0] + groups[1]
            self.assertEqual(calls[:len(inputs)], inputs)
            self.assertGreater(len(calls), len(inputs))
            self.assertLess(len(calls), 2 * len(inputs))
            self.assertTrue(all(reference() is None for reference in references))


if __name__ == "__main__":
    unittest.main()
