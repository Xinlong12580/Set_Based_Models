"""Bounded output chunks, aligned provenance, and safe publication; no EOS."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings
import weakref

import awkward as ak
import numpy as np
import torch

from raw_event_conversion import convert_raw_parquet_to_pt
from raw_event_conversion import convert as conversion
from raw_event_conversion.__main__ import main
from tests.test_raw_event_conversion import raw_fixture


class ChunkedConversionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="raw_chunked_conversion_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "input.parquet"
        self.output = self.root / "events.pt"
        ak.to_parquet(raw_fixture(), self.source)

    def catalog(self):
        return json.loads(self.output.with_suffix(".chunks.json").read_text())

    def load(self, path):
        return torch.load(path, map_location="cpu", weights_only=True)

    def test_compulsory_real_10000_boundary_and_source_rows(self):
        self.assertEqual(conversion.MAX_EVENTS_PER_OUTPUT_FILE, 10000)
        # Empty object/PF collections keep this real-boundary test inexpensive.
        ak.to_parquet(raw_fixture()[np.zeros(10001, dtype=np.int64)], self.source)
        outputs = convert_raw_parquet_to_pt(self.source, self.output, label=1,
                                           max_events=10001, max_events_per_file=20000)
        self.assertEqual([path.name for path in outputs], ["events_00000.pt", "events_00001.pt"])
        self.assertFalse(self.output.exists())
        for index, path in enumerate(outputs):
            sample = self.load(path)
            count = 10000 if index == 0 else 1
            self.assertEqual(len(sample["label"]), count)
            self.assertTrue(all(value.shape[0] == count for value in sample.values()
                                if isinstance(value, torch.Tensor)))
            self.assertEqual(sample["raw_source_file"].unique().tolist(), [0])
            self.assertEqual(sample["raw_source_row"].tolist(), list(range(index * 10000, index * 10000 + count)))
            self.assertEqual(json.loads(path.with_suffix(".sources.json").read_text()), [str(self.source)])
        index = self.catalog()
        self.assertEqual(index["event_count"], 10001)
        self.assertEqual(index["max_events_per_output_file"], 10000)
        self.assertEqual([entry["event_count"] for entry in index["files"]], [10000, 1])
        self.assertEqual([entry["path"] for entry in index["files"]], [path.name for path in outputs])

    def test_exactly_10000_keeps_single_filename_and_path_return(self):
        ak.to_parquet(raw_fixture()[np.zeros(10000, dtype=np.int64)], self.source)
        result = convert_raw_parquet_to_pt(self.source, self.output, label=1)
        self.assertEqual(result, self.output)
        self.assertEqual(len(self.load(result)["label"]), 10000)
        self.assertEqual([entry["path"] for entry in self.catalog()["files"]], ["events.pt"])
        self.assertFalse((self.root / "events_00000.pt").exists())

    def test_chunks_match_single_output_for_all_tensors_and_metadata(self):
        second = self.root / "second.parquet"
        ak.to_parquet(raw_fixture(), second)
        args = dict(label=5, max_events=5, max_events_per_file=3, include_pair_tensor=True)
        baseline_path = self.root / "baseline.pt"
        convert_raw_parquet_to_pt([self.source, second], baseline_path, **args)
        baseline = self.load(baseline_path)
        with patch.object(conversion, "MAX_EVENTS_PER_OUTPUT_FILE", 2):
            outputs = convert_raw_parquet_to_pt([self.source, second], self.output, **args)
        chunks = [self.load(path) for path in outputs]
        self.assertEqual([len(chunk["label"]) for chunk in chunks], [2, 2, 1])
        for key, value in baseline.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(torch.cat([chunk[key] for chunk in chunks]), value)
            else:
                self.assertTrue(all(chunk[key] == value for chunk in chunks), key)
        self.assertEqual(chunks[1]["raw_source_file"].tolist(), [0, 1])
        self.assertEqual(chunks[1]["raw_source_row"].tolist(), [2, 0])
        self.assertEqual(chunks[2]["raw_source_row"].tolist(), [1])

    def test_finished_chunk_tensors_are_released_before_building_next(self):
        references = []
        original_save = torch.save
        original_build = conversion.build_event_tensor

        def save(sample, path):
            references.append(weakref.ref(sample["pair_obj"]))
            return original_save(sample, path)

        def build(*args, **kwargs):
            self.assertTrue(all(reference() is None for reference in references))
            return original_build(*args, **kwargs)

        with patch.object(conversion, "MAX_EVENTS_PER_OUTPUT_FILE", 2), \
                patch.object(conversion, "build_event_tensor", new=build), \
                patch.object(conversion.torch, "save", new=save):
            outputs = convert_raw_parquet_to_pt(self.source, self.output, label=1, include_pair_tensor=True)
        self.assertEqual(len(outputs), 2)
        self.assertTrue(all(reference() is None for reference in references))

    def test_later_serialization_failure_preserves_previous_dataset(self):
        convert_raw_parquet_to_pt(self.source, self.output, label=1)
        protected = [self.output, self.output.with_suffix(".sources.json"), self.output.with_suffix(".chunks.json")]
        before = [path.read_bytes() for path in protected]
        original_save = torch.save
        saved = 0

        def save(sample, path):
            nonlocal saved
            saved += 1
            if saved == 2:
                raise RuntimeError("second chunk failed")
            return original_save(sample, path)

        with patch.object(conversion, "MAX_EVENTS_PER_OUTPUT_FILE", 2), \
                patch.object(conversion.torch, "save", side_effect=save):
            with self.assertRaisesRegex(RuntimeError, "second chunk failed"):
                convert_raw_parquet_to_pt(self.source, self.output, label=5)
        self.assertEqual([path.read_bytes() for path in protected], before)
        self.assertEqual(list(self.root.glob("events_*.pt")), [])
        self.assertEqual(list(self.root.glob(".raw_event_conversion_*")), [])

    def test_runtime_shortage_after_staging_a_chunk_publishes_nothing(self):
        second = self.root / "short.parquet"
        ak.to_parquet(raw_fixture()[:1], second)
        original_read = ak.from_parquet
        staged_before_second = []

        def read(path, **kwargs):
            if path == str(second):
                staged_before_second.extend(self.root.glob(".raw_event_conversion_*/chunk_00000.pt"))
            return original_read(path, **kwargs)

        with patch.object(conversion, "MAX_EVENTS_PER_OUTPUT_FILE", 2), \
                patch.object(conversion.ak, "from_parquet", side_effect=read):
            with self.assertRaisesRegex(ValueError, "at most 4 total"):
                convert_raw_parquet_to_pt([self.source, second], self.output, label=1,
                                          max_events=6, max_events_per_file=3)
        self.assertEqual(len(staged_before_second), 1)
        self.assertEqual(list(self.root.glob("*.pt")), [])
        self.assertEqual(list(self.root.glob("*.json")), [])
        self.assertEqual(list(self.root.glob(".raw_event_conversion_*")), [])

    def test_no_overwrite_checks_indexed_targets_before_publishing(self):
        existing = self.root / "events_00001.pt"
        existing.write_bytes(b"keep this")
        with patch.object(conversion, "MAX_EVENTS_PER_OUTPUT_FILE", 2):
            with self.assertRaises(FileExistsError):
                convert_raw_parquet_to_pt(self.source, self.output, label=1, overwrite=False)
        self.assertEqual(existing.read_bytes(), b"keep this")
        self.assertFalse((self.root / "events_00000.pt").exists())
        self.assertFalse(self.output.with_suffix(".chunks.json").exists())

    def test_indexed_output_symlinks_and_input_hardlinks_are_protected(self):
        target = self.root / "events_00000.pt"
        import os
        for symbolic in (False, True):
            with self.subTest(symbolic=symbolic):
                if symbolic:
                    target.symlink_to(self.source)
                else:
                    os.link(self.source, target)
                before = self.source.read_bytes()
                with patch.object(conversion, "MAX_EVENTS_PER_OUTPUT_FILE", 2):
                    with self.assertRaisesRegex(ValueError, "symbolic link|input file"):
                        convert_raw_parquet_to_pt(self.source, self.output, label=1)
                self.assertEqual(self.source.read_bytes(), before)
                self.assertFalse((self.root / "events_00001.pt").exists())
                target.unlink()

    def test_rerun_catalog_excludes_older_extra_chunks_without_deleting_them(self):
        ak.to_parquet(raw_fixture()[[0, 1, 2, 0, 1]], self.source)
        with patch.object(conversion, "MAX_EVENTS_PER_OUTPUT_FILE", 2):
            previous = convert_raw_parquet_to_pt(self.source, self.output, label=1)
            obsolete = previous[-1]
            before = obsolete.read_bytes()
            ak.to_parquet(raw_fixture(), self.source)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                current = convert_raw_parquet_to_pt(self.source, self.output, label=5)
        self.assertEqual(obsolete.read_bytes(), before)
        self.assertEqual([entry["path"] for entry in self.catalog()["files"]], [path.name for path in current])
        self.assertNotIn(obsolete.name, [entry["path"] for entry in self.catalog()["files"]])
        self.assertEqual(self.catalog()["event_count"], 3)

    def test_cli_prints_each_chunk_filename(self):
        output = io.StringIO()
        with patch.object(conversion, "MAX_EVENTS_PER_OUTPUT_FILE", 2), redirect_stdout(output):
            main(["--input", str(self.source), "--output", str(self.output), "--label", "1"])
        for index in range(2):
            self.assertIn(f"Saved {self.root / f'events_{index:05d}.pt'}", output.getvalue())

    def test_older_single_output_is_not_mistaken_for_new_chunked_dataset(self):
        convert_raw_parquet_to_pt(self.source, self.output, label=1)
        before = self.output.read_bytes()
        with patch.object(conversion, "MAX_EVENTS_PER_OUTPUT_FILE", 2), \
                warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            outputs = convert_raw_parquet_to_pt(self.source, self.output, label=5)
        self.assertEqual(self.output.read_bytes(), before)
        self.assertTrue(any("does not update the older single file" in str(item.message) for item in caught))
        self.assertEqual([item["path"] for item in self.catalog()["files"]], [path.name for path in outputs])
        self.assertNotIn(self.output.name, [item["path"] for item in self.catalog()["files"]])

    def test_pair_enabled_conversion_chunks_can_be_mixed_with_wildcards(self):
        from scripts.make_reference_anomalous import main as mix

        with patch.object(conversion, "MAX_EVENTS_PER_OUTPUT_FILE", 2):
            convert_raw_parquet_to_pt(self.source, self.root / "background.pt", label=1,
                                      include_pair_tensor=True)
            convert_raw_parquet_to_pt(self.source, self.root / "signal.pt", label=2,
                                      include_pair_tensor=True)
        with redirect_stdout(io.StringIO()):
            mix(["--background", str(self.root / "background_*.pt"),
                 "--signal", str(self.root / "signal_*.pt"),
                 "--output-dir", str(self.root / "mixture"),
                 "--reference-count", "1", "--background-count", "1", "--signal-count", "1"])
        ids = []
        for name in ("reference", "anomalous"):
            sample = self.load(self.root / "mixture" / f"{name}.pt")
            for row, (file_index, source_row) in enumerate(zip(sample["source_file_index"].tolist(),
                                                             sample["source_index"].tolist())):
                source = self.load(sample["source_files"][file_index])
                if int(sample["source_id"][row]) == 0:
                    ids.append((file_index, source_row))
                for key, value in source.items():
                    if isinstance(value, torch.Tensor):
                        torch.testing.assert_close(sample[key][row], value[source_row])
                    else:
                        self.assertEqual(sample[key], value)
        self.assertEqual(len(ids), 2)
        self.assertEqual(len(set(ids)), 2)


if __name__ == "__main__":
    unittest.main()
