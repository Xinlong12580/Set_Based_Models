"""Conversion progress rendering and wiring, using only local fixtures."""

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import awkward as ak
import torch

from raw_event_conversion import convert_raw_parquet_to_pt
from raw_event_conversion import convert as conversion
from raw_event_conversion._progress import ConversionProgress
from raw_event_conversion.__main__ import main
from tests.test_raw_event_conversion import raw_fixture


class TerminalStream(io.StringIO):
    def isatty(self):
        return True


class ConversionProgressTests(unittest.TestCase):
    def test_event_bar_counts_and_completion(self):
        stream = io.StringIO()
        with ConversionProgress(4, 10000, stream=stream) as status:
            status.update("converting", files=2, events=5000, chunks=1, force=True)
        text = stream.getvalue()
        self.assertIn("[##########----------] events 5,000/10,000", text)
        self.assertIn("files 2/4 | chunks staged 1 | converting", text)
        self.assertIn("| done | elapsed", text)
        self.assertNotIn("\r", text)

    def test_unknown_event_total_uses_file_bar(self):
        stream = io.StringIO()
        with ConversionProgress(4, stream=stream) as status:
            status.update("file converted", files=2, events=300, force=True)
        self.assertIn("[##########----------] events 300 | files 2/4", stream.getvalue())

    def test_throttling_retains_counts_for_final_line(self):
        stream = io.StringIO()
        with patch("raw_event_conversion._progress.perf_counter", return_value=0.0) as clock:
            with ConversionProgress(4, stream=stream) as status:
                clock.return_value = 0.1
                status.update("file converted", files=1, events=7)
                self.assertNotIn("events 7", stream.getvalue())
                clock.return_value = 2.1
                status.update("reading next file")
                self.assertIn("events 7 | files 1/4", stream.getvalue())
                status.update("chunk staged", chunks=1)
        self.assertIn("chunks staged 1 | done", stream.getvalue())

    def test_terminal_updates_one_line_and_closes_on_failure(self):
        stream = TerminalStream()
        with self.assertRaisesRegex(RuntimeError, "read failed"):
            with ConversionProgress(2, stream=stream) as status:
                status.update("reading input.parquet", force=True)
                raise RuntimeError("read failed")
        text = stream.getvalue()
        self.assertIn("\rConvert [", text)
        self.assertIn("failed/interrupted", text)
        self.assertNotIn("| done |", text)
        self.assertTrue(text.endswith("\n"))
        self.assertEqual(text.count("\n"), 1)

    def test_disabled_progress_is_silent(self):
        stream = io.StringIO()
        with ConversionProgress(1, enabled=False, stream=stream) as status:
            status.update("converting", files=1, events=3, chunks=1, force=True)
        self.assertEqual(stream.getvalue(), "")


class ConversionProgressIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="raw_conversion_progress_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "input.parquet"
        self.output = self.root / "events.pt"
        ak.to_parquet(raw_fixture(), self.source)

    def test_api_is_silent_by_default_and_progress_preserves_all_tensors(self):
        stderr, stdout = io.StringIO(), io.StringIO()
        with redirect_stderr(stderr), redirect_stdout(stdout):
            convert_raw_parquet_to_pt(self.source, self.output, label=1, include_pair_tensor=True)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(stdout.getvalue(), "")
        visible_output = self.root / "visible.pt"
        with redirect_stderr(stderr), redirect_stdout(stdout):
            convert_raw_parquet_to_pt(self.source, visible_output, label=1,
                                      include_pair_tensor=True, progress=True)
        self.assertIn("| done |", stderr.getvalue())
        self.assertEqual(stdout.getvalue(), "")
        baseline = torch.load(self.output, weights_only=True)
        visible = torch.load(visible_output, weights_only=True)
        for key, value in baseline.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(visible[key], value)
            else:
                self.assertEqual(visible[key], value)

    def test_cli_progress_default_and_switches(self):
        for flags, expected in (([], True), (["--progress"], True), (["--no-progress"], False)):
            with self.subTest(flags=flags), \
                    patch("raw_event_conversion.__main__.convert_raw_parquet_to_pt",
                          return_value=self.output) as convert, redirect_stdout(io.StringIO()):
                main(["--input", str(self.source), "--output", str(self.output), "--label", "1", *flags])
                self.assertEqual(convert.call_args.kwargs["progress"], expected)

    def test_progress_tracks_chunks_and_stops_at_total_cap(self):
        absent = self.root / "not_needed.parquet"
        stream = io.StringIO()
        with patch.object(conversion, "MAX_EVENTS_PER_OUTPUT_FILE", 2), redirect_stderr(stream):
            outputs = convert_raw_parquet_to_pt([self.source, absent], self.output, label=1,
                                               max_events=3, progress=True)
        self.assertEqual(len(outputs), 2)
        text = stream.getvalue()
        self.assertIn("reading input.parquet", text)
        self.assertIn("saving chunk", text)
        self.assertIn("publishing outputs", text)
        self.assertIn("[####################] events 3/3 | files 1/2 | chunks staged 2 | done", text)
        self.assertTrue(self.output.with_suffix(".chunks.json").exists())

    def test_read_error_reports_failure_and_publishes_nothing(self):
        stream = io.StringIO()
        with redirect_stderr(stream), \
                patch.object(conversion.ak, "from_parquet", side_effect=OSError("read failed")):
            with self.assertRaisesRegex(OSError, "read failed"):
                convert_raw_parquet_to_pt(self.source, self.output, label=1, progress=True)
        self.assertIn("failed/interrupted", stream.getvalue())
        self.assertNotIn("| done |", stream.getvalue())
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(".raw_event_conversion_*")), [])


if __name__ == "__main__":
    unittest.main()
