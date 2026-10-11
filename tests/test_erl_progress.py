"""Core progress callbacks and optional stderr rendering do not change rules."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from casual_tests import run_ERL_Gini_nested as runner
from casual_tests._erl_progress import SearchProgress
from set_based_models import ExistentialRuleList, ObservableSpec, TPRFPRRuleScore
from tests.test_pair_erl_runner import fixture


class CoreProgressTests(unittest.TestCase):
    def model(self, **kwargs):
        return ExistentialRuleList((ObservableSpec("pt", "continuous"),),
                                   max_rule_depth=3, beam_width=4, max_thresholds=4,
                                   scoring_function=TPRFPRRuleScore(), **kwargs)

    def test_callbacks_count_candidates_and_preserve_search_results(self):
        signal = np.array([[[50.], [10.]], [[40.], [5.]], [[30.], [15.]]])
        background = np.array([[[20.], [10.]], [[15.], [5.]], [[30.], [5.]]])
        for similarity in (False, True):
            with self.subTest(similarity=similarity):
                quiet = self.model(beam_similarity_filter=similarity).fit(signal, background)
                updates = []
                observed = self.model(beam_similarity_filter=similarity).fit(
                    signal, background, progress_callback=updates.append)
                self.assertEqual(len(quiet.candidate_rules_), len(observed.candidate_rules_))
                for first, second in zip(quiet.candidate_rules_, observed.candidate_rules_):
                    self.assertEqual(first.cuts, second.cuts)
                    self.assertEqual(first.statistics, second.statistics)
                    self.assertEqual(first.generation_score, second.generation_score)
                    np.testing.assert_array_equal(first.signal_selection, second.signal_selection)
                    np.testing.assert_array_equal(first.background_selection, second.background_selection)
                stages = {update["stage"] for update in updates}
                self.assertTrue({"depth_started", "generating_cuts", "evaluating_cuts",
                                 "parent_finished", "filtering_beam", "depth_finished", "finished"} <= stages)
                self.assertEqual(updates[-1]["candidate_rules"], len(observed.candidate_rules_))
                self.assertEqual(updates[-1]["stage"], "finished")
                counts = [update["evaluated_cuts"] for update in updates]
                self.assertEqual(counts, sorted(counts))
                self.assertGreater(counts[-1], 0)
                self.assertEqual([update["depth"] for update in updates if update["stage"] == "depth_started"], [1, 2, 3])
                for update in updates:
                    self.assertLessEqual(update["parents_completed"], update["parents_total"])
                    self.assertLessEqual(update["cuts_completed"], update["cuts_total"])
                    if update["stage"] == "filtering_beam":
                        self.assertLessEqual(update["filter_completed"], update["filter_total"])

    def test_empty_search_reports_early_completion_not_fake_remaining_depths(self):
        updates = []
        data = np.ones((3, 1, 1))
        model = self.model().fit(data, data, progress_callback=updates.append)
        self.assertEqual(model.candidate_rules_, [])
        self.assertEqual(updates[-1]["stage"], "finished")
        self.assertEqual(updates[-1]["depth"], 1)
        self.assertEqual(updates[-1]["max_depth"], 3)
        self.assertEqual(updates[-1]["evaluated_cuts"], 0)

    def test_core_is_silent_even_with_a_nonprinting_callback(self):
        data = np.ones((3, 1, 1))
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            self.model().fit(data, data)
            self.model().fit(data, data, progress_callback=lambda status: None)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "")

    def test_bad_callback_is_rejected_before_input_validation(self):
        for callback in (False, 42, "progress"):
            with self.subTest(callback=callback), self.assertRaisesRegex(TypeError, "progress_callback"):
                self.model().fit(progress_callback=callback)


def status(stage="evaluating_cuts", **changes):
    return {"stage": stage, "depth": 2, "max_depth": 3, "parents_completed": 2,
            "parents_total": 5, "cuts_completed": 50, "cuts_total": 100,
            "evaluated_cuts": 1000, "retained": 4, "filter_completed": 10,
            "filter_total": 20, "candidate_rules": 8, **changes}


class ProgressDisplayTests(unittest.TestCase):
    def test_non_tty_uses_readable_lines_without_escape_codes_or_eta(self):
        output = io.StringIO()
        with SearchProgress(output, min_interval=0) as progress:
            progress(status())
            progress(status("filtering_beam"))
            progress(status("finished"))
        text = output.getvalue()
        self.assertIn("depth 2/3", text)
        self.assertIn("parents 2/5", text)
        self.assertIn("cuts 50/100", text)
        self.assertIn("filtering beam 10/20", text)
        self.assertIn("evaluated 1000", text)
        self.assertIn("search complete (8 rules)", text)
        self.assertNotIn("\r", text)
        self.assertNotIn("\x1b", text)
        self.assertNotIn("ETA", text)

    def test_rendering_is_throttled_but_completion_is_always_printed(self):
        output = io.StringIO()
        with patch("casual_tests._erl_progress.perf_counter", side_effect=[0., 0., .1, .2, 2.1, 2.2]):
            with SearchProgress(output, min_interval=2) as progress:
                progress(status("depth_started"))
                progress(status(cuts_completed=1))
                progress(status(cuts_completed=2))
                progress(status(cuts_completed=3))
                progress(status("finished"))
        text = output.getvalue()
        self.assertNotIn("cuts 1/100", text)
        self.assertNotIn("cuts 2/100", text)
        self.assertIn("cuts 3/100", text)
        self.assertIn("search complete", text)
        self.assertEqual(len(text.splitlines()), 4)

    def test_tty_updates_in_place_and_closes_on_interrupt(self):
        output = io.StringIO()
        output.isatty = lambda: True
        with self.assertRaises(KeyboardInterrupt):
            with SearchProgress(output, min_interval=0) as progress:
                progress(status())
                raise KeyboardInterrupt()
        self.assertIn("\rERL depth", output.getvalue())
        self.assertTrue(output.getvalue().endswith("\n"))

    def test_finished_then_context_exit_only_emits_one_terminal_newline(self):
        output = io.StringIO()
        output.isatty = lambda: True
        with SearchProgress(output) as progress:
            progress(status("finished"))
        self.assertEqual(output.getvalue().count("\n"), 1)


class RunnerProgressTests(unittest.TestCase):
    def test_progress_works_for_all_modes_and_preserves_rules_and_metrics(self):
        with tempfile.TemporaryDirectory(prefix="erl_progress_") as directory:
            root = Path(directory)
            signal, background = root / "anomalous.pt", root / "reference.pt"
            torch.save(fixture(True), signal)
            torch.save(fixture(False), background)
            modes = ([], ["--object-only"], ["--event-only"],
                     ["--pair-tensor", "--object-only", "--object-cut-priority", "type_id_1", "type_id_2"])
            for index, options in enumerate(modes):
                reports = []
                for enabled in (False, True):
                    target = root / f"report_{index}_{enabled}.txt"
                    stdout, stderr = io.StringIO(), io.StringIO()
                    with redirect_stdout(stdout), redirect_stderr(stderr):
                        runner.main(["--signal", str(signal), "--background", str(background),
                                     "--report", str(target), "--print-top", "1", "--print-reference-top", "0",
                                     "--print-pruned-top", "1", *options, *(["--progress"] if enabled else [])])
                    reports.append(json.loads(target.read_text()))
                    self.assertNotIn("ERL depth", stdout.getvalue())
                    self.assertEqual(reports[-1]["configuration"]["progress"], enabled)
                    if enabled:
                        self.assertIn("ERL depth", stderr.getvalue())
                        self.assertIn("search complete", stderr.getvalue())
                    else:
                        self.assertEqual(stderr.getvalue(), "")
                self.assertEqual(reports[0]["results"], reports[1]["results"])
                self.assertEqual(reports[0]["summary"], reports[1]["summary"])


if __name__ == "__main__":
    unittest.main()
