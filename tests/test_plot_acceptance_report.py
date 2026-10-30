"""Tests for the structured-report acceptance-plane plotter."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Matplotlib is optional")
class AcceptanceReportPlotTests(unittest.TestCase):
    def report(self):
        def rule(rank, rule_id, signal, background, direction, retained):
            return {
                "rank": rank,
                "rule_id": rule_id,
                "gini_gain": 0.1 / rank,
                "simplified_rule": f"EXISTS same slot: pt > {10 * rank}",
                "retained_after_similarity_pruning": retained,
                "metrics": {
                    "signal_acceptance": signal,
                    "background_acceptance": background,
                    "acceptance_difference": signal - background,
                    "direction": direction,
                },
            }

        return {
            "format": "set_based_models_quantitative_report",
            "schema_version": 1,
            "model": "ERL_Gini",
            "inputs": {"signal": {"path": "/data/example/anomalous.pt"}},
            "results": {
                "rules": [
                    rule(1, 0, 0.20, 0.01, "mixture-enriched", True),
                    rule(2, 1, 0.15, 0.02, "mixture-enriched", False),
                    rule(3, 2, 0.50, 0.70, "reference-enriched", True),
                ]
            },
        }

    def test_selection_defaults_to_pruned_mixture_rules(self):
        from scripts.plot_acceptance_report import select_paths

        eligible, selected = select_paths(
            self.report(), rules="pruned", direction="mixture", top=10
        )
        self.assertEqual([rule["rule_id"] for rule in eligible], [0])
        self.assertEqual(selected, eligible)

    def test_cli_writes_png_from_json_text_report(self):
        from scripts.plot_acceptance_report import main

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = root / "report.txt"
            output = root / "acceptance.png"
            report.write_text(json.dumps(self.report()), encoding="utf-8")
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                main([str(report), "--output", str(output), "--top", "1"])
            self.assertTrue(output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertIn(str(output), stdout.getvalue())

    def test_training_report_default_plot_name(self):
        from scripts.plot_acceptance_report import main

        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "training.report.json"
            report.write_text(json.dumps(self.report()), encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                main([str(report), "--top", "1"])
            self.assertTrue((report.parent / "training.acceptance.png").read_bytes().startswith(b"\x89PNG"))

    def test_batch_plotting_includes_training_and_legacy_reports_but_not_models(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in ("training.report.json", "training.model.json", "legacy.txt"):
                (directory / name).write_text("{}")
            calls = directory / "calls.jsonl"
            python = directory / "mock-python"
            python.write_text(f"#!{sys.executable}\nimport json, sys\nfrom pathlib import Path\n"
                              f"with Path({str(calls)!r}).open('a') as handle:\n    handle.write(json.dumps(sys.argv[1:]) + '\\n')\n")
            python.chmod(0o755)
            result = subprocess.run(["bash", str(root / "scripts/plot_all_reports.sh"), str(directory)],
                                    env={**os.environ, "PYTHON_BIN": str(python)}, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            args = [json.loads(line) for line in calls.read_text().splitlines()]
            self.assertEqual([Path(call[1]).name for call in args], ["training.report.json", "legacy.txt"])
            self.assertEqual([Path(call[call.index("--output") + 1]).name for call in args],
                             ["training.acceptance.png", "legacy_acceptance.png"])


if __name__ == "__main__":
    unittest.main()
