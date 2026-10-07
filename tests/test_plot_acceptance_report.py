"""Tests for the structured-report acceptance-plane plotter."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
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


if __name__ == "__main__":
    unittest.main()
