"""Structure and arithmetic checks for the synthetic documentation fixtures."""

from copy import deepcopy
import json
from pathlib import Path
import re
import unittest

try:
    import jsonschema
except ImportError:
    jsonschema = None


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "reports"
SCHEMA_PATH = ROOT / "schemas" / "quantitative_report.schema.json"


def load_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def gini_gain(signal, background, signal_total, background_total):
    def impurity(s, b):
        return 2 * s * b / (s + b) ** 2 if s + b else 0.0

    total = signal_total + background_total
    passed = signal + background
    return (impurity(signal_total, background_total)
            - passed / total * impurity(signal, background)
            - (total - passed) / total * impurity(
                signal_total - signal, background_total - background))


class ReportDocumentTests(unittest.TestCase):
    def setUp(self):
        self.erl = load_json(EXAMPLES / "erl_synthetic.json")
        self.edt = load_json(EXAMPLES / "edt_synthetic.json")

    def assert_metrics(self, metrics, inputs):
        s, b = metrics["signal_accepted"], metrics["background_accepted"]
        st, bt = inputs["signal"]["event_count"], inputs["background"]["event_count"]
        self.assertEqual(metrics["signal_total"], st)
        self.assertEqual(metrics["background_total"], bt)
        self.assertLessEqual(s, st)
        self.assertLessEqual(b, bt)
        sa, ba = s / st, b / bt
        self.assertAlmostEqual(metrics["signal_acceptance"], sa)
        self.assertAlmostEqual(metrics["background_acceptance"], ba)
        self.assertAlmostEqual(metrics["acceptance_difference"], sa - ba)
        if b:
            self.assertAlmostEqual(metrics["acceptance_ratio"], sa / ba)
        else:
            self.assertIsNone(metrics["acceptance_ratio"])
        self.assertAlmostEqual(metrics["mixture_label_purity"], s / (s + b) if s + b else 0)
        direction = "mixture-enriched" if sa > ba else "reference-enriched" if sa < ba else "equal-acceptance"
        self.assertEqual(metrics["direction"], direction)
        for role, count in (("signal", s), ("background", b)):
            selected = metrics[f"selected_{role}_stored_label_counts"]
            self.assertEqual(sum(selected.values()), count)
            for label, accepted in selected.items():
                self.assertLessEqual(accepted, inputs[role]["stored_label_counts"][label])

    def test_fixture_annotations_and_metrics(self):
        for report in (self.erl, self.edt):
            self.assertTrue(report["example_metadata"]["synthetic"])
            for dataset in report["inputs"].values():
                self.assertEqual(sum(dataset["stored_label_counts"].values()), dataset["event_count"])
                self.assertTrue(dataset["path"].startswith("synthetic/"))
            records = next(iter(report["results"].values()))
            for record in records:
                self.assert_metrics(record["metrics"], report["inputs"])

    def test_erl_prefix_scores_and_ranking(self):
        rules = self.erl["results"]["rules"]
        config = self.erl["configuration"]
        self.assertEqual([rule["rank"] for rule in rules], [1, 2, 3])
        self.assertEqual([rule["score"] for rule in rules], sorted((rule["score"] for rule in rules), reverse=True))
        for rule in rules:
            metrics = rule["metrics"]
            gain = gini_gain(metrics["signal_accepted"], metrics["background_accepted"], 100, 100)
            self.assertAlmostEqual(rule["gini_gain"], gain)
            self.assertAlmostEqual(rule["reference_penalty_term"], config["reference_penalty"] * metrics["background_acceptance"])
            self.assertAlmostEqual(rule["score"], gain - rule["reference_penalty_term"])
            self.assertAlmostEqual(rule["estimated_reference_rate_khz"], config["input_rate_mhz"] * 1000 * metrics["background_acceptance"])
            self.assertEqual(rule["depth"], len(rule["steps"]))
            self.assertEqual(rule["effective_depth_after_simplification"], len(rule["normalized_cuts"]))
            previous = (100, 100)
            for depth, step in enumerate(rule["steps"], 1):
                s, b = step["prefix_signal_accepted"], step["prefix_background_accepted"]
                self.assertEqual(step["depth"], depth)
                self.assertLessEqual(s, previous[0])
                self.assertLessEqual(b, previous[1])
                previous = (s, b)
                self.assertAlmostEqual(step["prefix_signal_acceptance"], s / 100)
                self.assertAlmostEqual(step["prefix_reference_acceptance"], b / 100)
                self.assertAlmostEqual(step["prefix_gini_gain"], gini_gain(s, b, 100, 100))
                self.assertAlmostEqual(step["prefix_reference_penalty_term"], config["reference_penalty"] * b / 100)
                self.assertAlmostEqual(step["prefix_score"], step["prefix_gini_gain"] - step["prefix_reference_penalty_term"])
            self.assertEqual(previous, (metrics["signal_accepted"], metrics["background_accepted"]))
            self.assertEqual(rule["direction"], metrics["direction"])

    def test_erl_pruning_and_completion(self):
        best, stricter, incomplete = self.erl["results"]["rules"]
        self.assertTrue(best["retained_after_final_pruning"])
        self.assertEqual(self.erl["summary"]["pruned_rule_ids"], [best["rule_id"]])
        self.assertEqual(stricter["pruning"]["status"], "stricter_score_improvement")
        self.assertEqual(stricter["pruning"]["pruned_by_rule_id"], best["rule_id"])
        improvement = stricter["score"] - best["score"]
        self.assertAlmostEqual(stricter["pruning"]["score_improvement"], improvement)
        self.assertAlmostEqual(stricter["pruning"]["score_gap"], -improvement)
        self.assertLess(improvement, self.erl["configuration"]["stricter_score_improvement"])
        self.assertGreater(improvement, -0.02)
        self.assertEqual(incomplete["object_cut_priority_remaining"], ["pt:>"])
        self.assertEqual(incomplete["pruning"]["status"], "incomplete_object_priority")
        self.assertIsNone(incomplete["pruning"]["pruned_by_rule_id"])
        for rule in (best, stricter):
            self.assertTrue(rule["object_cut_priority_complete"])
            self.assertEqual([step["cut"]["observable"] for step in rule["steps"][:2]], ["type_id", "pt"])
            self.assertEqual(rule["steps"][1]["cut"]["comparison"], ">")
        self.assertEqual(self.erl["summary"]["candidate_rule_count"], 3)
        for rule in (stricter, incomplete):
            self.assertIsNone(rule["pruned_rank"])
            self.assertFalse(rule["retained_after_similarity_pruning"])

    def test_edt_partition_and_default_selection(self):
        leaves = self.edt["results"]["leaves"]
        prediction = self.edt["summary"]["default_prediction"]
        self.assert_metrics(prediction, self.edt["inputs"])
        for role in ("signal", "background"):
            self.assertEqual(sum(leaf["metrics"][f"{role}_accepted"] for leaf in leaves), 100)
            self.assertEqual(sum(leaf["metrics"][f"{role}_accepted"] for leaf in leaves
                                 if leaf["predicted_class_at_default_threshold"]), prediction[f"{role}_accepted"])
        self.assertEqual([leaf["rank"] for leaf in leaves], [1, 2])
        for leaf in leaves:
            self.assertAlmostEqual(leaf["leaf_signal_fraction"], leaf["metrics"]["mixture_label_purity"])
            self.assertEqual(leaf["predicted_class_at_default_threshold"], int(leaf["leaf_signal_fraction"] >= self.edt["configuration"]["decision_threshold"]))
            step = leaf["path"][0]
            self.assertAlmostEqual(step["source_split_gain"], gini_gain(30, 4, 100, 100))
        self.assertEqual(leaves[0]["path"][0]["cut"], leaves[1]["path"][0]["cut"])
        self.assertEqual(leaves[1]["path"][0]["branch"], "negative")
        self.assertIn("<= 0", leaves[1]["path"][0]["selected_condition"])

    def test_documentation_links_resolve(self):
        for relative in ("docs/AI_GUIDE.md", "docs/REPORT_FORMAT.md", "examples/reports/README.md"):
            path = ROOT / relative
            for target in re.findall(r"\[[^\]]+\]\(([^)\s]+)\)", path.read_text(encoding="utf-8")):
                if "://" not in target:
                    self.assertTrue((path.parent / target.split("#", 1)[0]).exists(), (relative, target))

    @unittest.skipIf(jsonschema is None, "jsonschema is an optional documentation-check dependency")
    def test_schema_and_fixtures(self):
        schema = load_json(SCHEMA_PATH)
        jsonschema.Draft202012Validator.check_schema(schema)
        validator = jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())
        validator.validate(self.erl)
        validator.validate(self.edt)
        extended = deepcopy(self.erl)
        extended["future_optional_field"] = {"note": "Additive extensions are permitted."}
        extended["configuration"]["stricter_score_improvement"] = -0.02
        validator.validate(extended)
        unsupported = deepcopy(self.erl)
        unsupported["schema_version"] = 2
        self.assertFalse(validator.is_valid(unsupported))
        missing_metric = deepcopy(self.edt)
        del missing_metric["results"]["leaves"][0]["metrics"]["signal_total"]
        self.assertFalse(validator.is_valid(missing_metric))


if __name__ == "__main__":
    unittest.main()
