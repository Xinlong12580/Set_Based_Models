from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np

from collide2v_trigger import RuleParseError, TriggerMenu, parse_erl_model
from test_erl_parser import add_photon_count, data_with_schema, rule, threshold


def model(data, cuts, selected):
    source = rule(data, cuts[0])
    records = [{**rule(data, conditions)["rule"], "rule_id": index + 7} for index, conditions in enumerate(cuts)]
    return {"format": "set_based_models_erl", "schema_version": 1,
            "observables": source["observables"], "event_observables": source["event_observables"],
            "rules": records, "pruning": None if selected is None else {"selected_rule_ids": selected}}


class ModelParserTests(unittest.TestCase):
    def setUp(self):
        self.data = data_with_schema([{"met": 150}, {"met": 250}, {"met": 350}])
        self.model = model(self.data, [[threshold("met", value)] for value in (100, 200, 300)], [9, 7])

    def test_retained_rules_preserve_order_and_join_the_menu(self):
        source = deepcopy(self.model)
        triggers = parse_erl_model(self.model, self.data, name_prefix="Learned", target_rate_khz=20)
        self.assertEqual([trigger.name for trigger in triggers], ["Learned_9", "Learned_7"])
        self.assertEqual([trigger.target_rate_khz for trigger in triggers], [20, 20])
        bits = TriggerMenu(triggers=triggers).trigger_bits(self.data, [trigger.name for trigger in triggers])
        np.testing.assert_array_equal(bits["Learned_9"], [False, False, True])
        np.testing.assert_array_equal(bits["Learned_7"], [True, True, True])
        self.assertEqual(self.model, source)

    def test_explicit_rules_all_candidates_and_saved_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            path.write_text(json.dumps(self.model))
            triggers = parse_erl_model(path, self.data, rule_ids=[8, 9])
            self.assertEqual([trigger.definition["source_rule_id"] for trigger in triggers], [8, 9])
            all_triggers = parse_erl_model(path, self.data, all_candidates=True)
            self.assertEqual([trigger.definition["source_rule_id"] for trigger in all_triggers], [7, 8, 9])

    def test_unpruned_and_empty_final_selections(self):
        self.model["pruning"] = None
        with self.assertRaisesRegex(RuleParseError, "no final selection"):
            parse_erl_model(self.model, self.data)
        self.assertEqual(len(parse_erl_model(self.model, self.data, rule_ids=[8])), 1)
        self.model["pruning"] = {"selected_rule_ids": []}
        self.assertEqual(parse_erl_model(self.model, self.data), [])

    def test_any_unsupported_rule_fails_with_its_id(self):
        self.model["pruning"]["selected_rule_ids"] = [7, 8]
        self.model["rules"][1]["steps"][0]["cut"]["comparison"] = "<="
        with self.assertRaisesRegex(RuleParseError, "Rule 8:.*no supported tunable"):
            parse_erl_model(self.model, self.data)

    def test_skip_unsupported_warns_and_preserves_remaining_order(self):
        self.model["pruning"]["selected_rule_ids"] = [9, 7, 8]
        self.model["rules"][0]["steps"][0]["cut"]["comparison"] = "<="
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            triggers = parse_erl_model(self.model, self.data, skip_unsupported=True)
        self.assertEqual([trigger.definition["source_rule_id"] for trigger in triggers], [9, 8])
        self.assertEqual(len(caught), 1)
        self.assertTrue(issubclass(caught[0].category, UserWarning))
        self.assertRegex(str(caught[0].message), "Skipping rule 7:.*no supported tunable")

    def test_all_unsupported_returns_empty_list_with_a_warning_per_rule(self):
        for record in self.model["rules"]:
            record["steps"][0]["cut"]["comparison"] = "<="
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            triggers = parse_erl_model(self.model, self.data, skip_unsupported=True)
        self.assertEqual(triggers, [])
        self.assertEqual(len(caught), 2)
        self.assertIn("Skipping rule 9:", str(caught[0].message))
        self.assertIn("Skipping rule 7:", str(caught[1].message))

    def test_skip_does_not_hide_model_or_selection_errors(self):
        with warnings.catch_warnings(record=True) as caught:
            with self.assertRaisesRegex(RuleParseError, "missing rule IDs"):
                parse_erl_model(self.model, self.data, rule_ids=[100], skip_unsupported=True)
            self.model["schema_version"] = 99
            with self.assertRaisesRegex(RuleParseError, "model format/version"):
                parse_erl_model(self.model, self.data, skip_unsupported=True)
        self.assertEqual(caught, [])

    def test_per_rule_tuning_and_target_rates(self):
        data = data_with_schema([{"met": 300}])
        count = add_photon_count(data, [3], [35])
        source = model(data, [[threshold(count, 3, ">="), threshold("met", 200)], [threshold("met", 100)]], [7, 8])
        triggers = parse_erl_model(source, data, tune={7: [1]}, target_rate_khz={7: 5, 8: 20})
        self.assertEqual([trigger.target_rate_khz for trigger in triggers], [5, 20])
        self.assertEqual(triggers[0].cut_actions[0]["action"], "fixed")
        self.assertTrue(all(trigger(data)[0] for trigger in triggers))

    def test_ambiguous_or_missing_rule_selections_fail(self):
        for kwargs in ({"rule_ids": [7, 7]}, {"rule_ids": [100]}, {"rule_ids": [7], "all_candidates": True}):
            with self.assertRaises(RuleParseError):
                parse_erl_model(self.model, self.data, **kwargs)
        self.model["rules"][1]["rule_id"] = 7
        with self.assertRaisesRegex(RuleParseError, "duplicate candidate rule IDs"):
            parse_erl_model(self.model, self.data)


if __name__ == "__main__":
    unittest.main()
