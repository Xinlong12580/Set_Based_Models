"""Persisted event overlaps and explicitly unrelated cross-model blocks."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from set_based_models import ExistentialRuleList, ObservableSpec
from set_based_models.rule_list.similarity import event_jaccard_matrix
from tests.test_erl_serialization import fit_model, score


def expected_jaccard(masks):
    result = np.ones((len(masks), len(masks)))
    for i, left in enumerate(masks):
        for j, right in enumerate(masks):
            union = np.count_nonzero(left | right)
            result[i, j] = np.count_nonzero(left & right) / union if union else 1.
    return result


class ERLSimilarityTests(unittest.TestCase):
    def test_packed_calculation_matches_event_sets_including_empty_and_partial_bytes(self):
        rng = np.random.default_rng(42)
        for count in (0, 1, 7, 9, 129, 513):
            with self.subTest(events=count):
                masks = rng.random((137, count)) < .4
                masks[:2] = False
                masks[2] = True
                masks[3] = masks[4]
                matrix = event_jaccard_matrix(iter(masks))
                np.testing.assert_array_equal(matrix, expected_jaccard(masks))
                np.testing.assert_array_equal(matrix, matrix.T)
                self.assertFalse(matrix.flags.writeable)
                self.assertEqual(matrix.dtype, np.float64)
        self.assertEqual(event_jaccard_matrix([]).shape, (0, 0))
        with self.assertRaisesRegex(RuntimeError, "recalculate"):
            event_jaccard_matrix([None])
        with self.assertRaisesRegex(ValueError, "event rows"):
            event_jaccard_matrix([np.zeros(9, bool), np.zeros(16, bool)])

    def assert_measured(self, model):
        self.assertEqual(model.jaccard_source_, "event_masks")
        self.assertEqual(model.jaccard_blocks_, (len(model.candidate_rules_),))
        for role, matrix in (("signal", model.signal_jaccard_), ("background", model.background_jaccard_)):
            masks = [getattr(rule, role + "_selection") for rule in model.candidate_rules_]
            np.testing.assert_array_equal(matrix, expected_jaccard(masks))

    def test_fit_recalculate_and_disk_round_trip_in_every_mode(self):
        for mode in ("event", "object", "combined"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                model, inputs = fit_model(mode)
                self.assert_measured(model)
                original = model.to_dict()
                record = original["event_jaccard"]
                self.assertEqual(record["rule_ids"], list(range(len(model.candidate_rules_))))
                self.assertEqual(record["statistics_source"], "training")
                path = Path(temporary) / "model.json"
                model.save(path)
                restored = ExistentialRuleList.load(path)
                self.assertEqual(restored.to_dict(), original)
                self.assertTrue(all(r.signal_selection is None and r.background_selection is None
                                    for r in restored.candidate_rules_))
                self.assertFalse(restored.signal_jaccard_.flags.writeable)
                with patch("set_based_models.rule_list.pruning._jaccard", side_effect=AssertionError("Use saved matrices")):
                    restored.prune_candidates(**model.pruning_configuration_)
                self.assertEqual(restored.pruning_result_.decisions, model.pruning_result_.decisions)
                self.assertEqual([r.rule_id for r in restored.selected_rules_], [r.rule_id for r in model.selected_rules_])
                signal, options = inputs[1]
                background, background_options = inputs[0]
                restored.recalculate_candidates(signal, background, signal_slot_mask=options.get("slot_mask"),
                    background_slot_mask=background_options.get("slot_mask"), signal_event_features=options.get("event_features"),
                    background_event_features=background_options.get("event_features"), scoring_function=score)
                self.assert_measured(restored)
                self.assertEqual(restored.to_dict()["event_jaccard"]["statistics_source"], "recalculated")
                self.assertEqual(model.to_dict(), original)

    def test_merge_preserves_blocks_and_assumes_zero_between_models_without_mutating_sources(self):
        first, _ = fit_model("event")
        second, _ = fit_model("event")
        snapshots = [model.to_dict() for model in (first, second)]
        first = ExistentialRuleList.from_dict(snapshots[0])
        second = ExistentialRuleList.from_dict(snapshots[1])
        combined = ExistentialRuleList.merge([first, second])
        n, m = len(first.candidate_rules_), len(second.candidate_rules_)
        self.assertEqual(combined.jaccard_source_, "block_diagonal")
        self.assertEqual(combined.jaccard_blocks_, (n, m))
        for role in ("signal", "background"):
            matrix = getattr(combined, role + "_jaccard_")
            np.testing.assert_array_equal(matrix[:n, :n], getattr(first, role + "_jaccard_"))
            np.testing.assert_array_equal(matrix[n:, n:], getattr(second, role + "_jaccard_"))
            np.testing.assert_array_equal(matrix[:n, n:], np.zeros((n, m)))
            np.testing.assert_array_equal(matrix[n:, :n], np.zeros((m, n)))
        self.assertEqual([model.to_dict() for model in (first, second)], snapshots)
        self.assertEqual(combined.signal_jaccard_[0, n], 0.)  # Identical cuts in different inputs are still assumed unrelated.
        restored = ExistentialRuleList.from_dict(combined.to_dict())
        self.assertEqual(restored.to_dict(), combined.to_dict())
        with self.assertRaisesRegex(RuntimeError, "recalculate"):
            restored.prune_candidates()

    def test_nested_blocks_and_validation_replace_assumptions_with_full_measured_overlaps(self):
        first, inputs = fit_model("event")
        second, _ = fit_model("event")
        third, _ = fit_model("event")
        nested = ExistentialRuleList.merge([ExistentialRuleList.merge([first, second]), third])
        direct = ExistentialRuleList.merge([first, second, third])
        np.testing.assert_array_equal(nested.signal_jaccard_, direct.signal_jaccard_)
        self.assertEqual(nested.jaccard_blocks_, direct.jaccard_blocks_)
        self.assertEqual(nested.signal_jaccard_[0, len(first.candidate_rules_)], 0.)
        nested.recalculate_candidates(signal_event_features=inputs[0][1]["event_features"],
                                      background_event_features=inputs[1][1]["event_features"], scoring_function=score)
        self.assert_measured(nested)
        self.assertEqual(nested.signal_jaccard_[0, len(first.candidate_rules_)], 1.)
        nested.prune_candidates(similarity_filter=True)
        restored = ExistentialRuleList.from_dict(nested.to_dict())
        restored.prune_candidates(similarity_filter=True)
        self.assertEqual(restored.pruning_result_.decisions, nested.pruning_result_.decisions)

    def test_legacy_models_and_empty_pools(self):
        fitted, inputs = fit_model("event")
        legacy = fitted.to_dict()
        legacy.pop("event_jaccard")
        restored = ExistentialRuleList.from_dict(legacy)
        self.assertIsNone(restored.signal_jaccard_)
        with self.assertRaisesRegex(RuntimeError, "recalculate"):
            restored.compute_jaccard_matrices()
        combined = ExistentialRuleList.merge([fitted, restored])
        self.assertIsNone(combined.signal_jaccard_)
        self.assertNotIn("event_jaccard", combined.to_dict())
        combined.recalculate_candidates(signal_event_features=inputs[0][1]["event_features"],
                                        background_event_features=inputs[1][1]["event_features"], scoring_function=score)
        self.assert_measured(combined)
        empty = ExistentialRuleList(event_observables=[ObservableSpec("x", "continuous")], scoring_function=score)
        empty.fit(signal_event_features=[[1]], background_event_features=[[1]])
        self.assertEqual(empty.signal_jaccard_.shape, (0, 0))
        restored = ExistentialRuleList.from_dict(empty.to_dict())
        self.assertEqual(restored.background_jaccard_.shape, (0, 0))
        merged = ExistentialRuleList.merge([empty, restored])
        self.assertEqual(merged.signal_jaccard_.shape, (0, 0))

    def test_failed_recalculation_keeps_cached_matrices_unchanged(self):
        model, inputs = fit_model("event")
        before = model.to_dict()
        matrices = model.signal_jaccard_, model.background_jaccard_
        with self.assertRaisesRegex(ValueError, "finite scores"):
            model.recalculate_candidates(signal_event_features=inputs[0][1]["event_features"],
                background_event_features=inputs[1][1]["event_features"], scoring_function=lambda stats: float("nan"))
        self.assertEqual(model.to_dict(), before)
        self.assertIs(model.signal_jaccard_, matrices[0])
        self.assertIs(model.background_jaccard_, matrices[1])

    def test_matrix_rule_order_and_dimensions_are_checked_when_loading(self):
        fitted, _ = fit_model("event")
        payload = fitted.to_dict()
        for mutate in (lambda r: r["rule_ids"].reverse(), lambda r: r["signal"].pop(),
                       lambda r: r["background"][0].__setitem__(0, 2.)):
            record = copy.deepcopy(payload)
            mutate(record["event_jaccard"])
            with self.assertRaises(ValueError):
                ExistentialRuleList.from_dict(record)


if __name__ == "__main__":
    unittest.main()
