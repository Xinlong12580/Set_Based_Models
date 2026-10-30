"""Pooling independent searches, then selecting on common validation data."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from set_based_models import (
    CategoryCut, ExistentialRule, ExistentialRuleList, ObservableSpec,
    RuleStatistics, RuleStep, ThresholdCut,
)
from tests.test_erl_serialization import fit_model, score


class ERLMergingTests(unittest.TestCase):
    def test_pool_all_candidates_without_caps_or_source_mutation(self):
        for mode in ("event", "object", "combined"):
            with self.subTest(mode=mode):
                left, inputs = fit_model(mode)
                right, _ = fit_model(mode)
                right.beam_width = 2
                right.beam_novelty_weight = .03
                right.max_candidate_rules = 1
                snapshots = [model.to_dict() for model in (left, right)]
                combined = ExistentialRuleList.merge(iter([left, right]), source_names=["left", "right"])
                originals = left.candidate_rules_ + right.candidate_rules_
                self.assertEqual(len(combined.candidate_rules_), len(originals))
                self.assertIsNone(combined.max_candidate_rules)
                self.assertEqual([r.rule_id for r in combined.candidate_rules_], list(range(len(originals))))
                self.assertEqual(combined.statistics_source_, "merged")
                self.assertEqual(combined.object_cut_priority, ())
                self.assertEqual(combined.selected_rules_, [])
                self.assertIsNone(combined.pruning_result_)
                self.assertIsNone(combined.pruning_configuration_)
                for original, pooled in zip(originals, combined.candidate_rules_):
                    self.assertIsNot(original, pooled)
                    self.assertEqual(original.steps, pooled.steps)
                    self.assertEqual(original.minimum_count, pooled.minimum_count)
                    self.assertIsNone(pooled.signal_selection)
                    self.assertIsNone(pooled.background_selection)
                    for array, kwargs in inputs:
                        np.testing.assert_array_equal(pooled.predict(array, **kwargs), original.predict(array, **kwargs))
                for settings in ({"similarity_filter": False}, {"similarity_filter": True},
                                 {"similarity_filter": False, "stricter_score_improvement": 0}):
                    with self.assertRaisesRegex(RuntimeError, "recalculate_candidates"):
                        combined.prune_candidates(**settings)
                array, kwargs = inputs[0]
                with self.assertRaisesRegex(RuntimeError, "prune_candidates"):
                    combined.predict(array, **kwargs)
                np.testing.assert_array_equal(combined.predict(array, rule_ids=[0], **kwargs),
                                              originals[0].predict(array, **kwargs))
                self.assertEqual([m.to_dict() for m in (left, right)], snapshots)
                combined.merge_provenance_["sources"][0]["configuration"]["beam_width"] = 999
                self.assertEqual(left.to_dict(), snapshots[0])

    def test_recalculate_prune_and_saved_inference_in_all_modes(self):
        for mode in ("event", "object", "combined"):
            with self.subTest(mode=mode):
                left, inputs = fit_model(mode)
                right, _ = fit_model(mode)
                before = left.to_dict()
                combined = ExistentialRuleList.merge([left, right])
                values, options = inputs[0]
                reference, reference_options = inputs[1]
                n_signal = len(options["event_features"]) if values is None else len(values)
                n_background = len(reference_options["event_features"]) if reference is None else len(reference)
                signal_weight = np.arange(1, n_signal + 1)
                background_weight = np.arange(1, n_background + 1)
                with patch("set_based_models.rule_list.model.BeamRuleGenerator.generate") as generate:
                    combined.recalculate_candidates(
                        values, reference, signal_slot_mask=options.get("slot_mask"),
                        background_slot_mask=reference_options.get("slot_mask"),
                        signal_event_features=options.get("event_features"),
                        background_event_features=reference_options.get("event_features"),
                        signal_weight=signal_weight, background_weight=background_weight,
                        scoring_function=score,
                    )
                    generate.assert_not_called()
                self.assertEqual(combined.statistics_source_, "recalculated")
                for candidate in combined.candidate_rules_:
                    expected = candidate.predict(values, **options)
                    np.testing.assert_array_equal(candidate.signal_selection, expected)
                    self.assertEqual(candidate.statistics.signal_weight_accepted, signal_weight[expected].sum())
                    self.assertEqual(candidate.generation_score, score(candidate.statistics))
                retained = combined.prune_candidates(similarity_filter=True, stricter_score_improvement=-.01,
                                                       max_reference_acceptance=.5)
                self.assertTrue(retained)
                self.assertLessEqual(len(retained), len(left.candidate_rules_))
                self.assertTrue(any(d["status"] == "equivalent_rule"
                                    for d in combined.pruning_result_.decisions.values()))
                expected_or = np.logical_or.reduce([r.predict(values, **options) for r in retained])
                np.testing.assert_array_equal(combined.predict(values, **options), expected_or)
                restored = ExistentialRuleList.from_dict(combined.to_dict())
                self.assertEqual(restored.to_dict(), combined.to_dict())
                np.testing.assert_array_equal(restored.predict(values, **options), expected_or)
                self.assertEqual(left.to_dict(), before)

    def test_load_many_paths_and_pending_round_trip(self):
        first, _ = fit_model("event")
        second, _ = fit_model("event")
        second.beam_novelty_weight = .01
        with tempfile.TemporaryDirectory() as temporary:
            paths = [Path(temporary) / "first.model.json", Path(temporary) / "second.model.json"]
            for model, path in zip((first, second), paths):
                model.save(path)
            combined = ExistentialRuleList.load_many(iter(paths), scoring_function=score)
            self.assertIs(combined.scoring_function, score)
            provenance = combined.merge_provenance_
            self.assertEqual([s["name"] for s in provenance["sources"]], [str(p) for p in paths])
            self.assertEqual(provenance["sources"][1]["configuration"]["beam_novelty_weight"], .01)
            expected = [{"source_index": i, "rule_id": r.rule_id}
                        for i, m in enumerate((first, second)) for r in m.candidate_rules_]
            self.assertEqual(provenance["rules"], expected)
            path = Path(temporary) / "pooled.model.json"
            combined.save(path)
            restored = ExistentialRuleList.load(path)
            self.assertEqual(restored.to_dict(), combined.to_dict())
            with self.assertRaisesRegex(RuntimeError, "recalculate_candidates"):
                restored.prune_candidates(similarity_filter=False)

    def test_nested_merges_preserve_original_origins_and_detach_metadata(self):
        first, _ = fit_model("event")
        second, _ = fit_model("event")
        third, _ = fit_model("event")
        pooled = ExistentialRuleList.merge([first, second], source_names=["one", "two"])
        before = copy.deepcopy(pooled.merge_provenance_)
        nested = ExistentialRuleList.merge([pooled, third], source_names=["pooled", "three"])
        direct = ExistentialRuleList.merge([first, second, third], source_names=["one", "two", "three"])
        self.assertEqual(nested.to_dict(), direct.to_dict())
        nested.merge_provenance_["sources"][0]["name"] = "changed"
        self.assertEqual(pooled.merge_provenance_, before)

    def test_each_source_priority_is_used_before_equivalence_pruning(self):
        objects = (ObservableSpec("pt", "continuous"),
                   ObservableSpec("type_id", "categorical", {0: "jet", 1: "photon"}))
        first = ExistentialRuleList(objects, object_cut_priority=["pt:>", "pt:<="])
        second = ExistentialRuleList(objects, object_cut_priority=["type_id"])
        stats = RuleStatistics(1, 2, 0, 2, 1., 2., 0., 2., 1, 1)
        lower = RuleStep(1, ThresholdCut(0, ">", 10), stats, 1.)
        upper = RuleStep(2, ThresholdCut(0, "<=", 30), stats, 1.)
        kind = RuleStep(1, CategoryCut(1, {1}), stats, 1.)
        first.candidate_rules_ = [ExistentialRule(0, objects, (lower,)),
                                  ExistentialRule(1, objects, (lower, upper))]
        second.candidate_rules_ = [ExistentialRule(0, objects, (kind,)),
                                   ExistentialRule(1, objects, (lower,))]
        combined = ExistentialRuleList.merge([first, second], scoring_function=score)
        combined.recalculate_candidates([[[20, 1]], [[40, 0]]], [[[5, 0]], [[15, 0]]])
        restored = ExistentialRuleList.from_dict(combined.to_dict())
        retained = restored.prune_candidates(require_complete_priority=True, similarity_filter=False,
                                               stricter_score_improvement=-1)
        self.assertEqual({r.rule_id for r in retained}, {1, 2})
        self.assertEqual(restored.pruning_result_.decisions[0]["status"], "incomplete_object_priority")
        self.assertEqual(restored.pruning_result_.decisions[3]["status"], "incomplete_object_priority")
        # Without completion, identical lower-bound rules may compete across sources.
        restored.prune_candidates(require_complete_priority=False, similarity_filter=False,
                                   stricter_score_improvement=-1)
        self.assertEqual(restored.pruning_result_.decisions[3]["status"], "equivalent_rule")

    def test_counts_and_same_slot_semantics_survive_pooling(self):
        objects = (ObservableSpec("x", "continuous"), ObservableSpec("y", "continuous"))
        models = []
        for count in (1, 2):
            model = ExistentialRuleList(objects, minimum_count=count)
            stats = RuleStatistics(0, 0, 0, 0, 0., 0., 0., 0., 1, count)
            steps = (RuleStep(1, ThresholdCut(0, ">", 0), stats, 0.),
                     RuleStep(2, ThresholdCut(1, ">", 0), stats, 0.))
            model.candidate_rules_ = [ExistentialRule(0, objects, steps, minimum_count=count)]
            models.append(model)
        combined = ExistentialRuleList.merge(models, scoring_function=score)
        objects_signal = [[[1, 0], [0, 1], [0, 0]], [[1, 1], [0, 0], [0, 0]],
                          [[1, 1], [1, 1], [0, 0]]]
        combined.recalculate_candidates(objects_signal, [[[0, 0]]])
        np.testing.assert_array_equal(combined.get_rule(0).signal_selection, [False, True, True])
        np.testing.assert_array_equal(combined.get_rule(1).signal_selection, [False, False, True])
        self.assertEqual([r.minimum_count for r in combined.candidate_rules_], [1, 2])

    def test_incompatible_schemas_are_rejected(self):
        objects = (ObservableSpec("x", "continuous", validity_column="valid"),
                   ObservableSpec("valid", "boolean"),
                   ObservableSpec("kind", "categorical", {0: "jet", 1: "photon"}))
        events = (ObservableSpec("ht", "continuous"), ObservableSpec("count", "ordinal"))
        original = ExistentialRuleList(objects, event_observables=events)
        alternatives = [
            ExistentialRuleList(objects[::-1], event_observables=events),
            ExistentialRuleList((ObservableSpec("x", "continuous"), *objects[1:]), event_observables=events),
            ExistentialRuleList((*objects[:2], ObservableSpec("kind", "categorical", {0: "photon", 1: "jet"})),
                                event_observables=events),
            ExistentialRuleList(objects, event_observables=events[::-1]),
            ExistentialRuleList(objects, event_observables=(events[0], ObservableSpec("count", "continuous"))),
            ExistentialRuleList(objects),
            ExistentialRuleList(event_observables=events),
        ]
        for other in alternatives:
            with self.subTest(schema=other.observables), self.assertRaisesRegex(ValueError, "schemas"):
                ExistentialRuleList.merge([original, other])

    def test_empty_pools_refit_and_failed_recalculation(self):
        specs = (ObservableSpec("x", "continuous"),)
        empty = ExistentialRuleList(event_observables=specs)
        combined = ExistentialRuleList.merge([empty], scoring_function=score)
        combined.recalculate_candidates(signal_event_features=[[2]], background_event_features=[[0]])
        self.assertEqual(combined.prune_candidates(), [])
        np.testing.assert_array_equal(combined.predict(event_features=[[1], [2]]), [False, False])
        with self.assertRaisesRegex(ValueError, "at least one"):
            ExistentialRuleList.merge([])
        with self.assertRaisesRegex(ValueError, "at least one"):
            ExistentialRuleList.load_many([])
        with self.assertRaisesRegex(ValueError, "source_names"):
            ExistentialRuleList.merge([empty], source_names=[])
        model, inputs = fit_model("event")
        pooled = ExistentialRuleList.merge([model])
        before = pooled.to_dict()
        with self.assertRaises(NotImplementedError):
            pooled.recalculate_candidates(signal_event_features=inputs[0][1]["event_features"],
                                            background_event_features=inputs[1][1]["event_features"])
        self.assertEqual(pooled.to_dict(), before)
        pooled.fit(signal_event_features=inputs[0][1]["event_features"],
                   background_event_features=inputs[1][1]["event_features"], scoring_function=score)
        self.assertIsNone(pooled.merge_provenance_)
        self.assertEqual(pooled.statistics_source_, "training")


if __name__ == "__main__":
    unittest.main()
