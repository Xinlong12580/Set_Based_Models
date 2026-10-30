"""Portable core model round trips and frozen inference, using only NumPy."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np

from set_based_models import (
    CategoryCut, EventCut, ExistentialRule, ExistentialRuleList, ObservableSpec,
    RuleStatistics, RuleStep, ThresholdCut,
)
from set_based_models.rule_list.serialization import MODEL_FORMAT, SCHEMA_VERSION


OBJECTS = (
    ObservableSpec("pt", "continuous"),
    ObservableSpec("type_id", "categorical", {0: "jet", 1: "photon", 2: "electron"}),
    ObservableSpec("isolation", "continuous", validity_column="isolation_valid"),
    ObservableSpec("isolation_valid", "boolean"),
)
EVENTS = (
    ObservableSpec("ht", "continuous"),
    ObservableSpec("met", "continuous", validity_column="met_valid"),
    ObservableSpec("met_valid", "boolean"),
    ObservableSpec("count", "ordinal"),
    ObservableSpec("flag", "boolean"),
)
SIGNAL = np.array([[[40, 1, .1, 1], [10, 0, np.nan, 0]],
                   [[50, 1, .2, 1], [20, 0, .8, 1]],
                   [[60, 2, np.nan, 0], [0, 0, 0, 0]],
                   [[30, 1, .3, 1], [15, 0, .9, 1]]], float)
BACKGROUND = np.array([[[10, 0, .9, 1], [0, 0, 0, 0]],
                       [[20, 1, .8, 1], [10, 0, np.nan, 0]],
                       [[15, 0, .7, 1], [0, 0, 0, 0]],
                       [[30, 2, .6, 1], [5, 0, .8, 1]],
                       [[10, 0, np.nan, 0], [5, 0, .7, 1]]], float)
SIGNAL_EVENTS = np.array([[100, 40, 1, 2, 1], [120, 50, 1, 3, 1],
                          [90, np.nan, 0, 2, 0], [110, 60, 1, 1, 1]], float)
BACKGROUND_EVENTS = np.array([[30, 10, 1, 0, 0], [50, 20, 1, 1, 0],
                              [20, np.nan, 0, 0, 0], [70, 30, 1, 2, 1],
                              [25, 5, 1, 0, 0]], float)


def score(stats):
    return stats.true_positive_rate - stats.false_positive_rate - .001 * stats.rule_depth


def fit_model(mode):
    objects = mode != "event"
    events = mode in ("event", "combined")
    model = ExistentialRuleList(
        OBJECTS if objects else (), event_observables=EVENTS if events else (),
        max_rule_depth=3, beam_width=12, max_candidate_rules=24, max_thresholds=5,
        beam_similarity_filter=False, min_signal_efficiency=.01,
        scoring_function=lambda stats: score(stats),  # A non-importable callable is intentional.
    )
    fit_args = {"signal_weight": [1, 2, 3, 4], "background_weight": [5, 4, 3, 2, 1]}
    if objects:
        fit_args.update(signal_slot_mask=SIGNAL[..., 0] > 0, background_slot_mask=BACKGROUND[..., 0] > 0)
    if events:
        fit_args.update(signal_event_features=SIGNAL_EVENTS, background_event_features=BACKGROUND_EVENTS)
    model.fit(SIGNAL if objects else None, BACKGROUND if objects else None, **fit_args)
    model.prune_candidates(similarity_filter=True, stricter_score_improvement=-.01,
                           max_reference_acceptance=.5)
    inputs = [(SIGNAL if objects else None, {}), (BACKGROUND if objects else None, {})]
    for (array, kwargs), values in zip(inputs, (SIGNAL_EVENTS, BACKGROUND_EVENTS)):
        if objects:
            kwargs["slot_mask"] = array[..., 0] > 0
        if events:
            kwargs["event_features"] = values
    return model, inputs


class ERLSerializationTests(unittest.TestCase):
    def assert_same_model(self, original, loaded, inputs):
        self.assertEqual(loaded.observables, original.observables)
        self.assertEqual(loaded.event_observables, original.event_observables)
        self.assertEqual(loaded.input_mode_, original.input_mode_)
        self.assertEqual(loaded.to_dict(), original.to_dict())
        self.assertEqual([rule.rule_id for rule in loaded.selected_rules_],
                         [rule.rule_id for rule in original.selected_rules_])
        for left, right in zip(original.candidate_rules_, loaded.candidate_rules_):
            self.assertEqual(left.steps, right.steps)
            self.assertEqual(left.to_text(), right.to_text())
            self.assertIsNone(right.signal_selection)
            self.assertIsNone(right.background_selection)
            for array, kwargs in inputs:
                np.testing.assert_array_equal(left.predict(array, **kwargs), right.predict(array, **kwargs))
                np.testing.assert_array_equal(left.slot_mask(array, **kwargs), right.slot_mask(array, **kwargs))
                for depth in range(left.depth + 1):
                    np.testing.assert_array_equal(left.predict(array, depth=depth, **kwargs),
                                                  right.predict(array, depth=depth, **kwargs))
        for array, kwargs in inputs:
            np.testing.assert_array_equal(original.predict(array, **kwargs), loaded.predict(array, **kwargs))
            ids = [rule.rule_id for rule in original.candidate_rules_[:3]]
            np.testing.assert_array_equal(original.predict(array, rule_ids=ids, **kwargs),
                                          loaded.predict(array, rule_ids=ids, **kwargs))
        self.assertEqual(loaded.pruning_result_.decisions, original.pruning_result_.decisions)
        self.assertEqual(loaded.pruning_result_.normalized_cuts, original.pruning_result_.normalized_cuts)
        self.assertEqual(loaded.pruning_configuration_, original.pruning_configuration_)

    def test_round_trip_all_input_modes_and_weighted_histories(self):
        with tempfile.TemporaryDirectory() as temporary:
            for mode in ("object", "event", "combined"):
                with self.subTest(mode=mode):
                    model, inputs = fit_model(mode)
                    self.assertTrue(model.candidate_rules_)
                    path = Path(temporary) / mode / "model.json"
                    self.assertEqual(model.save(path), path)
                    self.assert_same_model(model, ExistentialRuleList.load(path), inputs)
                    self.assertTrue(all(rule.signal_selection is not None for rule in model.candidate_rules_))

    def test_schema_and_cut_records_are_plain_json_not_python_code(self):
        model, inputs = fit_model("combined")
        payload = json.loads(json.dumps(model.to_dict(), allow_nan=False))
        self.assertEqual(payload["format"], MODEL_FORMAT)
        self.assertEqual(payload["schema_version"], SCHEMA_VERSION)
        self.assertEqual(payload["observables"][1]["categories"], [
            {"code": 0, "label": "jet"}, {"code": 1, "label": "photon"}, {"code": 2, "label": "electron"},
        ])
        self.assertEqual(payload["observables"][2]["validity_column"], "isolation_valid")
        text = json.dumps(payload)
        for excluded in ("signal_selection", "background_selection", "scoring_function", "numpy", "lambda"):
            self.assertNotIn(excluded, text)
        loaded = ExistentialRuleList.from_dict(payload)
        self.assert_same_model(model, loaded, inputs)
        payload["configuration"]["object_cut_priority"].append("pt")
        payload["pruning"]["decisions"][0]["decision"]["status"] = "edited"
        self.assertEqual(loaded.object_cut_priority, ())
        self.assertNotEqual(loaded.pruning_result_.decisions[0]["status"], "edited")

    def test_pair_slots_repeated_directed_priorities_and_completion(self):
        specs = (ObservableSpec("type_id_1", "categorical", {0: "jet", 1: "photon"}),
                 ObservableSpec("type_id_2", "categorical", {0: "jet", 1: "photon"}),
                 ObservableSpec("pair_mass", "continuous", validity_column="pair_mass_valid"),
                 ObservableSpec("pair_mass_valid", "boolean"))
        signal = np.array([[[1, 1, mass, 1], [0, 1, mass + 30, 1]] for mass in (80, 120, 125, 130, 170)], float)
        background = np.array([[[1, 1, mass, 1], [0, 0, 20, 1]] for mass in (10, 90, 110, 150, 200)], float)
        signal[0, 0, 2:] = [np.nan, 0]
        model = ExistentialRuleList(
            specs, max_rule_depth=4, beam_width=40, max_candidate_rules=None,
            max_thresholds=None, beam_similarity_filter=False, scoring_function=score,
            object_cut_priority=["type_id_1", "type_id_2", "pair_mass:>", "pair_mass:<="],
        ).fit(signal, background)
        model.prune_candidates(require_complete_priority=True, similarity_filter=False)
        self.assertTrue(model.selected_rules_)
        self.assertEqual({rule.depth for rule in model.selected_rules_}, {4})
        loaded = ExistentialRuleList.from_dict(model.to_dict())
        self.assert_same_model(model, loaded, [(signal, {}), (background, {})])
        loaded.recalculate_candidates(signal, background, scoring_function=score)
        self.assertEqual(loaded.object_cut_priority, model.object_cut_priority)
        for before, after in zip(model.candidate_rules_, loaded.candidate_rules_):
            self.assertEqual(after.steps, before.steps)
            np.testing.assert_array_equal(after.signal_selection, before.predict(signal))
            np.testing.assert_array_equal(after.background_selection, before.predict(background))
        loaded.prune_candidates(require_complete_priority=True, similarity_filter=True)
        self.assertTrue(all(rule.depth == 4 for rule in loaded.selected_rules_))

    def test_all_comparison_directions_multicategory_and_shared_feature_names(self):
        objects = (ObservableSpec("x", "continuous", validity_column="valid"),
                   ObservableSpec("kind", "categorical", {0: "zero", 1: "one", 2: "two"}),
                   ObservableSpec("valid", "boolean"))
        events = (ObservableSpec("x", "continuous"), ObservableSpec("flag", "boolean"))
        cuts = (ThresholdCut(0, ">=", 1), ThresholdCut(0, "<", 3),
                CategoryCut(1, {1, 2}), EventCut(ThresholdCut(0, ">", 0)),
                EventCut(ThresholdCut(0, "<=", 2)), EventCut(CategoryCut(1, {1})))
        stats = RuleStatistics(2, 4, 1, 5, 2., 4., 1., 5., 1, 2)
        steps = tuple(RuleStep(i, cut, stats, -.1) for i, cut in enumerate(cuts, 1))
        rule = ExistentialRule(0, objects, steps, minimum_count=2, event_observables=events)
        model = ExistentialRuleList(objects, event_observables=events, max_rule_depth=6, minimum_count=2)
        model.candidate_rules_ = [rule]
        model.input_mode_ = "combined"
        model.prune_candidates(similarity_filter=False)
        values = np.array([[[1, 1, 1], [2, 2, 1]], [[1, 0, 1], [4, 1, 1]],
                           [[np.nan, 1, 0], [2, 2, 1]], [[1, 1, 1], [2, 2, 1]]])
        event_values = [[2, 1], [1, 1], [1, 1], [0, 1]]
        loaded = ExistentialRuleList.from_dict(model.to_dict())
        np.testing.assert_array_equal(loaded.predict(values, event_features=event_values), [True, False, False, False])
        self.assertEqual(loaded.get_rule(0).cuts, cuts)

    def test_or_counts_each_rule_independently_before_combining(self):
        specs = (ObservableSpec("x", "continuous"), ObservableSpec("y", "continuous"))
        model = ExistentialRuleList(specs, minimum_count=2)
        stats = RuleStatistics(0, 0, 0, 0, 0., 0., 0., 0., 1, 2)
        model.candidate_rules_ = [ExistentialRule(i, specs, (RuleStep(1, ThresholdCut(i, ">", 0), stats, 0.),), 2)
                                  for i in range(2)]
        model.prune_candidates(similarity_filter=False)
        loaded = ExistentialRuleList.from_dict(model.to_dict())
        values = [[[1, -1], [-1, 1]], [[1, -1], [1, -1]]]
        np.testing.assert_array_equal(loaded.predict(values), [False, True])

    def test_selected_rule_order_and_empty_final_selection(self):
        model, inputs = fit_model("object")
        selected = [model.candidate_rules_[2], model.candidate_rules_[0]]
        model.selected_rules_ = selected
        model.pruning_result_.retained = tuple(selected)
        loaded = ExistentialRuleList.from_dict(model.to_dict())
        self.assertEqual([r.rule_id for r in loaded.selected_rules_], [2, 0])
        model.prune_candidates(require_complete_priority=False, max_reference_acceptance=0,
                               similarity_filter=False)
        # Explicitly exercise a legitimate final empty selection, not an unpruned model.
        model.selected_rules_ = []
        model.pruning_result_.retained = ()
        loaded = ExistentialRuleList.from_dict(model.to_dict())
        for values, kwargs in inputs:
            self.assertFalse(loaded.predict(values, **kwargs).any())

    def test_unpruned_predictions_require_explicit_rule_choice(self):
        model, inputs = fit_model("object")
        model.pruning_result_ = None
        model.pruning_configuration_ = None
        model.selected_rules_ = []
        loaded = ExistentialRuleList.from_dict(model.to_dict())
        array, kwargs = inputs[0]
        with self.assertRaisesRegex(RuntimeError, "prune_candidates"):
            loaded.predict(array, **kwargs)
        np.testing.assert_array_equal(loaded.predict(array, rule_ids=[0], **kwargs),
                                      model.get_rule(0).predict(array, **kwargs))
        self.assertFalse(loaded.predict(array, rule_ids=[], **kwargs).any())

    def test_empty_fit_and_empty_event_batches_round_trip(self):
        model = ExistentialRuleList(event_observables=[ObservableSpec("count", "ordinal")], scoring_function=score)
        model.fit(signal_event_features=[[1], [1]], background_event_features=[[1], [1]])
        self.assertEqual(model.candidate_rules_, [])
        model.prune_candidates(similarity_filter=False)
        loaded = ExistentialRuleList.from_dict(model.to_dict())
        np.testing.assert_array_equal(loaded.predict(event_features=[[1], [2]]), [False, False])
        self.assertEqual(loaded.predict(event_features=np.empty((0, 1))).shape, (0,))

    def test_fresh_data_and_weighted_evaluation_do_not_use_training_masks(self):
        original, _ = fit_model("event")
        loaded = ExistentialRuleList.from_dict(original.to_dict())
        signal = [[130, 70, 1, 4, 1], [15, 3, 1, 0, 0], [100, np.nan, 0, 2, 1]]
        background = [[20, 4, 1, 0, 0], [60, 10, 1, 1, 0]]
        options = dict(signal_event_features=signal, background_event_features=background,
                       signal_weight=[3, 2, 1], background_weight=[2, 1])
        self.assertEqual(loaded.evaluate(**options), original.evaluate(**options))
        self.assertEqual(loaded.evaluate_rule(0, **options), original.evaluate_rule(0, **options))
        self.assertEqual(loaded.evaluate(rule_ids=iter([0, 1]), **options),
                         original.evaluate(rule_ids=[0, 1], **options))

    def test_refit_requires_explicit_scorer_and_resets_selection(self):
        original, _ = fit_model("object")
        loaded = ExistentialRuleList.from_dict(original.to_dict())
        with self.assertRaisesRegex(NotImplementedError, "scoring function"):
            loaded.fit(SIGNAL, BACKGROUND)
        loaded.fit(SIGNAL, BACKGROUND, scoring_function=score)
        self.assertIsNone(loaded.pruning_result_)
        self.assertIsNone(loaded.pruning_configuration_)
        self.assertEqual(loaded.selected_rules_, [])
        self.assertTrue(all(r.signal_selection is not None for r in loaded.candidate_rules_))
        configured = ExistentialRuleList.from_dict(original.to_dict(), scoring_function=score)
        configured.fit(SIGNAL, BACKGROUND)
        self.assertTrue(configured.candidate_rules_)

    def test_mask_free_repruning_uses_matrices_and_legacy_models_need_masks(self):
        model, _ = fit_model("combined")
        loaded = ExistentialRuleList.from_dict(model.to_dict())
        loaded.prune_candidates(similarity_filter=False, stricter_score_improvement=.01)
        loaded.prune_candidates(similarity_filter=True)
        legacy = model.to_dict()
        legacy.pop("event_jaccard")
        loaded = ExistentialRuleList.from_dict(legacy)
        with self.assertRaisesRegex(RuntimeError, "training masks"):
            loaded.prune_candidates(similarity_filter=True)

    def test_save_overwrite_warning_protection_and_atomic_failure(self):
        model, _ = fit_model("event")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.json"
            model.save(path, overwrite=False)
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                model.save(path, overwrite=False)
            self.assertEqual(path.read_bytes(), before)
            with self.assertWarnsRegex(UserWarning, "Overwriting"):
                model.save(path)
            with warnings.catch_warnings(), patch("set_based_models.rule_list.serialization.os.replace", side_effect=OSError("disk failure")):
                warnings.simplefilter("ignore")
                with self.assertRaisesRegex(OSError, "disk failure"):
                    model.save(path)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(temporary).glob("*.tmp")), [])
            self.assertEqual(ExistentialRuleList.load(path).to_dict(), model.to_dict())

    def test_model_format_errors_and_missing_rule_references(self):
        model, _ = fit_model("combined")
        original = model.to_dict()
        mutations = [
            lambda d: d.update(format="set_based_models_quantitative_report"),
            lambda d: d.update(schema_version=999),
            lambda d: d["rules"][0].update(rule_id=999),
            lambda d: d["rules"][0]["steps"][0]["cut"].update(scope="unknown"),
            lambda d: d["rules"][0]["steps"][0]["cut"].update(type="unknown"),
            lambda d: d["rules"][0]["steps"][0]["cut"].update(observable="missing"),
            lambda d: d["rules"][0]["steps"][0].update(generation_score=float("nan")),
            lambda d: d["pruning"].update(selected_rule_ids=[999]),
            lambda d: d["pruning"].update(selected_rule_ids=[0, 0]),
            lambda d: d["pruning"]["normalized_cuts"].pop(),
            lambda d: d["observables"][2].update(validity_column="missing"),
        ]
        for mutate in mutations:
            payload = copy.deepcopy(original)
            mutate(payload)
            with self.subTest(mutation=mutate), self.assertRaises((ValueError, TypeError, KeyError)):
                ExistentialRuleList.from_dict(payload)

    def test_numpy_configuration_scalars_and_none_limits(self):
        model, _ = fit_model("object")
        model.beam_width = np.int64(12)
        model.max_candidate_rules = None
        model.max_thresholds = None
        payload = model.to_dict()
        self.assertIs(type(payload["configuration"]["beam_width"]), int)
        loaded = ExistentialRuleList.from_dict(payload)
        self.assertEqual(loaded.beam_width, 12)
        self.assertIsNone(loaded.max_thresholds)

    def test_load_and_infer_in_fresh_numpy_only_process(self):
        model, _ = fit_model("event")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.json"
            model.save(path)
            code = '''
import importlib.abc
import sys
class NoStudyDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'awkward', 'matplotlib', 'casual_tests', 'set_based_edt'}:
            raise ModuleNotFoundError('Not available: ' + fullname, name=fullname)
sys.meta_path.insert(0, NoStudyDependencies())
from set_based_models import ExistentialRuleList
model = ExistentialRuleList.load(sys.argv[1])
assert model.predict(event_features=[[130, 70, 1, 4, 1]]).shape == (1,)
assert model.get_rule(0).signal_selection is None
'''
            source = Path(__file__).resolve().parents[1] / "src"
            result = subprocess.run([sys.executable, "-B", "-c", code, str(path)], cwd=temporary,
                                    env={**os.environ, "PYTHONPATH": str(source), "PYTHONDONTWRITEBYTECODE": "1"},
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_recalculate_all_candidates_and_prefixes_on_weighted_validation_inputs(self):
        for mode in ("object", "event", "combined"):
            with self.subTest(mode=mode):
                original, _ = fit_model(mode)
                before = original.to_dict()
                model = ExistentialRuleList.from_dict(before)
                objects, events = mode != "event", mode != "object"
                signal = BACKGROUND[:3] if objects else None
                background = SIGNAL[:2, :1] if objects else None  # New slot count is allowed.
                kwargs = {"signal_weight": [2., 3., 4.], "background_weight": [5., 6.]}
                if objects:
                    kwargs.update(signal_slot_mask=signal[..., 0] > 0, background_slot_mask=background[..., 0] > 0)
                if events:
                    kwargs.update(signal_event_features=BACKGROUND_EVENTS[:3], background_event_features=SIGNAL_EVENTS[:2])
                with patch("set_based_models.rule_list.model.BeamRuleGenerator.generate") as generate:
                    self.assertIs(model.recalculate_candidates(signal, background, scoring_function=score, **kwargs), model)
                    generate.assert_not_called()
                self.assertEqual(original.to_dict(), before)
                self.assertEqual(model.statistics_source_, "recalculated")
                self.assertIs(model.scoring_function, score)
                self.assertEqual(model.selected_rules_, [])
                self.assertIsNone(model.pruning_result_)
                self.assertIsNone(model.pruning_configuration_)
                self.assertEqual(len(model.candidate_rules_), len(original.candidate_rules_))
                for rule, training in zip(model.candidate_rules_, original.candidate_rules_):
                    self.assertEqual((rule.rule_id, rule.cuts, rule.minimum_count),
                                     (training.rule_id, training.cuts, training.minimum_count))
                    for step in rule.steps:
                        ss = rule.predict(signal, depth=step.depth, slot_mask=kwargs.get("signal_slot_mask"),
                                          event_features=kwargs.get("signal_event_features"))
                        bs = rule.predict(background, depth=step.depth, slot_mask=kwargs.get("background_slot_mask"),
                                          event_features=kwargs.get("background_event_features"))
                        expected = RuleStatistics(int(ss.sum()), 3, int(bs.sum()), 2,
                                                  float(np.array([2., 3., 4.])[ss].sum()), 9.,
                                                  float(np.array([5., 6.])[bs].sum()), 11., step.depth, rule.minimum_count)
                        self.assertEqual(step.statistics, expected)
                        self.assertEqual(step.generation_score, score(expected))
                    np.testing.assert_array_equal(rule.signal_selection, ss)
                    np.testing.assert_array_equal(rule.background_selection, bs)
                    self.assertFalse(rule.signal_selection.flags.writeable)
                model.prune_candidates(similarity_filter=True, stricter_score_improvement=-.01,
                                       max_reference_acceptance=.5)
                self.assertTrue(all(r.statistics.false_positive_rate <= .5 for r in model.selected_rules_))
                restored = ExistentialRuleList.from_dict(model.to_dict())
                self.assertEqual(restored.to_dict(), model.to_dict())
                self.assertEqual(restored.statistics_source_, "recalculated")
                self.assertTrue(all(r.signal_selection is None for r in restored.candidate_rules_))

    def test_recalculate_similarity_uses_new_scores_not_original_candidate_order(self):
        specs = (ObservableSpec("x", "continuous"), ObservableSpec("y", "continuous"))
        stats = RuleStatistics(1, 2, 0, 2, 1., 2., 0., 2., 1, 1)
        first = RuleStep(1, EventCut(ThresholdCut(0, ">", 1)), stats, 100.)
        second = RuleStep(2, EventCut(ThresholdCut(1, ">", 1)), stats, 0.)
        model = ExistentialRuleList(event_observables=specs)
        model.candidate_rules_ = [ExistentialRule(0, (), (first,), event_observables=specs),
                                  ExistentialRule(1, (), (first, second), event_observables=specs)]
        model.recalculate_candidates(signal_event_features=[[2, 2], [3, 3]],
                                     background_event_features=[[0, 0], [1, 1]],
                                     scoring_function=lambda statistics: statistics.rule_depth)
        self.assertEqual([r.rule_id for r in model.candidate_rules_], [0, 1])
        retained = model.prune_candidates(similarity_filter=True)
        self.assertEqual([r.rule_id for r in retained], [1])
        self.assertEqual(model.pruning_result_.decisions[0]["pruned_by_rule_id"], 1)
        self.assertEqual(model.pruning_result_.decisions[0]["score_gap"], 1.)

    def test_recalculate_counts_matching_slots_not_independent_cut_matches(self):
        specs = (ObservableSpec("pt_1", "continuous"), ObservableSpec("pt_2", "continuous"))
        stats = RuleStatistics(0, 1, 0, 1, 0., 1., 0., 1., 1, 2)
        steps = (RuleStep(1, ThresholdCut(0, ">", 0), stats, 0.),
                 RuleStep(2, ThresholdCut(1, ">", 0), stats, 0.))
        model = ExistentialRuleList(specs, minimum_count=2, scoring_function=score)
        model.candidate_rules_ = [ExistentialRule(0, specs, steps, minimum_count=2)]
        signal = [[[1, 0], [0, 1], [1, 1]], [[1, 1], [1, 1], [0, 0]]]
        model.recalculate_candidates(signal, [[[1, 0], [0, 1]]])
        np.testing.assert_array_equal(model.get_rule(0).signal_selection, [False, True])
        np.testing.assert_array_equal(model.get_rule(0).background_selection, [False])
        self.assertEqual(model.get_rule(0).statistics.signal_accepted, 1)

    def test_recalculate_failures_leave_existing_model_and_selection_unchanged(self):
        model, _ = fit_model("combined")
        before, candidates, selected = model.to_dict(), model.candidate_rules_, model.selected_rules_
        options = {"signal_slot_mask": SIGNAL[..., 0] > 0, "background_slot_mask": BACKGROUND[..., 0] > 0,
                   "signal_event_features": SIGNAL_EVENTS, "background_event_features": BACKGROUND_EVENTS}
        calls = []

        def bad_score(statistics):
            calls.append(statistics)
            return 0. if len(calls) < 3 else float("nan")

        with self.assertRaisesRegex(ValueError, "finite scores"):
            model.recalculate_candidates(SIGNAL, BACKGROUND, scoring_function=bad_score, **options)
        self.assertEqual(len(calls), 3)
        self.assertEqual(model.to_dict(), before)
        self.assertIs(model.candidate_rules_, candidates)
        self.assertIs(model.selected_rules_, selected)
        with self.assertRaisesRegex(ValueError, "Object tensors"):
            model.recalculate_candidates(signal_event_features=SIGNAL_EVENTS,
                                         background_event_features=BACKGROUND_EVENTS, scoring_function=score)
        self.assertEqual(model.to_dict(), before)
        loaded = ExistentialRuleList.from_dict(before)
        with self.assertRaises(NotImplementedError):
            loaded.recalculate_candidates(SIGNAL, BACKGROUND, **options)
        self.assertEqual(loaded.to_dict(), before)

    def test_recalculate_empty_batches_and_empty_candidate_pool(self):
        model, _ = fit_model("event")
        model.recalculate_candidates(signal_event_features=np.empty((0, len(EVENTS))),
                                     background_event_features=np.empty((0, len(EVENTS))), scoring_function=score)
        self.assertTrue(all(r.statistics.signal_total == r.statistics.background_total == 0
                            and r.signal_selection.shape == (0,) for r in model.candidate_rules_))
        model.prune_candidates()
        model.candidate_rules_ = []
        model.recalculate_candidates(signal_event_features=SIGNAL_EVENTS, background_event_features=BACKGROUND_EVENTS)
        self.assertEqual(model.prune_candidates(), [])

    def test_fit_and_load_scorer_defaults_and_legacy_statistics_source(self):
        model = ExistentialRuleList(event_observables=EVENTS, max_rule_depth=1)
        model.fit(signal_event_features=SIGNAL_EVENTS, background_event_features=BACKGROUND_EVENTS, scoring_function=score)
        self.assertIs(model.scoring_function, score)
        self.assertEqual(model.statistics_source_, "training")
        payload = model.to_dict()
        payload.pop("statistics_source")
        loaded = ExistentialRuleList.from_dict(payload, scoring_function=score)
        self.assertEqual(loaded.statistics_source_, "training")
        loaded.recalculate_candidates(signal_event_features=SIGNAL_EVENTS, background_event_features=BACKGROUND_EVENTS)
        self.assertEqual(loaded.statistics_source_, "recalculated")
        loaded.fit(signal_event_features=SIGNAL_EVENTS, background_event_features=BACKGROUND_EVENTS)
        self.assertEqual(loaded.statistics_source_, "training")


if __name__ == "__main__":
    unittest.main()
