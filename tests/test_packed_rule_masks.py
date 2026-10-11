"""Packed search states are exact, compact, and release dense working masks."""

from dataclasses import fields
import unittest
import weakref
from unittest.mock import patch

import numpy as np

from set_based_models import EventCut, ExistentialRule, ObservableSpec, SetInput, TPRFPRRuleScore
from set_based_models.rule_list import generation
from set_based_models.rule_list.pruning import prune_rule_candidates


class DenseReferenceGenerator(generation.BeamRuleGenerator):
    """Pre-packing search/storage policy, retained only as a regression oracle."""

    def generate(self, signal, background):
        indices = {spec.name: index for index, spec in enumerate(signal.observables)}
        priority = tuple(indices[generation._priority_entry(entry)[0]] for entry in self.object_cut_priority)
        sw, bw = generation._weights(signal), generation._weights(background)
        root = generation._Candidate((), signal.slot_mask.copy(), background.slot_mask.copy(),
                                     np.ones(signal.n_samples, bool), np.ones(background.n_samples, bool))
        beam, pool = [root], []
        for depth in range(1, self.max_rule_depth + 1):
            expanded, by_rule, by_state = [], set(), set()
            for parent in beam:
                for cut in self._candidate_cuts(parent, signal, background):
                    key = generation._canonical_key((*[step.cut for step in parent.steps], cut))
                    if key in by_rule:
                        continue
                    by_rule.add(key)
                    if isinstance(cut, EventCut):
                        sm, bm = parent.signal_mask, parent.background_mask
                        ss = parent.signal_selection & cut.evaluate(signal)
                        bs = parent.background_selection & cut.evaluate(background)
                    else:
                        sm, bm = parent.signal_mask & cut.evaluate(signal), parent.background_mask & cut.evaluate(background)
                        ss = parent.signal_selection & (np.count_nonzero(sm, axis=1) >= self.minimum_count)
                        bs = parent.background_selection & (np.count_nonzero(bm, axis=1) >= self.minimum_count)
                    stats = generation._statistics(ss, bs, sw, bw, depth, self.minimum_count)
                    if stats.true_positive_rate < self.min_signal_efficiency:
                        continue
                    score = float(self.scoring_function(stats))
                    if not np.isfinite(score):
                        continue
                    progress = parent.object_priority_progress
                    if not isinstance(cut, EventCut) and progress < len(priority) and cut.observable_index == priority[progress]:
                        progress += 1
                    signature = (progress, *(generation._mask_signature(mask) for mask in (ss, bs, sm, bm)))
                    if signature in by_state:
                        continue
                    by_state.add(signature)
                    expanded.append(generation._Candidate(
                        parent.steps + (generation.RuleStep(depth, cut, stats, score),), sm, bm, ss, bs, progress))
            expanded.sort(key=lambda candidate: candidate.score, reverse=True)
            beam = self._dense_retain(expanded)
            pool.extend(beam)
            if not beam:
                break
        pool.sort(key=lambda candidate: candidate.score, reverse=True)
        if self.max_candidate_rules is not None:
            pool = pool[:self.max_candidate_rules]
        return [ExistentialRule(index, signal.observables, candidate.steps, self.minimum_count,
                                candidate.signal_selection, candidate.background_selection,
                                event_observables=signal.event_observables)
                for index, candidate in enumerate(pool)]

    def _dense_retain(self, candidates):
        if not self.beam_similarity_filter:
            return candidates[:self.beam_width]
        retained = []
        for candidate in candidates:
            redundant = any(
                candidate.object_priority_progress == previous.object_priority_progress
                and generation._jaccard(candidate.signal_selection, previous.signal_selection) >= self.beam_signal_similarity_threshold
                and generation._jaccard(candidate.background_selection, previous.background_selection) >= self.beam_background_similarity_threshold
                and generation._jaccard(candidate.signal_mask, previous.signal_mask) >= self.beam_signal_similarity_threshold
                and generation._jaccard(candidate.background_mask, previous.background_mask) >= self.beam_background_similarity_threshold
                for previous in retained)
            if not redundant:
                retained.append(candidate)
                if len(retained) == self.beam_width:
                    break
        return retained


def generator(cls=generation.BeamRuleGenerator, **options):
    settings = dict(max_rule_depth=3, beam_width=8, max_candidate_rules=20,
                    max_thresholds=5, minimum_count=1, min_signal_efficiency=.01,
                    scoring_function=TPRFPRRuleScore(), beam_similarity_filter=True,
                    beam_signal_similarity_threshold=.8, beam_background_similarity_threshold=.9)
    settings.update(options)
    return cls(**settings)


class PackedMaskTests(unittest.TestCase):
    def test_roundtrip_handles_empty_shapes_and_incomplete_bytes(self):
        rng = np.random.default_rng(72)
        for shape in ((0,), (3, 0), (1, 1), (3, 7), (11, 231)):
            mask = rng.integers(0, 2, size=shape).astype(bool)
            packed = generation._mask_signature(mask)
            restored = generation._unpack_mask(packed, shape)
            np.testing.assert_array_equal(restored, mask)
            self.assertEqual(restored.dtype, bool)
            self.assertEqual(len(packed), (mask.size + 7) // 8)

    def test_packed_jaccard_is_exact_for_padding_zero_masks_and_large_counts(self):
        rng = np.random.default_rng(81)
        for shape in ((0,), (3, 0), (1,), (7,), (9, 231), (301, 231)):
            for probability in (0, .1, .5, 1):
                first, second = (rng.random(shape) < probability for _ in range(2))
                self.assertEqual(generation._packed_jaccard(generation._mask_signature(first),
                                                           generation._mask_signature(second)),
                                 generation._jaccard(first, second))

    def test_candidate_reuses_signature_and_contains_no_dense_arrays(self):
        rng = np.random.default_rng(7)
        sm, bm = rng.random((11, 231)) < .2, rng.random((13, 231)) < .2
        ss, bs = sm.any(axis=1), bm.any(axis=1)
        signature = (2, *(generation._mask_signature(mask) for mask in (ss, bs, sm, bm)))
        candidate = generation._PackedCandidate((), signature)
        self.assertIs(candidate.state, signature)
        self.assertTrue(all(isinstance(value, bytes) for value in candidate.state[1:]))
        dense = candidate.unpack(sm.shape, bm.shape)
        for first, second in ((dense.signal_mask, sm), (dense.background_mask, bm),
                              (dense.signal_selection, ss), (dense.background_selection, bs)):
            np.testing.assert_array_equal(first, second)
        self.assertEqual(dense.object_priority_progress, 2)
        packed_bytes = sum(len(value) for value in candidate.state[1:])
        dense_bytes = sum(value.nbytes for value in (ss, bs, sm, bm))
        self.assertLess(packed_bytes, dense_bytes / 7.9)

    def test_completed_layer_pool_has_no_slot_mask_fields(self):
        self.assertEqual([field.name for field in fields(generation._RuleCandidate)],
                         ["steps", "signal_selection", "background_selection"])


class PackedSearchTests(unittest.TestCase):
    def assert_same_rules(self, first, second, signal, background):
        self.assertEqual(len(first), len(second))
        for old, new in zip(first, second):
            self.assertEqual(old.rule_id, new.rule_id)
            self.assertEqual(old.steps, new.steps)
            self.assertEqual(old.to_text(), new.to_text())
            np.testing.assert_array_equal(old.signal_selection, new.signal_selection)
            np.testing.assert_array_equal(old.background_selection, new.background_selection)
            for rule, data, selection in ((new, signal, new.signal_selection), (new, background, new.background_selection)):
                np.testing.assert_array_equal(rule.predict(data.X if data.n_slots else None,
                                                          slot_mask=data.slot_mask if data.n_slots else None,
                                                          event_features=data.event_features), selection)
        options = dict(similarity_filter=True, stricter_score_improvement=.001, max_reference_acceptance=.6)
        old, new = (prune_rule_candidates(rules, **options) for rules in (first, second))
        self.assertEqual([rule.rule_id for rule in old.retained], [rule.rule_id for rule in new.retained])
        self.assertEqual(old.decisions, new.decisions)
        self.assertEqual(old.normalized_cuts, new.normalized_cuts)

    def test_dense_reference_matches_object_event_and_mixed_weighted_search(self):
        rng = np.random.default_rng(43)
        specs = (ObservableSpec("pt", "continuous"), ObservableSpec("type_id", "categorical", {0: "jet", 1: "photon"}),
                 ObservableSpec("eta", "continuous", validity_column="eta_valid"), ObservableSpec("eta_valid", "boolean"))
        event_specs = (ObservableSpec("activity", "continuous"),)
        arrays = []
        for rows in (7, 9):
            values = rng.normal(size=(rows, 3, 4))
            values[..., 0] = rng.uniform(1, 60, size=(rows, 3))
            values[..., 1] = rng.integers(0, 2, size=(rows, 3))
            values[..., 3] = rng.integers(0, 2, size=(rows, 3))
            values[..., 2][values[..., 3] == 0] = np.nan
            slots = rng.random((rows, 3)) > .2
            events = rng.uniform(0, 100, size=(rows, 1))
            arrays.append((values, slots, events, rng.uniform(.1, 3, size=rows)))
        for mode in ("object", "event", "combined"):
            for similarity in (False, True):
                for minimum_count in (1, 2):
                    for priority in ((), ("type_id", "pt"), ("type_id", "pt", "pt"),
                                     ("type_id", "pt:>", "pt:<=")) if mode != "event" else ((),):
                        with self.subTest(mode=mode, similarity=similarity, count=minimum_count, priority=priority):
                            data = [SetInput.from_arrays(values if mode != "event" else None,
                                                         specs if mode != "event" else (),
                                                         slot_mask=slots if mode != "event" else None,
                                                         event_features=events if mode != "object" else None,
                                                         event_observables=event_specs if mode != "object" else (), sample_weight=weight)
                                    for values, slots, events, weight in arrays]
                            options = dict(beam_similarity_filter=similarity, minimum_count=minimum_count, object_cut_priority=priority)
                            old = generator(DenseReferenceGenerator, **options).generate(*data)
                            new = generator(**options).generate(*data)
                            self.assert_same_rules(old, new, *data)

    def test_pair_search_matches_dense_reference_with_actual_converter_schema(self):
        from casual_tests._quantitative_reporting import _observable_from_schema
        from casual_tests.run_ERL_Gini_penalized_event_only import ReferencePenalizedGini
        from tests.test_pair_erl_runner import fixture
        data = []
        for is_signal in (True, False):
            sample = fixture(is_signal)
            specs = tuple(_observable_from_schema(item["name"], item) for item in sample["pair_obj_feature_schema"])
            data.append(SetInput.from_arrays(sample["pair_obj"].numpy(), specs,
                                             slot_mask=sample["pair_slot_mask"].numpy()))
        for similarity in (False, True):
            options = dict(beam_similarity_filter=similarity, object_cut_priority=("type_id_1", "type_id_2"),
                           scoring_function=ReferencePenalizedGini(10))
            old = generator(DenseReferenceGenerator, **options).generate(*data)
            new = generator(**options).generate(*data)
            self.assert_same_rules(old, new, *data)

    def test_dense_working_masks_are_released_before_filtering_every_layer(self):
        specs = (ObservableSpec("pt", "continuous"),)
        signal = SetInput.from_arrays(np.arange(21.).reshape(7, 3, 1), specs)
        background = SetInput.from_arrays(np.arange(27.).reshape(9, 3, 1), specs)
        references, filter_checks = [], []
        original = generation._Candidate

        def working(*args, **kwargs):
            candidate = original(*args, **kwargs)
            references.extend((weakref.ref(candidate.signal_mask), weakref.ref(candidate.background_mask)))
            return candidate

        def progress(status):
            if status["stage"] == "filtering_beam":
                self.assertTrue(all(reference() is None for reference in references))
                filter_checks.append(status["depth"])

        # A Mock's call history would itself keep all dense argument arrays alive.
        with patch.object(generation, "_Candidate", new=working):
            rules = generator().generate(signal, background, progress_callback=progress)
        self.assertTrue(rules)
        self.assertTrue(filter_checks)
        self.assertTrue(all(reference() is None for reference in references))


if __name__ == "__main__":
    unittest.main()
