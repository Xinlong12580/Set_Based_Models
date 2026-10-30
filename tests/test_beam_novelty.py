"""Novelty selection preserves base scores, packed states and beta-zero behavior."""

import unittest

import numpy as np

from set_based_models import EventCut, ExistentialRuleList, ObservableSpec, RuleStatistics, RuleStep, ThresholdCut
from set_based_models.rule_list import generation


def candidate(score, signal, background=None, slots=None, background_slots=None, progress=0):
    signal = np.asarray(signal, bool)
    background = np.zeros(len(signal), bool) if background is None else np.asarray(background, bool)
    slots = np.ones((len(signal), 2), bool) if slots is None else slots
    background_slots = np.ones((len(background), 2), bool) if background_slots is None else background_slots
    stats = RuleStatistics(int(signal.sum()), len(signal), int(background.sum()), len(background),
                           float(signal.sum()), float(len(signal)), float(background.sum()), float(len(background)), 1, 1)
    step = RuleStep(1, EventCut(ThresholdCut(0, ">", score)), stats, score)
    state = (progress, *(generation._mask_signature(mask) for mask in (signal, background, slots, background_slots)))
    return generation._PackedCandidate((step,), state)


def generator(weight=0., similarity=False, width=2):
    return generation.BeamRuleGenerator(max_rule_depth=2, beam_width=width, max_candidate_rules=10,
                                        max_thresholds=4, minimum_count=1, min_signal_efficiency=0.,
                                        scoring_function=lambda stats: stats.true_positive_rate - stats.false_positive_rate,
                                        beam_novelty_weight=weight, beam_similarity_filter=similarity)


def eager_selection(candidates, settings):
    remaining, retained = list(enumerate(candidates)), []
    while remaining and len(retained) < settings.beam_width:
        eligible = []
        for index, item in remaining:
            similarities = [generation._packed_jaccard(item.state[1], previous.state[1]) for previous in retained]
            if settings.beam_similarity_filter and any(settings._similar(item, previous) for previous in retained):
                continue
            utility = item.score + settings.beam_novelty_weight * (1. - max(similarities, default=0.))
            eligible.append((utility, -index, index, item))
        if not eligible:
            break
        _, _, chosen, item = max(eligible, key=lambda row: row[:2])
        retained.append(item)
        remaining = [(index, other) for index, other in remaining if index != chosen]
    return retained


class BeamNoveltyTests(unittest.TestCase):
    def test_lower_score_distinct_candidate_is_retained_without_altering_scores(self):
        broad = candidate(1., [1, 1, 1, 1, 0, 0, 0, 0])
        near = candidate(.99, [1, 1, 1, 1, 1, 0, 0, 0])
        rare = candidate(.8, [0, 0, 0, 0, 0, 0, 1, 1])
        items = [broad, near, rare]
        self.assertEqual(generator()._retain_diverse(items), [broad, near])
        self.assertEqual(generator(.3)._retain_diverse(items), [broad, rare])
        self.assertEqual([item.score for item in items], [1., .99, .8])
        self.assertTrue(all(isinstance(value, bytes) for item in items for value in item.state[1:]))

    def test_lazy_heap_matches_full_greedy_selection_with_ties_and_signed_weights(self):
        rng = np.random.default_rng(456)
        for repeat in range(8):
            items = [candidate(round(float(rng.uniform(-.1, .1)), 2), rng.random(17) < rng.uniform(0, 1),
                               rng.random(13) < rng.uniform(0, 1), rng.random((17, 2)) < .5,
                               rng.random((13, 2)) < .5, progress=int(rng.integers(0, 3))) for _ in range(25)]
            items += [candidate(.03, np.zeros(17, bool), np.zeros(13, bool)),
                      candidate(.03, np.zeros(17, bool), np.zeros(13, bool))]
            items.sort(key=lambda item: item.score, reverse=True)
            for weight in (-.5, -.01, .001, .03, .5):
                for similarity in (False, True):
                    with self.subTest(repeat=repeat, weight=weight, similarity=similarity):
                        settings = generator(weight, similarity, width=8)
                        expected = eager_selection(items, settings)
                        observed = settings._retain_diverse(items)
                        self.assertEqual([id(item) for item in observed], [id(item) for item in expected])

    def test_hard_similarity_still_respects_priority_progress_and_slot_masks(self):
        first = candidate(1., [1, 0, 1])
        duplicate = candidate(.99, [1, 0, 1])
        priority = candidate(.98, [1, 0, 1], progress=1)
        slots = candidate(.97, [1, 0, 1], slots=np.zeros((3, 2), bool))
        chosen = generator(.1, similarity=True, width=4)._retain_diverse([first, duplicate, priority, slots])
        self.assertEqual(chosen, [first, priority, slots])

    def test_novelty_uses_only_signal_event_overlap_not_reference_or_slot_differences(self):
        first = candidate(1., [1, 1, 0, 0], [1, 1, 0, 0])
        different_reference = candidate(.99, [1, 1, 0, 0], [0, 0, 1, 1], slots=np.zeros((4, 2), bool))
        different_signal = candidate(.9, [0, 0, 1, 1], [1, 1, 0, 0])
        self.assertEqual(generator(.2)._retain_diverse([first, different_reference, different_signal]), [first, different_signal])

    def test_progress_is_bounded_and_empty_inputs_work(self):
        items = [candidate(1., [1, 0]), candidate(.9, [0, 1]), candidate(.8, [1, 1])]
        updates = []
        generator(.2, width=5)._retain_diverse(items, progress_callback=lambda *args: updates.append(args))
        self.assertTrue(updates)
        self.assertTrue(all(0 <= processed <= total == len(items) and retained <= 3 for processed, total, retained in updates))
        self.assertEqual([u[0] for u in updates], sorted(u[0] for u in updates))
        self.assertEqual(generator(.2)._retain_diverse([]), [])

    def test_novelty_survives_global_pool_cap_and_scores_remain_intrinsic(self):
        signal = np.zeros((100, 3))
        signal[:80, 0], signal[:75, 1], signal[90:, 2] = 1, 1, 1
        background = np.zeros((100, 3))
        specs = [ObservableSpec(name, "continuous") for name in ("broad", "near", "rare")]
        score = lambda stats: stats.true_positive_rate - stats.false_positive_rate
        options = dict(event_observables=specs, max_rule_depth=2, beam_width=2, max_candidate_rules=2,
                       beam_similarity_filter=False, scoring_function=score)
        default = ExistentialRuleList(**options).fit(signal_event_features=signal, background_event_features=background)
        zero = ExistentialRuleList(**options, beam_novelty_weight=0.).fit(signal_event_features=signal, background_event_features=background)
        novel = ExistentialRuleList(**options, beam_novelty_weight=1.).fit(signal_event_features=signal, background_event_features=background)
        self.assertEqual(default.to_dict(), zero.to_dict())
        self.assertFalse(any(step.cut.cut.observable_index == 2 for rule in default.candidate_rules_ for step in rule.steps))
        self.assertTrue(any(step.cut.cut.observable_index == 2 for rule in novel.candidate_rules_ for step in rule.steps))
        self.assertEqual(len(novel.candidate_rules_), 2)
        for rule in novel.candidate_rules_:
            for step in rule.steps:
                self.assertEqual(step.generation_score, score(step.statistics))
        loaded = ExistentialRuleList.from_dict(novel.to_dict(), scoring_function=score)
        self.assertEqual(loaded.beam_novelty_weight, 1.)
        loaded.recalculate_candidates(signal_event_features=signal, background_event_features=background)
        loaded.prune_candidates(similarity_filter=True)
        self.assertTrue(loaded.selected_rules_)
        legacy = zero.to_dict()
        legacy["configuration"].pop("beam_novelty_weight")
        self.assertEqual(ExistentialRuleList.from_dict(legacy).beam_novelty_weight, 0.)

    def test_nonfinite_weights_are_rejected_but_negative_weights_are_supported(self):
        for weight in (np.nan, np.inf, -np.inf):
            with self.subTest(weight=weight), self.assertRaisesRegex(ValueError, "finite"):
                generator(weight)
        self.assertEqual(generator(-.1).beam_novelty_weight, -.1)


if __name__ == "__main__":
    unittest.main()
