import numpy as np
import pytest

from set_based_models import ExistentialRuleList, ObservableSpec


OBSERVABLES = (
    ObservableSpec("pt", "continuous"),
    ObservableSpec("btag", "continuous"),
)


def score(statistics):
    return statistics.true_positive_rate - statistics.false_positive_rate


def training_data():
    signal = np.array([
        [[80.0, 0.9], [10.0, 0.1]],
        [[70.0, 0.8], [20.0, 0.2]],
        [[60.0, 0.7], [15.0, 0.1]],
    ])
    background = np.array([
        [[80.0, 0.1], [10.0, 0.9]],
        [[70.0, 0.2], [20.0, 0.8]],
        [[30.0, 0.3], [15.0, 0.2]],
    ])
    return signal, background


def test_beam_generation_returns_executable_variable_depth_rules():
    signal, background = training_data()
    model = ExistentialRuleList(
        OBSERVABLES, max_rule_depth=2, beam_width=5,
        max_candidate_rules=10, scoring_function=score,
    ).fit(signal, background)
    assert model.candidate_rules_
    assert len(model.candidate_rules_) <= 10
    assert all(1 <= rule.depth <= 2 for rule in model.candidate_rules_)
    assert all(len(rule.steps) == rule.depth for rule in model.candidate_rules_)


def test_rule_supports_slot_and_event_execution_with_same_slot_semantics():
    signal, background = training_data()
    model = ExistentialRuleList(
        OBSERVABLES, max_rule_depth=2, beam_width=10, scoring_function=score,
    ).fit(signal, background)
    rule = max(model.candidate_rules_, key=lambda item: item.generation_score)
    assert isinstance(rule.matches_slot(signal[0, 0]), bool)
    assert rule.matches_slots(signal).shape == signal.shape[:2]
    assert rule.predict(signal).shape == (signal.shape[0],)
    assert rule.slot_mask(signal).shape == signal.shape[:2]
    assert len(rule.cut_chain()) == rule.depth
    assert rule.steps[-1].statistics is rule.statistics


def test_rule_evaluation_matches_stored_training_statistics():
    signal, background = training_data()
    model = ExistentialRuleList(
        OBSERVABLES, max_rule_depth=2, beam_width=5, scoring_function=score,
    ).fit(signal, background)
    rule = model.get_rule(0)
    performance = model.evaluate_rule(0, signal, background)
    assert performance.true_positive_rate == rule.statistics.true_positive_rate
    assert performance.false_positive_rate == rule.statistics.false_positive_rate
    np.testing.assert_array_equal(rule.signal_selection, rule.predict(signal))
    np.testing.assert_array_equal(rule.background_selection, rule.predict(background))
    assert not rule.signal_selection.flags.writeable
    assert not rule.background_selection.flags.writeable


def test_default_score_is_an_explicit_placeholder():
    signal, background = training_data()
    with pytest.raises(NotImplementedError, match="scoring function"):
        ExistentialRuleList(OBSERVABLES, max_rule_depth=1).fit(signal, background)


def jaccard(left, right):
    union = np.count_nonzero(left | right)
    return np.count_nonzero(left & right) / union if union else 1.0


def test_beam_retention_discards_rules_similar_in_both_classes():
    signal, background = training_data()
    threshold = 0.5
    model = ExistentialRuleList(
        OBSERVABLES,
        max_rule_depth=1,
        beam_width=20,
        beam_signal_similarity_threshold=threshold,
        beam_background_similarity_threshold=threshold,
        scoring_function=score,
    ).fit(signal, background)

    rules = model.candidate_rules_
    for index, rule in enumerate(rules):
        for previous in rules[:index]:
            signal_similarity = jaccard(rule.predict(signal), previous.predict(signal))
            background_similarity = jaccard(
                rule.predict(background), previous.predict(background)
            )
            assert not (
                signal_similarity >= threshold
                and background_similarity >= threshold
            )


def test_beam_similarity_thresholds_are_validated():
    with pytest.raises(ValueError, match="beam_signal_similarity_threshold"):
        ExistentialRuleList(
            OBSERVABLES,
            beam_signal_similarity_threshold=1.1,
            scoring_function=score,
        )


def test_beam_similarity_filter_can_be_disabled():
    signal, background = training_data()
    common = dict(
        max_rule_depth=1,
        beam_width=20,
        beam_signal_similarity_threshold=0.0,
        beam_background_similarity_threshold=0.0,
        scoring_function=score,
    )
    filtered = ExistentialRuleList(
        OBSERVABLES, beam_similarity_filter=True, **common
    ).fit(signal, background)
    unfiltered = ExistentialRuleList(
        OBSERVABLES, beam_similarity_filter=False, **common
    ).fit(signal, background)

    assert len(filtered.candidate_rules_) == 1
    assert len(unfiltered.candidate_rules_) > len(filtered.candidate_rules_)


def test_post_training_candidate_pruning_returns_a_new_ranked_list():
    signal, background = training_data()
    model = ExistentialRuleList(
        OBSERVABLES,
        max_rule_depth=2,
        beam_width=20,
        beam_similarity_filter=False,
        scoring_function=score,
    ).fit(signal, background)
    original = list(model.candidate_rules_)

    pruned = model.prune_candidates(
        signal_similarity_threshold=0.0,
        background_similarity_threshold=0.0,
    )

    assert len(pruned) == 1
    assert pruned[0] is original[0]
    assert model.candidate_rules_ == original
