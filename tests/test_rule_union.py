import numpy as np
import pytest

from set_based_models import (
    ExistentialRule,
    ExistentialRuleList,
    ObservableSpec,
    RuleStatistics,
    RuleStep,
    RuleUnion,
    ThresholdCut,
)


OBSERVABLES = (
    ObservableSpec("x", "continuous"),
    ObservableSpec("y", "continuous"),
)


def make_rule(rule_id, observable_index, *, minimum_count=1):
    statistics = RuleStatistics(
        signal_accepted=0,
        signal_total=0,
        background_accepted=0,
        background_total=0,
        signal_weight_accepted=0.0,
        signal_weight_total=0.0,
        background_weight_accepted=0.0,
        background_weight_total=0.0,
        rule_depth=1,
        minimum_count=minimum_count,
    )
    step = RuleStep(
        depth=1,
        cut=ThresholdCut(observable_index, ">", 0.0),
        statistics=statistics,
        generation_score=0.0,
    )
    return ExistentialRule(rule_id, OBSERVABLES, (step,), minimum_count)


def test_build_union_and_execute_on_events_and_slots():
    x_rule = make_rule(0, 0)
    y_rule = make_rule(1, 1)
    model = ExistentialRuleList(OBSERVABLES)
    model.candidate_rules_ = [x_rule, y_rule]

    union = model.build_union([0, 1])
    events = np.array(
        [
            [[1.0, -1.0], [-1.0, -1.0]],
            [[-1.0, 1.0], [-1.0, -1.0]],
            [[-1.0, -1.0], [-1.0, -1.0]],
        ]
    )

    assert union.rule_ids == (0, 1)
    assert union.n_rules == 2
    np.testing.assert_array_equal(union(events), [True, True, False])
    assert union.matches_slot([1.0, -1.0])
    assert union.matches_slot([-1.0, 1.0])
    assert not union.matches_slot([-1.0, -1.0])
    np.testing.assert_array_equal(
        union.matches_slots(events[0]), [True, False]
    )


def test_rule_counting_is_done_before_union():
    union = RuleUnion(
        OBSERVABLES,
        (make_rule(0, 0, minimum_count=2), make_rule(1, 1, minimum_count=2)),
    )
    # Each slot matches a different rule. The combined slot mask has two true
    # entries, but neither individual count-based rule accepts the event.
    events = np.array([[[1.0, -1.0], [-1.0, 1.0]]])

    np.testing.assert_array_equal(union.slot_mask(events), [[True, True]])
    np.testing.assert_array_equal(union.predict(events), [False])


def test_empty_union_is_always_false():
    union = RuleUnion(OBSERVABLES, ())
    events = np.ones((2, 3, 2))

    np.testing.assert_array_equal(union.predict(events), [False, False])
    np.testing.assert_array_equal(union.slot_mask(events), np.zeros((2, 3), dtype=bool))
    assert not union.matches_slot([1.0, 1.0])
    assert union.to_text() == "FALSE"


def test_build_union_rejects_duplicate_indices():
    model = ExistentialRuleList(OBSERVABLES)
    model.candidate_rules_ = [make_rule(0, 0)]

    with pytest.raises(ValueError, match="duplicates"):
        model.build_union([0, 0])
