"""Mixed event/slot rules preserve both event gates and object identity."""

import numpy as np
import pytest

from set_based_models import (
    Branch, CategoryCut, CountAggregation, EventCut, ExistentialDecisionTree,
    ExistentialRule, ExistentialRuleList, Node, ObservableSpec, RuleStatistics,
    RuleStep, RuleUnion, SetInput, ThresholdCut, TPRFPRRuleScore, gini_gain,
)


SLOTS = (ObservableSpec("pt", "continuous"), ObservableSpec("btag", "continuous"))
EVENTS = (ObservableSpec("HT", "continuous"),)


def make_rule(cuts, *, minimum_count=1, rule_id=0, event_observables=EVENTS):
    steps = tuple(
        RuleStep(i, cut, RuleStatistics(0, 0, 0, 0, 0., 0., 0., 0., i, minimum_count), 0.)
        for i, cut in enumerate(cuts, 1)
    )
    return ExistentialRule(rule_id, SLOTS, steps, minimum_count,
                           event_observables=event_observables)


def test_input_validates_event_shape_schema_and_values():
    x = np.zeros((2, 3, 2))
    specs = (ObservableSpec("n_electrons", "ordinal"),
             ObservableSpec("region", "categorical", {0: "barrel", 1: "endcap"}),
             ObservableSpec("trigger", "boolean"))
    data = SetInput.from_arrays(x, SLOTS, event_observables=specs,
                                event_features=[[2, 0, 1], [0, 1, 0]])
    assert data.event_features.shape == (2, 3)
    assert data.event_features.dtype == np.float64
    for bad in (np.zeros((1, 3)), np.zeros((2, 2)), np.zeros((2, 1, 3))):
        with pytest.raises(ValueError, match="event_features must have shape"):
            SetInput.from_arrays(x, SLOTS, event_observables=specs, event_features=bad)
    for bad in ([[1.5, 0, 1], [0, 1, 0]], [[1, 2, 1], [0, 1, 0]],
                [[1, 0, 2], [0, 1, 0]], [[np.nan, 0, 1], [0, 1, 0]]):
        with pytest.raises(ValueError):
            SetInput.from_arrays(x, SLOTS, event_observables=specs, event_features=bad)
    with pytest.raises(ValueError, match="required"):
        SetInput.from_arrays(x, SLOTS, event_observables=specs)
    with pytest.raises(ValueError, match="shape"):
        SetInput.from_arrays(x, SLOTS, event_features=np.zeros((2, 1)))


@pytest.mark.parametrize("model_class", [ExistentialDecisionTree, ExistentialRuleList])
def test_models_validate_event_schema_at_construction(model_class):
    with pytest.raises(ValueError, match="unique"):
        model_class(SLOTS, event_observables=EVENTS * 2)
    with pytest.raises(TypeError, match="ObservableSpec"):
        model_class(SLOTS, event_observables=["HT"])


@pytest.mark.parametrize("spec,signal_events,background_events", [
    (ObservableSpec("HT", "continuous"), [[800.], [900.]], [[100.], [200.]]),
    (ObservableSpec("n_electrons", "ordinal"), [[2], [3]], [[0], [1]]),
    (ObservableSpec("trigger", "boolean"), [[1], [1]], [[0], [0]]),
    (ObservableSpec("region", "categorical", {0: "a", 1: "b"}), [[1], [1]], [[0], [0]]),
])
def test_both_models_learn_event_only_cuts_even_without_valid_slots(
    spec, signal_events, background_events
):
    x = np.full((2, 2, 2), np.nan)
    slots = np.zeros((2, 2), dtype=bool)
    fit_args = dict(signal_event_features=signal_events,
                    background_event_features=background_events,
                    signal_slot_mask=slots, background_slot_mask=slots,
                    signal_weight=[2., 3.], background_weight=[4., 5.])
    tree = ExistentialDecisionTree(SLOTS, event_observables=(spec,), max_depth=1,
                                   minimum_count=2, gain_function=gini_gain).fit(x, x, **fit_args)
    assert isinstance(tree.nodes_[0].cut, EventCut)
    assert tree.nodes_[0].aggregation is None
    np.testing.assert_array_equal(tree.predict(x, slot_mask=slots,
                                               event_features=signal_events), [1, 1])
    np.testing.assert_array_equal(tree.predict(x, slot_mask=slots,
                                               event_features=background_events), [0, 0])
    performance = tree.evaluate(x, x, **fit_args)
    assert performance.signal_weight_accepted == 5.
    assert performance.background_weight_accepted == 0.
    np.testing.assert_array_equal(tree.decision_function(x, slot_mask=slots,
                                                         event_features=signal_events), [1., 1.])
    leaf = int(tree.apply(x, slot_mask=slots, event_features=signal_events)[0])
    assert tree.evaluate_node(leaf, x, x, **fit_args) == performance
    assert "event:" in tree.cut_chain_text(leaf)

    model = ExistentialRuleList(SLOTS, event_observables=(spec,), max_rule_depth=1,
                                minimum_count=2, scoring_function=TPRFPRRuleScore()).fit(
                                    x, x, **fit_args)
    rule = model.get_rule(0)
    assert isinstance(rule.cuts[0], EventCut)
    np.testing.assert_array_equal(rule(x, slot_mask=slots, event_features=signal_events), [True, True])
    np.testing.assert_array_equal(rule(x, slot_mask=slots, event_features=background_events), [False, False])
    assert model.evaluate_rule(0, x, x, **fit_args) == performance
    assert "EXISTS" not in rule.to_text()
    assert "COUNT" not in rule.to_text()
    union = model.build_union([0])
    np.testing.assert_array_equal(union(x, slot_mask=slots, event_features=signal_events), [True, True])
    with pytest.raises(ValueError, match="required"):
        tree.predict(x, slot_mask=slots)
    with pytest.raises(ValueError, match="required"):
        rule.predict(x, slot_mask=slots)


@pytest.mark.parametrize("event_first", [False, True])
def test_training_chooses_both_scopes_and_preserves_stored_rule_selections(event_first):
    signal = np.array([[[80., .9]], [[90., .9]]])
    background = np.array([[[80., .9]]] * (3 if event_first else 1) + [[[10., .9]]])
    signal_events = np.array([[800.], [900.]])
    background_events = np.array([[100.]] * (3 if event_first else 1) + [[800.]])
    kwargs = dict(signal_event_features=signal_events, background_event_features=background_events)
    tree = ExistentialDecisionTree(SLOTS, event_observables=EVENTS, max_depth=2,
                                   gain_function=gini_gain).fit(signal, background, **kwargs)
    assert isinstance(tree.nodes_[0].cut, EventCut) == event_first
    assert any(isinstance(n.cut, EventCut) for n in tree.nodes_)
    assert any(isinstance(n.cut, ThresholdCut) for n in tree.nodes_)
    assert tree.evaluate(signal, background, **kwargs).true_positive_rate == 1.
    assert tree.evaluate(signal, background, **kwargs).false_positive_rate == 0.

    model = ExistentialRuleList(SLOTS, event_observables=EVENTS, max_rule_depth=2,
                                beam_width=20, scoring_function=TPRFPRRuleScore()).fit(
                                    signal, background, **kwargs)
    best = model.get_rule(0)
    assert best.statistics.true_positive_rate == 1.
    assert best.statistics.false_positive_rate == 0.
    assert any(isinstance(c, EventCut) for c in best.cuts)
    assert any(isinstance(c, ThresholdCut) for c in best.cuts)
    for rule in model.candidate_rules_:
        np.testing.assert_array_equal(rule.signal_selection,
                                      rule.predict(signal, event_features=signal_events))
        np.testing.assert_array_equal(rule.background_selection,
                                      rule.predict(background, event_features=background_events))


def test_event_tree_branches_preserve_surviving_slots_and_exact_paths():
    # Slot -> event -> slot on BOTH event branches. A different low-pt object
    # must not be allowed to satisfy btag after passing the event cut.
    tree = ExistentialDecisionTree(SLOTS, event_observables=EVENTS).set_tree([
        Node(0, 0, cut=ThresholdCut(0, ">", 50), aggregation=CountAggregation(),
             positive_child_id=1, negative_child_id=2),
        Node(1, 1, 0, Branch.POSITIVE, cut=EventCut(ThresholdCut(0, ">", 600)),
             positive_child_id=3, negative_child_id=4),
        Node(2, 1, 0, Branch.NEGATIVE, score=0.),
        Node(3, 2, 1, Branch.POSITIVE, cut=ThresholdCut(1, ">", .8),
             aggregation=CountAggregation(), positive_child_id=5, negative_child_id=6),
        Node(4, 2, 1, Branch.NEGATIVE, cut=ThresholdCut(1, ">", .8),
             aggregation=CountAggregation(), positive_child_id=7, negative_child_id=8),
        Node(5, 3, 3, Branch.POSITIVE, score=1.),
        Node(6, 3, 3, Branch.NEGATIVE, score=0.),
        Node(7, 3, 4, Branch.POSITIVE, score=.75),
        Node(8, 3, 4, Branch.NEGATIVE, score=0.),
    ])
    x = np.array([[[80., .9], [10., .1]], [[80., .1], [10., .9]],
                  [[80., .9], [10., .1]], [[80., .1], [10., .9]],
                  [[10., .9], [10., .9]]])
    events = np.array([[800.], [800.], [100.], [100.], [800.]])
    leaves = tree.apply(x, event_features=events)
    np.testing.assert_array_equal(leaves, [5, 6, 7, 8, 2])
    for leaf in (2, 5, 6, 7, 8):
        result = tree.evaluate_chain(leaf, x, event_features=events)
        np.testing.assert_array_equal(result.selected_samples, leaves == leaf)
    for node_id in (3, 4):
        result = tree.evaluate_chain(node_id, x, event_features=events)
        assert np.all(result.surviving_slots[result.selected_samples] == [True, False])
    assert len(tree.slot_cut_chain(7)) == 2
    assert "FAIL [event: HT > 600]" in tree.cut_chain_text(7)
    np.testing.assert_array_equal(tree.slot_cut_function(7)(x),
                                  [[True, False], [False, False], [True, False],
                                   [False, False], [False, False]])


@pytest.mark.parametrize("event_position", [0, 1, 2])
def test_rule_event_gates_and_same_slot_count_are_independent_of_cut_order(event_position):
    cuts = [ThresholdCut(0, ">", 50), ThresholdCut(1, ">", .8)]
    cuts.insert(event_position, EventCut(ThresholdCut(0, ">", 600)))
    rule = make_rule(cuts, minimum_count=2)
    x = np.array([[[80., .9], [70., .9], [0., 0.]],
                  [[80., .9], [70., .9], [0., 0.]],
                  [[80., .1], [10., .9], [0., 0.]],
                  [[80., .9], [0., 0.], [0., 0.]]])
    events = np.array([[800.], [100.], [800.], [800.]])
    slots = np.array([[1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=bool)
    np.testing.assert_array_equal(rule.predict(x, slot_mask=slots, event_features=events),
                                  [True, False, False, False])
    np.testing.assert_array_equal(rule.slot_mask(x, slot_mask=slots, event_features=events),
                                  [[True, True, False], [False, False, False],
                                   [False, False, False], [True, False, False]])
    assert rule.matches_slot([80., .9], event_features=[800.])
    assert not rule.matches_slot([80., .9], event_features=[100.])
    np.testing.assert_array_equal(rule.matches_slots(x, event_features=events),
                                  rule.slot_mask(x, slot_mask=slots, event_features=events))
    assert "event: HT > 600" in rule.to_text()
    assert "COUNT >= 2 slot: pt > 50 AND btag > 0.8" in rule.to_text()


def test_union_applies_event_gates_to_each_rule_before_or():
    high_ht = make_rule([EventCut(ThresholdCut(0, ">", 600)), ThresholdCut(0, ">", 50)],
                        minimum_count=2)
    low_ht = make_rule([EventCut(ThresholdCut(0, "<=", 600)), ThresholdCut(1, ">", .8)],
                       minimum_count=2, rule_id=1)
    union = RuleUnion(SLOTS, (high_ht, low_ht), event_observables=EVENTS)
    x = np.array([[[80., .1], [80., .1]], [[10., .9], [10., .9]],
                  [[80., .1], [10., .9]], [[80., .1], [80., .1]]])
    events = np.array([[800.], [100.], [800.], [100.]])
    np.testing.assert_array_equal(union.predict(x, event_features=events), [True, True, False, False])
    np.testing.assert_array_equal(union.slot_mask(x, event_features=events),
                                  [[True, True], [True, True], [True, False], [False, False]])
    assert union.matches_slot([80., .1], event_features=[800.])
    assert not union.matches_slot([80., .1], event_features=[100.])
    np.testing.assert_array_equal(union.matches_slots(x, event_features=events),
                                  union.slot_mask(x, event_features=events))
    with pytest.raises(ValueError, match="schema"):
        RuleUnion(SLOTS, (high_ht,))


def test_event_and_slot_predicates_with_same_index_remain_distinct():
    # Both scopes use index zero and the same category codes; one event cut
    # and one slot cut are required to separate the classes.
    specs = (ObservableSpec("flag", "boolean"),)
    s = np.ones((1, 1, 1))
    b = np.array([[[1.]], [[0.]]])
    model = ExistentialRuleList(specs, event_observables=specs, max_rule_depth=2,
                                beam_width=20, scoring_function=TPRFPRRuleScore()).fit(
                                    s, b, signal_event_features=[[1.]],
                                    background_event_features=[[0.], [1.]])
    best = model.get_rule(0)
    assert best.statistics.true_positive_rate == 1.
    assert best.statistics.false_positive_rate == 0.
    assert any(isinstance(c, EventCut) for c in best.cuts)
    assert any(isinstance(c, CategoryCut) for c in best.cuts)


def test_exact_dedup_keeps_event_cuts_with_identical_slot_masks():
    specs = (ObservableSpec("a", "boolean"), ObservableSpec("b", "boolean"))
    x = np.zeros((2, 1, 2))
    model = ExistentialRuleList(SLOTS, event_observables=specs, max_rule_depth=1,
                                beam_width=20, beam_similarity_filter=False,
                                scoring_function=TPRFPRRuleScore()).fit(
                                    x, x, signal_event_features=[[0, 0], [1, 1]],
                                    background_event_features=[[0, 1], [1, 0]])
    # All four event cuts leave identical slots; their event selections differ.
    assert len(model.candidate_rules_) == 4


def test_mixed_training_with_padding_counts_and_weights_matches_execution():
    rng = np.random.default_rng(17)
    data = rng.uniform(size=(30, 4, 2))
    mask = rng.random((30, 4)) > .25
    events = np.column_stack((rng.uniform(size=30), rng.integers(0, 2, size=30)))
    data[~mask] = np.nan
    signal, background = data[:15], data[15:]
    sw, bw = rng.uniform(.1, 2., size=(2, 15))
    schema = (ObservableSpec("HT", "continuous"), ObservableSpec("trigger", "boolean"))
    kwargs = dict(signal_event_features=events[:15], background_event_features=events[15:],
                  signal_slot_mask=mask[:15], background_slot_mask=mask[15:],
                  signal_weight=sw, background_weight=bw)
    model = ExistentialRuleList(SLOTS, event_observables=schema, max_rule_depth=3,
                                beam_width=12, max_thresholds=4, minimum_count=2,
                                scoring_function=TPRFPRRuleScore()).fit(signal, background, **kwargs)
    assert model.candidate_rules_
    for rule in model.candidate_rules_:
        for depth, step in enumerate(rule.steps, 1):
            ss = rule.predict(signal, slot_mask=mask[:15], event_features=events[:15], depth=depth)
            bs = rule.predict(background, slot_mask=mask[15:], event_features=events[15:], depth=depth)
            assert np.count_nonzero(ss) == step.statistics.signal_accepted
            assert np.count_nonzero(bs) == step.statistics.background_accepted
            assert sw[ss].sum() == pytest.approx(step.statistics.signal_weight_accepted)
            assert bw[bs].sum() == pytest.approx(step.statistics.background_weight_accepted)
        np.testing.assert_array_equal(ss, rule.signal_selection)
        np.testing.assert_array_equal(bs, rule.background_selection)

    tree = ExistentialDecisionTree(SLOTS, event_observables=schema, max_depth=3,
                                   max_thresholds=4, minimum_count=2,
                                   gain_function=gini_gain).fit(signal, background, **kwargs)
    reached_leaves = tree.apply(data, slot_mask=mask, event_features=events)
    for node in tree.nodes_:
        selected = tree.evaluate_chain(node.node_id, data, slot_mask=mask,
                                        event_features=events).selected_samples
        assert selected[:15].sum() == node.statistics.n_signal
        assert selected[15:].sum() == node.statistics.n_background
        assert sw[selected[:15]].sum() == pytest.approx(node.statistics.signal_weight)
        assert bw[selected[15:]].sum() == pytest.approx(node.statistics.background_weight)
        if node.is_leaf:
            np.testing.assert_array_equal(selected, reached_leaves == node.node_id)
