import numpy as np
import pytest

from set_based_dt import (
    Branch, CountAggregation, ExistentialDecisionTree, Node,
    ObservableSpec, ThresholdCut,
)


OBSERVABLES = (
    ObservableSpec("pt", "continuous"),
    ObservableSpec("btag", "continuous"),
)


def fitted_tree() -> ExistentialDecisionTree:
    nodes = [
        Node(0, 0, cut=ThresholdCut(0, ">", 50), aggregation=CountAggregation(),
             positive_child_id=1, negative_child_id=2),
        Node(1, 1, parent_id=0, branch_from_parent=Branch.POSITIVE,
             cut=ThresholdCut(1, ">", 0.8), aggregation=CountAggregation(),
             positive_child_id=3, negative_child_id=4),
        Node(2, 1, parent_id=0, branch_from_parent=Branch.NEGATIVE, score=0.0),
        Node(3, 2, parent_id=1, branch_from_parent=Branch.POSITIVE, score=0.9),
        Node(4, 2, parent_id=1, branch_from_parent=Branch.NEGATIVE, score=0.1),
    ]
    return ExistentialDecisionTree(OBSERVABLES).set_tree(nodes)


def test_same_slot_semantics_and_prediction() -> None:
    tree = fitted_tree()
    X = np.array([
        [[60, 0.9], [10, 0.0]],
        [[60, 0.1], [10, 0.9]],
        [[20, 0.9], [10, 0.9]],
    ])
    assert tree.apply(X).tolist() == [3, 4, 2]
    assert tree.predict(X).tolist() == [1, 0, 0]
    assert np.allclose(tree.decision_function(X), [0.9, 0.1, 0.0])


def test_retrieves_and_evaluates_node_chain() -> None:
    tree = fitted_tree()
    X = np.array([[[60, 0.9], [55, 0.2]], [[60, 0.1], [10, 0.9]]])
    assert [step.source_node_id for step in tree.cut_chain(3)] == [0, 1]
    assert len(tree.slot_cut_chain(3)) == 2
    assert tree.evaluate_chain(3, X).selected_samples.tolist() == [True, False]
    assert "pt > 50" in tree.cut_chain_text(3)


def test_slot_function_does_not_apply_event_aggregation() -> None:
    X = np.array([[[60, 0.9], [60, 0.2], [10, 0.9]]])
    assert fitted_tree().slot_cut_function(3)(X).tolist() == [[True, False, False]]


def test_evaluates_tree_and_specific_node_chain() -> None:
    tree = fitted_tree()
    signal = np.array([[[60, 0.9]], [[60, 0.1]]])
    background = np.array([[[20, 0.9]], [[70, 0.1]]])
    performance = tree.evaluate(signal, background)
    assert performance.true_positive_rate == 0.5
    assert performance.false_positive_rate == 0.0
    assert tree.evaluate_node(3, signal, background).true_positive_rate == 0.5


def test_negative_branch_is_not_a_positive_slot_cut() -> None:
    tree = fitted_tree()
    assert tree.slot_cut_chain(2) == ()
    X = np.array([[[20, 0.9]], [[60, 0.9]]])
    assert tree.evaluate_chain(2, X).selected_samples.tolist() == [True, False]


def test_fit_requires_an_external_gain_function() -> None:
    with pytest.raises(ValueError, match="gain_function"):
        ExistentialDecisionTree(OBSERVABLES).fit(
            np.zeros((2, 1, 2)), np.zeros((3, 1, 2))
        )


def gini_gain(split) -> float:
    def impurity(region):
        total = region.total_weight
        if total == 0:
            return 0.0
        p = region.signal_weight / total
        return 2.0 * p * (1.0 - p)

    parent_weight = split.parent.total_weight
    child_loss = (
        split.positive.total_weight * impurity(split.positive)
        + split.negative.total_weight * impurity(split.negative)
    ) / parent_weight
    return impurity(split.parent) - child_loss


def test_fit_learns_a_basic_existential_cut() -> None:
    signal = np.array([[[80.0, 0.9]], [[70.0, 0.7]], [[60.0, 0.5]]])
    background = np.array([[[10.0, 0.1]], [[20.0, 0.2]], [[30.0, 0.3]]])
    tree = ExistentialDecisionTree(
        OBSERVABLES, max_depth=1, gain_function=gini_gain
    ).fit(signal, background)

    assert len(tree.nodes_) == 3
    assert tree.nodes_[0].cut is not None
    assert tree.predict(signal).tolist() == [1, 1, 1]
    assert tree.predict(background).tolist() == [0, 0, 0]
    assert tree.evaluate(signal, background).true_positive_rate == 1.0
    assert tree.evaluate(signal, background).false_positive_rate == 0.0


def test_fit_propagates_the_same_slot_mask() -> None:
    signal = np.array([
        [[80.0, 0.9], [10.0, 0.1]],
        [[70.0, 0.8], [20.0, 0.2]],
    ])
    background = np.array([
        [[80.0, 0.1], [10.0, 0.9]],
        [[70.0, 0.2], [20.0, 0.8]],
        [[20.0, 0.1], [10.0, 0.9]],
        [[30.0, 0.2], [20.0, 0.8]],
    ])
    tree = ExistentialDecisionTree(
        OBSERVABLES, max_depth=2, gain_function=gini_gain
    ).fit(signal, background)

    assert tree.predict(signal).tolist() == [1, 1]
    assert tree.predict(background).tolist() == [0, 0, 0, 0]
