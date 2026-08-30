import math

import pytest

from set_based_dt import (
    DirectTPRFPRGain,
    PartitionStatistics,
    SignificanceGain,
    SplitStatistics,
    WeightedGiniGain,
    entropy_gain,
    gini_gain,
    misclassification_gain,
)


def perfect_split() -> SplitStatistics:
    root = PartitionStatistics(10, 10, 10.0, 10.0)
    return SplitStatistics(
        parent=root,
        positive=PartitionStatistics(10, 0, 10.0, 0.0),
        negative=PartitionStatistics(0, 10, 0.0, 10.0),
        root=root,
    )


def test_standard_impurity_gains_reward_perfect_separation() -> None:
    split = perfect_split()
    assert gini_gain(split) == pytest.approx(0.5)
    assert entropy_gain(split) == pytest.approx(math.log(2.0))
    assert misclassification_gain(split) == pytest.approx(0.5)


def test_parameterized_gains_are_drop_in_callables() -> None:
    split = perfect_split()
    assert WeightedGiniGain(background_cost=20.0)(split) > 0
    assert SignificanceGain()(split) > 0
    assert DirectTPRFPRGain(lambda_fpr=10.0)(split) > 0


def test_direct_tpr_fpr_requires_root_normalization() -> None:
    split = perfect_split()
    missing_root = SplitStatistics(split.parent, split.positive, split.negative)
    with pytest.raises(ValueError, match="root statistics"):
        DirectTPRFPRGain(1.0)(missing_root)


def test_objective_parameter_validation() -> None:
    with pytest.raises(ValueError):
        WeightedGiniGain(background_cost=0)
    with pytest.raises(ValueError):
        SignificanceGain(-0.1)
    with pytest.raises(ValueError):
        DirectTPRFPRGain(-1.0)
