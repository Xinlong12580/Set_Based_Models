"""Drop-in greedy split gains; every callable uses larger-is-better."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .training import PartitionStatistics, SplitStatistics


def gini_gain(split: SplitStatistics) -> float:
    return _impurity_gain(split, _gini)


def entropy_gain(split: SplitStatistics) -> float:
    return _impurity_gain(split, _entropy)


def misclassification_gain(split: SplitStatistics) -> float:
    return _impurity_gain(split, _misclassification)


@dataclass(frozen=True, slots=True)
class WeightedGiniGain:
    signal_cost: float = 1.0
    background_cost: float = 1.0

    def __post_init__(self) -> None:
        _validate_positive(self.signal_cost, "signal_cost")
        _validate_positive(self.background_cost, "background_cost")

    def __call__(self, split: SplitStatistics) -> float:
        def weighted(region):
            return (self.signal_cost * region.signal_weight,
                    self.background_cost * region.background_weight)
        ps, pb = weighted(split.parent)
        ys, yb = weighted(split.positive)
        ns, nb = weighted(split.negative)
        total = ps + pb
        if total <= 0:
            return 0.0
        return float(_gini(ps, pb) - (ys + yb) / total * _gini(ys, yb)
                     - (ns + nb) / total * _gini(ns, nb))


@dataclass(frozen=True, slots=True)
class SignificanceGain:
    """Gain in additive S²/(S+B+(fB)²), where f is fractional B uncertainty."""

    background_uncertainty_fraction: float = 0.0

    def __post_init__(self) -> None:
        value = self.background_uncertainty_fraction
        if not np.isfinite(value) or value < 0:
            raise ValueError("background_uncertainty_fraction must be finite and non-negative")

    def __call__(self, split: SplitStatistics) -> float:
        def score(region: PartitionStatistics) -> float:
            signal, background = region.signal_weight, region.background_weight
            denominator = signal + background + (
                self.background_uncertainty_fraction * background
            ) ** 2
            return signal * signal / denominator if denominator > 0 else 0.0
        return float(score(split.positive) + score(split.negative) - score(split.parent))


@dataclass(frozen=True, slots=True)
class DirectTPRFPRGain:
    """Gain in max(0, TPR - lambda_fpr*FPR), normalized to root yields."""

    lambda_fpr: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.lambda_fpr) or self.lambda_fpr < 0:
            raise ValueError("lambda_fpr must be finite and non-negative")

    def __call__(self, split: SplitStatistics) -> float:
        if split.root is None:
            raise ValueError("DirectTPRFPRGain requires root statistics")
        total_signal = split.root.signal_weight
        total_background = split.root.background_weight
        if total_signal <= 0 or total_background <= 0:
            return 0.0
        def utility(region):
            value = (region.signal_weight / total_signal
                     - self.lambda_fpr * region.background_weight / total_background)
            return max(0.0, value)
        return float(utility(split.positive) + utility(split.negative)
                     - utility(split.parent))


def make_weighted_gini_gain(*, signal_cost=1.0, background_cost=1.0):
    return WeightedGiniGain(signal_cost, background_cost)


def make_significance_gain(*, background_uncertainty_fraction=0.0):
    return SignificanceGain(background_uncertainty_fraction)


def make_direct_tpr_fpr_gain(*, lambda_fpr):
    return DirectTPRFPRGain(lambda_fpr)


def _impurity_gain(split, impurity) -> float:
    total = split.parent.total_weight
    if total <= 0:
        return 0.0
    return float(impurity(split.parent.signal_weight, split.parent.background_weight)
                 - split.positive.total_weight / total
                 * impurity(split.positive.signal_weight, split.positive.background_weight)
                 - split.negative.total_weight / total
                 * impurity(split.negative.signal_weight, split.negative.background_weight))


def _gini(signal, background) -> float:
    total = signal + background
    if total <= 0:
        return 0.0
    probability = signal / total
    return 2.0 * probability * (1.0 - probability)


def _entropy(signal, background) -> float:
    total = signal + background
    if total <= 0:
        return 0.0
    probabilities = np.asarray((signal / total, background / total))
    probabilities = probabilities[probabilities > 0]
    return float(-np.sum(probabilities * np.log(probabilities)))


def _misclassification(signal, background) -> float:
    total = signal + background
    return min(signal, background) / total if total > 0 else 0.0


def _validate_positive(value, name) -> None:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
