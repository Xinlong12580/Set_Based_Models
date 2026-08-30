"""Ready-to-use scores for ranking generated existential rules.

All scorers accept one ``RuleStatistics`` instance and return a finite scalar
for which larger is better.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .statistics import RuleStatistics


@dataclass(frozen=True, slots=True)
class TPRFPRRuleScore:
    """Score ``TPR - lambda_fpr * FPR - length_penalty * depth``."""

    lambda_fpr: float = 1.0
    length_penalty: float = 0.0

    def __post_init__(self) -> None:
        _nonnegative(self.lambda_fpr, "lambda_fpr")
        _nonnegative(self.length_penalty, "length_penalty")

    def __call__(self, statistics: RuleStatistics) -> float:
        return float(
            statistics.true_positive_rate
            - self.lambda_fpr * statistics.false_positive_rate
            - self.length_penalty * statistics.rule_depth
        )


@dataclass(frozen=True, slots=True)
class TargetFPRRuleScore:
    """Softly penalize only the portion of FPR above a target."""

    target_fpr: float
    violation_penalty: float
    length_penalty: float = 0.0

    def __post_init__(self) -> None:
        _unit_interval(self.target_fpr, "target_fpr")
        _nonnegative(self.violation_penalty, "violation_penalty")
        _nonnegative(self.length_penalty, "length_penalty")

    def __call__(self, statistics: RuleStatistics) -> float:
        violation = max(0.0, statistics.false_positive_rate - self.target_fpr)
        return float(
            statistics.true_positive_rate
            - self.violation_penalty * violation
            - self.length_penalty * statistics.rule_depth
        )


@dataclass(frozen=True, slots=True)
class PurityCoverageRuleScore:
    """Reward rules that are simultaneously pure and cover signal."""

    length_penalty: float = 0.0

    def __post_init__(self) -> None:
        _nonnegative(self.length_penalty, "length_penalty")

    def __call__(self, statistics: RuleStatistics) -> float:
        return float(
            statistics.precision * statistics.true_positive_rate
            - self.length_penalty * statistics.rule_depth
        )


@dataclass(frozen=True, slots=True)
class FScoreRuleScore:
    """Weighted harmonic mean of precision and signal efficiency."""

    beta: float = 1.0
    length_penalty: float = 0.0

    def __post_init__(self) -> None:
        _positive(self.beta, "beta")
        _nonnegative(self.length_penalty, "length_penalty")

    def __call__(self, statistics: RuleStatistics) -> float:
        precision = statistics.precision
        recall = statistics.true_positive_rate
        beta_squared = self.beta * self.beta
        denominator = beta_squared * precision + recall
        f_score = (
            (1.0 + beta_squared) * precision * recall / denominator
            if denominator > 0
            else 0.0
        )
        return float(f_score - self.length_penalty * statistics.rule_depth)


@dataclass(frozen=True, slots=True)
class SignificanceRuleScore:
    """Score ``S² / (S+B+(fB)²)`` with an optional length penalty."""

    background_uncertainty_fraction: float = 0.0
    length_penalty: float = 0.0

    def __post_init__(self) -> None:
        _nonnegative(
            self.background_uncertainty_fraction,
            "background_uncertainty_fraction",
        )
        _nonnegative(self.length_penalty, "length_penalty")

    def __call__(self, statistics: RuleStatistics) -> float:
        signal = statistics.signal_weight_accepted
        background = statistics.background_weight_accepted
        denominator = (
            signal
            + background
            + (self.background_uncertainty_fraction * background) ** 2
        )
        significance_squared = signal * signal / denominator if denominator > 0 else 0.0
        return float(
            significance_squared - self.length_penalty * statistics.rule_depth
        )


@dataclass(frozen=True, slots=True)
class LikelihoodRatioRuleScore:
    """Smoothed log ratio of signal efficiency to background efficiency."""

    signal_smoothing: float = 1e-12
    background_smoothing: float = 1e-12
    length_penalty: float = 0.0

    def __post_init__(self) -> None:
        _positive(self.signal_smoothing, "signal_smoothing")
        _positive(self.background_smoothing, "background_smoothing")
        _nonnegative(self.length_penalty, "length_penalty")

    def __call__(self, statistics: RuleStatistics) -> float:
        ratio = (
            statistics.true_positive_rate + self.signal_smoothing
        ) / (
            statistics.false_positive_rate + self.background_smoothing
        )
        return float(
            np.log(ratio) - self.length_penalty * statistics.rule_depth
        )


def _nonnegative(value: float, name: str) -> None:
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")


def _positive(value: float, name: str) -> None:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _unit_interval(value: float, name: str) -> None:
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie between zero and one")
