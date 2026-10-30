"""Scoring and argument helpers shared by the current ERL study runner."""

import argparse
from dataclasses import dataclass
import math

from set_based_models import RuleStatistics
from casual_tests import _quantitative_reporting as reporting


@dataclass(frozen=True)
class ReferencePenalizedGini:
    """Gini gain minus reference acceptance, applied throughout beam search.

    Process labels never enter the objective. Negative scores remain eligible
    so that an initially weak prefix can become clean after another cut.
    """

    reference_penalty: float = 1.0

    def __post_init__(self):
        if not math.isfinite(self.reference_penalty) or self.reference_penalty < 0:
            raise ValueError("reference_penalty must be finite and nonnegative.")

    def __call__(self, statistics: RuleStatistics) -> float:
        return (reporting.gini_rule_score(statistics)
                - self.reference_penalty * statistics.false_positive_rate)


def _finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise argparse.ArgumentTypeError("value must be finite")
    return result


def _positive_float(value):
    result = _finite_float(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _nonnegative_float(value):
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError("value must be finite and nonnegative")
    return result


def _similarity_threshold(value):
    result = _nonnegative_float(value)
    if result > 1:
        raise argparse.ArgumentTypeError("similarity threshold must be between 0 and 1")
    return result
