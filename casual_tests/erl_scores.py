"""External ERL score plugins; all use weighted mixture/reference acceptance."""

from dataclasses import dataclass
import math


SCORE_NAMES = ("gini", "excess", "ratio")
DEFAULT_DELTA = 0.001


def _check_delta(delta):
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be positive and finite so zero-reference rules have finite scores")


@dataclass(frozen=True)
class ExcessRuleScore:
    """max(0, mixture acceptance - reference acceptance) / sqrt(b + delta)."""

    delta: float = DEFAULT_DELTA

    def __post_init__(self):
        _check_delta(self.delta)

    def __call__(self, statistics):
        a, b = statistics.true_positive_rate, statistics.false_positive_rate
        return max(0., a - b) / math.sqrt(b + self.delta)


@dataclass(frozen=True)
class AcceptanceRatioRuleScore:
    """Mixture acceptance / sqrt(reference acceptance + delta)."""

    delta: float = DEFAULT_DELTA

    def __post_init__(self):
        _check_delta(self.delta)

    def __call__(self, statistics):
        return statistics.true_positive_rate / math.sqrt(statistics.false_positive_rate + self.delta)


def make_score(name="gini", *, reference_penalty=1., delta=DEFAULT_DELTA):
    if name == "gini":
        from casual_tests._erl_common import ReferencePenalizedGini
        return ReferencePenalizedGini(reference_penalty)
    return {"excess": ExcessRuleScore, "ratio": AcceptanceRatioRuleScore}[name](delta)


def score_settings(name="gini", *, reference_penalty=1., delta=DEFAULT_DELTA):
    formulas = {
        "gini": "gini_gain - reference_penalty * reference_acceptance",
        "excess": "max(0, mixture_acceptance - reference_acceptance) / sqrt(reference_acceptance + delta)",
        "ratio": "mixture_acceptance / sqrt(reference_acceptance + delta)",
    }
    return {
        "score_name": name,
        "scoring_function": {"gini": "gini_gain_minus_reference_acceptance_penalty",
                             "excess": "regularized_excess_acceptance", "ratio": "regularized_acceptance_ratio"}[name],
        "score_formula": formulas[name],
        "score_parameters": {"reference_penalty": reference_penalty} if name == "gini" else {"delta": delta},
        "reference_penalty": reference_penalty if name == "gini" else None,
        "score_delta": delta if name != "gini" else None,
        "gini_gain_role": "objective_component" if name == "gini" else "diagnostic_only",
    }


def validation_score_settings(args, prefix):
    """Inherit a uniform beam choice; mixed grids default to Gini and DEFAULT_DELTA."""
    def single(value, default):
        if isinstance(value, list):
            return value[0] if len(set(value)) == 1 else default
        return value

    name = getattr(args, f"{prefix}_score")
    delta = getattr(args, f"{prefix}_score_delta")
    return (name if name is not None else single(args.score, "gini"),
            delta if delta is not None else single(args.score_delta, DEFAULT_DELTA))
