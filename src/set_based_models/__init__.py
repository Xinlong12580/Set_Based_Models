"""ERL and shared utilities, with lazy compatibility imports for optional EDT."""

from .core import (
    BinaryPerformance, CategoryCut, Comparison, CountAggregation, Cut, EventCut,
    ObservableKind, ObservableSpec, SetInput, SlotCut, ThresholdCut,
)
from .core.objectives import (
    DirectTPRFPRGain, SignificanceGain, WeightedGiniGain, entropy_gain, gini_gain,
    make_direct_tpr_fpr_gain, make_significance_gain,
    make_weighted_gini_gain, misclassification_gain,
)
from .core.statistics import PartitionStatistics, SplitStatistics
from .rule_list import (
    BeamRuleGenerator, ExistentialRule, ExistentialRuleList, FScoreRuleScore,
    LikelihoodRatioRuleScore, PurityCoverageRuleScore, RuleScorer,
    RulePruningResult, RuleStatistics, RuleStep, RuleUnion, SignificanceRuleScore, TPRFPRRuleScore,
    TargetFPRRuleScore, placeholder_rule_score,
)

__all__ = [name for name in globals() if not name.startswith("_")]

# Explicit historical imports still work when the extension is installed.
# They are deliberately excluded from __all__: importing ERL (including import *)
# must not load or require EDT.
_EDT_EXPORTS = frozenset({
    "Branch", "ChainEvaluation", "ExistentialDecisionTree",
    "Node", "NodeStatistics", "PathStep",
})


def _load_edt():
    from importlib import import_module

    try:
        return import_module("set_based_edt")
    except ModuleNotFoundError as error:
        if error.name != "set_based_edt":
            raise
        raise ModuleNotFoundError(
            "EDT is an optional extension. From the repository root, install it "
            "with: python -m pip install -e extensions/edt",
            name="set_based_edt",
        ) from error


def __getattr__(name):
    if name in _EDT_EXPORTS:
        return getattr(_load_edt(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
