"""Existential decision-tree model and training objectives."""

from .model import ChainEvaluation, ExistentialDecisionTree
from .nodes import Branch, Node, NodeStatistics, PathStep
from .objectives import (
    DirectTPRFPRGain, SignificanceGain, WeightedGiniGain, entropy_gain,
    gini_gain, make_direct_tpr_fpr_gain, make_significance_gain,
    make_weighted_gini_gain, misclassification_gain,
)
from .training import PartitionStatistics, SplitStatistics

__all__ = [
    "Branch", "ChainEvaluation", "DirectTPRFPRGain", "ExistentialDecisionTree",
    "Node", "NodeStatistics", "PartitionStatistics", "PathStep",
    "SignificanceGain", "SplitStatistics", "WeightedGiniGain", "entropy_gain",
    "gini_gain", "make_direct_tpr_fpr_gain", "make_significance_gain",
    "make_weighted_gini_gain", "misclassification_gain",
]
