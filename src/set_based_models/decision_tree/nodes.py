"""Persistent node types for a fitted existential decision tree."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..core.predicates import CountAggregation, SlotCut


class Branch(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


@dataclass(frozen=True, slots=True)
class NodeStatistics:
    n_signal: int
    n_background: int
    signal_weight: float
    background_weight: float
    impurity: float | None = None
    split_gain: float | None = None

    @property
    def signal_fraction(self) -> float:
        total = self.signal_weight + self.background_weight
        return self.signal_weight / total if total else 0.0


@dataclass(slots=True)
class Node:
    """One fitted node, linked to its relatives by stable integer IDs."""

    node_id: int
    depth: int
    parent_id: int | None = None
    branch_from_parent: Branch | str | None = None
    cut: SlotCut | None = None
    aggregation: CountAggregation | None = None
    positive_child_id: int | None = None
    negative_child_id: int | None = None
    statistics: NodeStatistics | None = None
    score: float | None = None
    predicted_class: int | None = None

    def __post_init__(self) -> None:
        if self.node_id < 0 or self.depth < 0:
            raise ValueError("node_id and depth must be non-negative")
        if self.branch_from_parent is not None:
            self.branch_from_parent = Branch(self.branch_from_parent)
        if self.predicted_class not in (None, 0, 1):
            raise ValueError("predicted_class must be 0, 1, or None")
        if self.score is not None and not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be between zero and one")

    @property
    def is_leaf(self) -> bool:
        return self.cut is None


@dataclass(frozen=True, slots=True)
class PathStep:
    source_node_id: int
    cut: SlotCut
    aggregation: CountAggregation
    branch: Branch
