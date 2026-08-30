"""Fitted-tree representation and evaluation for existential decision trees."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..core.data import ObservableKind, ObservableSpec, SetInput
from ..core.metrics import BinaryPerformance
from ..core.predicates import (
    CategoryCut, Comparison, CountAggregation, SlotCut, ThresholdCut,
)
from .nodes import Branch, Node, NodeStatistics, PathStep
from .training import (
    AppliedClassSplit, AppliedSplit, ClassTrainingState, NodeTrainingState,
    PartitionStatistics, SplitStatistics,
)


@dataclass(frozen=True, slots=True)
class ChainEvaluation:
    selected_samples: NDArray[np.bool_]
    surviving_slots: NDArray[np.bool_]


class ExistentialDecisionTree:
    """A shallow binary tree whose positive branches refine slot masks.

    A caller-supplied gain function scores each mask-aware event partition. Its
    input is :class:`SplitStatistics`, and larger returned values are better.
    """

    def __init__(
        self,
        observables: Sequence[ObservableSpec],
        *,
        max_depth: int = 6,
        criterion: str | Callable[..., float] = "gini",
        splitting_method: str = "greedy",
        decision_threshold: float = 0.5,
        gain_function: Callable[[SplitStatistics], float] | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_gain: float = 0.0,
        minimum_count: int = 1,
        max_thresholds: int | None = 64,
    ) -> None:
        self.observables = tuple(observables)
        if not self.observables or not all(
            isinstance(spec, ObservableSpec) for spec in self.observables
        ):
            raise TypeError("observables must be a non-empty sequence of ObservableSpec")
        if len({spec.name for spec in self.observables}) != len(self.observables):
            raise ValueError("Observable names must be unique")
        if isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 0:
            raise ValueError("max_depth must be a non-negative integer")
        if not callable(criterion) and criterion not in {"gini", "entropy", "cost_sensitive"}:
            raise ValueError("criterion must be gini, entropy, cost_sensitive, or a callable")
        if splitting_method not in {"greedy", "lookahead", "beam"}:
            raise ValueError("splitting_method must be greedy, lookahead, or beam")
        if not 0.0 <= decision_threshold <= 1.0:
            raise ValueError("decision_threshold must be between zero and one")
        if gain_function is not None and not callable(gain_function):
            raise TypeError("gain_function must be callable")
        if min_samples_split < 2 or min_samples_leaf < 1:
            raise ValueError("min_samples_split must be >= 2 and min_samples_leaf >= 1")
        if not np.isfinite(min_gain):
            raise ValueError("min_gain must be finite")
        if max_thresholds is not None and max_thresholds < 1:
            raise ValueError("max_thresholds must be positive or None")
        self.max_depth = max_depth
        self.criterion = criterion
        self.splitting_method = splitting_method
        self.decision_threshold = decision_threshold
        self.gain_function = gain_function
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain
        self.aggregation = CountAggregation(minimum_count)
        self.max_thresholds = max_thresholds
        self.nodes_: list[Node] = []
        self.root_id_: int | None = None
        self._root_statistics_: PartitionStatistics | None = None

    def fit(
        self,
        signal: ArrayLike,
        background: ArrayLike,
        *,
        signal_slot_mask: ArrayLike | None = None,
        background_slot_mask: ArrayLike | None = None,
        signal_weight: ArrayLike | None = None,
        background_weight: ArrayLike | None = None,
        gain_function: Callable[[SplitStatistics], float] | None = None,
    ) -> "ExistentialDecisionTree":
        """Greedily grow a mask-aware tree from separate class tensors."""

        signal_data = self._data(signal, signal_slot_mask, signal_weight)
        background_data = self._data(background, background_slot_mask, background_weight)
        if signal_data.X.shape[1:] != background_data.X.shape[1:]:
            raise ValueError("Signal and background must have identical slot/feature shape")
        scorer = gain_function or self.gain_function
        if scorer is None:
            raise ValueError("fit() requires an external gain_function")
        if self.splitting_method != "greedy":
            raise NotImplementedError("Only greedy training is currently implemented")
        signal_weights = self._training_weights(signal_data)
        background_weights = self._training_weights(background_data)
        root_state = NodeTrainingState(
            ClassTrainingState(np.arange(signal_data.n_samples), signal_data.slot_mask.copy()),
            ClassTrainingState(
                np.arange(background_data.n_samples), background_data.slot_mask.copy()
            ),
        )
        self._root_statistics_ = self._partition_statistics(
            root_state, signal_weights, background_weights
        )
        self.nodes_ = [Node(0, 0)]
        self.root_id_ = 0
        self._grow_node(
            0, root_state, signal_data, background_data,
            signal_weights, background_weights, scorer,
        )
        self._validate_tree(self.nodes_, 0)
        return self

    def set_tree(self, nodes: Sequence[Node], *, root_id: int = 0) -> "ExistentialDecisionTree":
        """Install and validate an already-trained tree."""

        installed = list(nodes)
        self._validate_tree(installed, root_id)
        self.nodes_ = installed
        self.root_id_ = root_id
        return self

    def get_node(self, node_id: int) -> Node:
        self._require_fitted()
        if not isinstance(node_id, Integral) or not 0 <= node_id < len(self.nodes_):
            raise KeyError(f"Unknown node {node_id}")
        node_id = int(node_id)
        node = self.nodes_[node_id]
        if node.node_id != node_id:
            raise KeyError(f"Unknown node {node_id}")
        return node

    def cut_chain(self, node_id: int) -> tuple[PathStep, ...]:
        """Return the exact ordered event-routing path to a node."""

        current = self.get_node(node_id)
        reverse: list[PathStep] = []
        while current.parent_id is not None:
            parent = self.get_node(current.parent_id)
            assert parent.cut is not None and parent.aggregation is not None
            assert current.branch_from_parent is not None
            reverse.append(
                PathStep(parent.node_id, parent.cut, parent.aggregation, current.branch_from_parent)
            )
            current = parent
        return tuple(reversed(reverse))

    def slot_cut_chain(self, node_id: int) -> tuple[SlotCut, ...]:
        """Return the positive-branch slot-cut conjunction leading to a node."""

        return tuple(step.cut for step in self.cut_chain(node_id) if step.branch is Branch.POSITIVE)

    def cut_chain_text(self, node_id: int) -> str:
        steps = self.cut_chain(node_id)
        if not steps:
            return "ROOT (all samples, all valid slots)"
        lines = []
        for step in steps:
            outcome = "PASS" if step.branch is Branch.POSITIVE else "FAIL"
            lines.append(
                f"node {step.source_node_id}: {outcome} "
                f"[{step.aggregation.to_text()}: {step.cut.to_text(self.observables)}]"
            )
        return "\n".join(lines)

    def evaluate_chain(
        self, node_id: int, X: ArrayLike, *, slot_mask: ArrayLike | None = None
    ) -> ChainEvaluation:
        """Evaluate a node's exact root-to-node chain on every sample."""

        data = self._data(X, slot_mask)
        selected = np.ones(data.n_samples, dtype=np.bool_)
        active = data.slot_mask.copy()
        for step in self.cut_chain(node_id):
            candidates = active & step.cut.evaluate(data)
            passed = step.aggregation.evaluate(candidates)
            selected &= passed if step.branch is Branch.POSITIVE else ~passed
            if step.branch is Branch.POSITIVE:
                active = np.where(selected[:, None], candidates, active)
        return ChainEvaluation(selected, active & selected[:, None])

    def slot_cut_function(
        self, node_id: int
    ) -> Callable[[ArrayLike, ArrayLike | None], NDArray[np.bool_]]:
        """Compile the positive slot cuts without event-level aggregation."""

        cuts = self.slot_cut_chain(node_id)

        def evaluate(X: ArrayLike, slot_mask: ArrayLike | None = None) -> NDArray[np.bool_]:
            data = self._data(X, slot_mask)
            result = data.slot_mask.copy()
            for cut in cuts:
                result &= cut.evaluate(data)
            return result

        return evaluate

    def apply(self, X: ArrayLike, *, slot_mask: ArrayLike | None = None) -> NDArray[np.int64]:
        """Return the terminal node ID reached by every sample."""

        self._require_fitted()
        data = self._data(X, slot_mask)
        active = data.slot_mask.copy()
        current = np.full(data.n_samples, self.root_id_, dtype=np.int64)
        unfinished = np.ones(data.n_samples, dtype=np.bool_)
        while np.any(unfinished):
            for node_id in np.unique(current[unfinished]):
                node = self.get_node(int(node_id))
                rows = np.flatnonzero(unfinished & (current == node_id))
                if node.is_leaf:
                    unfinished[rows] = False
                    continue
                assert node.cut is not None and node.aggregation is not None
                subset = self._slice_data(data, rows)
                candidates = active[rows] & node.cut.evaluate(subset)
                passed = node.aggregation.evaluate(candidates)
                positive_rows, negative_rows = rows[passed], rows[~passed]
                active[positive_rows] = candidates[passed]
                assert node.positive_child_id is not None and node.negative_child_id is not None
                current[positive_rows] = node.positive_child_id
                current[negative_rows] = node.negative_child_id
        return current

    def decision_function(
        self, X: ArrayLike, *, slot_mask: ArrayLike | None = None
    ) -> NDArray[np.float64]:
        leaves = self.apply(X, slot_mask=slot_mask)
        scores = np.empty(leaves.size, dtype=np.float64)
        for node_id in np.unique(leaves):
            node = self.get_node(int(node_id))
            score = node.score
            if score is None and node.statistics is not None:
                score = node.statistics.signal_fraction
            if score is None:
                raise ValueError(f"Leaf {node_id} has neither a score nor statistics")
            scores[leaves == node_id] = score
        return scores

    def predict(self, X: ArrayLike, *, slot_mask: ArrayLike | None = None) -> NDArray[np.int64]:
        leaves = self.apply(X, slot_mask=slot_mask)
        result = np.empty(leaves.size, dtype=np.int64)
        for node_id in np.unique(leaves):
            node = self.get_node(int(node_id))
            if node.predicted_class is not None:
                prediction = node.predicted_class
            else:
                score = node.score
                if score is None and node.statistics is not None:
                    score = node.statistics.signal_fraction
                if score is None:
                    raise ValueError(f"Leaf {node_id} has no classification output")
                prediction = int(score >= self.decision_threshold)
            result[leaves == node_id] = prediction
        return result

    def evaluate(
        self,
        signal: ArrayLike,
        background: ArrayLike,
        *,
        signal_slot_mask: ArrayLike | None = None,
        background_slot_mask: ArrayLike | None = None,
        signal_weight: ArrayLike | None = None,
        background_weight: ArrayLike | None = None,
    ) -> BinaryPerformance:
        return BinaryPerformance.from_selections(
            self.predict(signal, slot_mask=signal_slot_mask).astype(bool),
            self.predict(background, slot_mask=background_slot_mask).astype(bool),
            signal_weight=signal_weight,
            background_weight=background_weight,
        )

    def evaluate_node(
        self,
        node_id: int,
        signal: ArrayLike,
        background: ArrayLike,
        *,
        signal_slot_mask: ArrayLike | None = None,
        background_slot_mask: ArrayLike | None = None,
        signal_weight: ArrayLike | None = None,
        background_weight: ArrayLike | None = None,
    ) -> BinaryPerformance:
        """Treat reaching ``node_id`` as acceptance and evaluate its chain."""

        return BinaryPerformance.from_selections(
            self.evaluate_chain(node_id, signal, slot_mask=signal_slot_mask).selected_samples,
            self.evaluate_chain(
                node_id, background, slot_mask=background_slot_mask
            ).selected_samples,
            signal_weight=signal_weight,
            background_weight=background_weight,
        )

    def _data(
        self,
        X: ArrayLike,
        slot_mask: ArrayLike | None = None,
        sample_weight: ArrayLike | None = None,
    ) -> SetInput:
        return SetInput.from_arrays(
            X, self.observables, slot_mask=slot_mask, sample_weight=sample_weight
        )

    def _grow_node(self, node_id, state, signal_data, background_data,
                   signal_weights, background_weights, scorer) -> None:
        node = self.nodes_[node_id]
        parent_stats = self._partition_statistics(state, signal_weights, background_weights)
        node.statistics = self._node_statistics(parent_stats)
        if (node.depth >= self.max_depth
                or parent_stats.total_count < self.min_samples_split
                or parent_stats.signal_count == 0
                or parent_stats.background_count == 0):
            self._make_leaf(node)
            return
        best_cut, best_split, best_gain = None, None, -np.inf
        for cut in self._candidate_cuts(state, signal_data, background_data):
            split = self._apply_candidate(
                cut, state, signal_data, background_data,
                signal_weights, background_weights,
            )
            if (split.statistics.positive.total_count < self.min_samples_leaf
                    or split.statistics.negative.total_count < self.min_samples_leaf):
                continue
            gain = float(scorer(split.statistics))
            if np.isfinite(gain) and gain > best_gain:
                best_cut, best_split, best_gain = cut, split, gain
        if best_cut is None or best_split is None or best_gain <= self.min_gain:
            self._make_leaf(node)
            return
        node.cut, node.aggregation = best_cut, self.aggregation
        node.statistics = self._node_statistics(parent_stats, best_gain)
        positive_id, negative_id = len(self.nodes_), len(self.nodes_) + 1
        node.positive_child_id, node.negative_child_id = positive_id, negative_id
        self.nodes_.extend([
            Node(positive_id, node.depth + 1, node_id, Branch.POSITIVE),
            Node(negative_id, node.depth + 1, node_id, Branch.NEGATIVE),
        ])
        positive_state, negative_state = self._child_states(state, best_split)
        self._grow_node(positive_id, positive_state, signal_data, background_data,
                        signal_weights, background_weights, scorer)
        self._grow_node(negative_id, negative_state, signal_data, background_data,
                        signal_weights, background_weights, scorer)

    def _candidate_cuts(self, state, signal_data, background_data) -> tuple[SlotCut, ...]:
        candidates: list[SlotCut] = []
        for index, spec in enumerate(self.observables):
            if spec.kind in (ObservableKind.CONTINUOUS, ObservableKind.ORDINAL):
                arrays = []
                for class_state, data in ((state.signal, signal_data),
                                          (state.background, background_data)):
                    rows = class_state.sample_indices
                    if rows.size:
                        valid = class_state.active_slot_mask & data.value_mask[rows, :, index]
                        arrays.append(data.X[rows, :, index][valid])
                if not arrays:
                    continue
                unique = np.unique(np.concatenate(arrays))
                if unique.size < 2:
                    continue
                thresholds = unique[:-1] + np.diff(unique) / 2.0
                if self.max_thresholds is not None and thresholds.size > self.max_thresholds:
                    positions = np.linspace(0, thresholds.size - 1,
                                            self.max_thresholds, dtype=np.int64)
                    thresholds = thresholds[np.unique(positions)]
                for threshold in thresholds:
                    candidates.extend([
                        ThresholdCut(index, Comparison.GREATER, float(threshold)),
                        ThresholdCut(index, Comparison.LESS_EQUAL, float(threshold)),
                    ])
            else:
                allowed = spec.allowed_codes
                assert allowed is not None
                candidates.extend(CategoryCut(index, {code}) for code in sorted(allowed))
        return tuple(candidates)

    def _apply_candidate(self, cut, state, signal_data, background_data,
                         signal_weights, background_weights) -> AppliedSplit:
        signal = self._apply_to_class(cut, state.signal, signal_data)
        background = self._apply_to_class(cut, state.background, background_data)
        stats = SplitStatistics(
            self._partition_statistics(state, signal_weights, background_weights),
            PartitionStatistics(
                int(signal.positive.sum()), int(background.positive.sum()),
                float(signal_weights[state.signal.sample_indices[signal.positive]].sum()),
                float(background_weights[state.background.sample_indices[background.positive]].sum()),
            ),
            PartitionStatistics(
                int(signal.negative.sum()), int(background.negative.sum()),
                float(signal_weights[state.signal.sample_indices[signal.negative]].sum()),
                float(background_weights[state.background.sample_indices[background.negative]].sum()),
            ),
            root=self._root_statistics_,
        )
        return AppliedSplit(signal, background, stats)

    def _apply_to_class(self, cut, state, data) -> AppliedClassSplit:
        subset = self._slice_data(data, state.sample_indices)
        candidate_mask = state.active_slot_mask & cut.evaluate(subset)
        return AppliedClassSplit(self.aggregation.evaluate(candidate_mask), candidate_mask)

    @staticmethod
    def _child_states(state, split) -> tuple[NodeTrainingState, NodeTrainingState]:
        def children(parent, result):
            return (
                ClassTrainingState(parent.sample_indices[result.positive],
                                   result.candidate_mask[result.positive]),
                ClassTrainingState(parent.sample_indices[result.negative],
                                   parent.active_slot_mask[result.negative]),
            )
        sp, sn = children(state.signal, split.signal)
        bp, bn = children(state.background, split.background)
        return NodeTrainingState(sp, bp), NodeTrainingState(sn, bn)

    @staticmethod
    def _partition_statistics(state, signal_weights, background_weights):
        return PartitionStatistics(
            state.signal.sample_indices.size, state.background.sample_indices.size,
            float(signal_weights[state.signal.sample_indices].sum()),
            float(background_weights[state.background.sample_indices].sum()),
        )

    @staticmethod
    def _node_statistics(stats, split_gain=None) -> NodeStatistics:
        return NodeStatistics(stats.signal_count, stats.background_count,
                              stats.signal_weight, stats.background_weight,
                              split_gain=split_gain)

    @staticmethod
    def _training_weights(data) -> NDArray[np.float64]:
        return (np.ones(data.n_samples, dtype=np.float64)
                if data.sample_weight is None else data.sample_weight)

    @staticmethod
    def _make_leaf(node: Node) -> None:
        assert node.statistics is not None
        node.score = node.statistics.signal_fraction

    @staticmethod
    def _slice_data(data: SetInput, rows: NDArray[np.int64]) -> SetInput:
        return SetInput(
            X=data.X[rows], observables=data.observables,
            slot_mask=data.slot_mask[rows], value_mask=data.value_mask[rows],
        )

    def _require_fitted(self) -> None:
        if self.root_id_ is None:
            raise RuntimeError("No fitted tree is installed; call set_tree() first")

    def _validate_tree(self, nodes: list[Node], root_id: int) -> None:
        if not nodes or [node.node_id for node in nodes] != list(range(len(nodes))):
            raise ValueError("Nodes must be non-empty, ordered, and indexed consecutively from zero")
        if not 0 <= root_id < len(nodes):
            raise ValueError("root_id does not identify a node")
        root = nodes[root_id]
        if root.parent_id is not None or root.branch_from_parent is not None or root.depth != 0:
            raise ValueError("The root must have depth zero and no parent branch")
        visited: set[int] = set()

        def visit(node_id: int, depth: int) -> None:
            if node_id in visited:
                raise ValueError("Tree contains a cycle or a node with multiple parents")
            visited.add(node_id)
            node = nodes[node_id]
            if node.depth != depth or depth > self.max_depth:
                raise ValueError(f"Invalid depth for node {node_id}")
            if node.is_leaf:
                if any(x is not None for x in (
                    node.aggregation, node.positive_child_id, node.negative_child_id
                )):
                    raise ValueError(f"Leaf {node_id} contains split information")
                return
            if node.aggregation is None:
                raise ValueError(f"Internal node {node_id} requires an aggregation")
            node.cut.validate(self.observables)
            if node.positive_child_id is None or node.negative_child_id is None:
                raise ValueError(f"Internal node {node_id} requires two children")
            for child_id, branch in (
                (node.positive_child_id, Branch.POSITIVE),
                (node.negative_child_id, Branch.NEGATIVE),
            ):
                if not 0 <= child_id < len(nodes):
                    raise ValueError(f"Node {node_id} refers to unknown child {child_id}")
                child = nodes[child_id]
                if child.parent_id != node_id or child.branch_from_parent is not branch:
                    raise ValueError(f"Child {child_id} has inconsistent parent bookkeeping")
                visit(child_id, depth + 1)

        visit(root_id, 0)
        if len(visited) != len(nodes):
            raise ValueError("Tree contains nodes unreachable from the root")
