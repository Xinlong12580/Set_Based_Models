"""Public gain-function inputs and internal mask-aware training state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class PartitionStatistics:
    """Weighted event yields in one region of a candidate split."""

    signal_count: int
    background_count: int
    signal_weight: float
    background_weight: float

    @property
    def total_count(self) -> int:
        return self.signal_count + self.background_count

    @property
    def total_weight(self) -> float:
        return self.signal_weight + self.background_weight

    @property
    def signal_fraction(self) -> float:
        return self.signal_weight / self.total_weight if self.total_weight else 0.0


@dataclass(frozen=True, slots=True)
class SplitStatistics:
    """The sole argument passed to an externally supplied gain function.

    Gain functions must return one finite scalar, with larger values denoting a
    better split.
    """

    parent: PartitionStatistics
    positive: PartitionStatistics
    negative: PartitionStatistics
    root: PartitionStatistics | None = None


@dataclass(slots=True)
class ClassTrainingState:
    """Temporary state for one class at one node."""

    sample_indices: NDArray[np.int64]
    active_slot_mask: NDArray[np.bool_]


@dataclass(slots=True)
class NodeTrainingState:
    signal: ClassTrainingState
    background: ClassTrainingState


@dataclass(slots=True)
class AppliedClassSplit:
    positive: NDArray[np.bool_]
    candidate_mask: NDArray[np.bool_]

    @property
    def negative(self) -> NDArray[np.bool_]:
        return ~self.positive


@dataclass(slots=True)
class AppliedSplit:
    signal: AppliedClassSplit
    background: AppliedClassSplit
    statistics: SplitStatistics
