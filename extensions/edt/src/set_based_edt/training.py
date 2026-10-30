"""Internal mask-aware EDT training state and shared statistics aliases."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from set_based_models.core.statistics import PartitionStatistics, SplitStatistics


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
