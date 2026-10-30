"""Weighted event partitions shared by rule and tree scoring."""

from __future__ import annotations

from dataclasses import dataclass


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
