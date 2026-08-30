"""Binary selection metrics for whole trees and individual node chains."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True, slots=True)
class BinaryPerformance:
    signal_accepted: int
    signal_total: int
    background_accepted: int
    background_total: int
    signal_weight_accepted: float
    signal_weight_total: float
    background_weight_accepted: float
    background_weight_total: float

    @property
    def true_positive_rate(self) -> float:
        return _ratio(self.signal_weight_accepted, self.signal_weight_total)

    @property
    def false_positive_rate(self) -> float:
        return _ratio(self.background_weight_accepted, self.background_weight_total)

    @property
    def precision(self) -> float:
        return _ratio(
            self.signal_weight_accepted,
            self.signal_weight_accepted + self.background_weight_accepted,
        )

    @classmethod
    def from_selections(
        cls,
        signal_selection: ArrayLike,
        background_selection: ArrayLike,
        *,
        signal_weight: ArrayLike | None = None,
        background_weight: ArrayLike | None = None,
    ) -> "BinaryPerformance":
        signal = np.asarray(signal_selection, dtype=np.bool_)
        background = np.asarray(background_selection, dtype=np.bool_)
        if signal.ndim != 1 or background.ndim != 1:
            raise ValueError("Selections must be one-dimensional")
        sw = _weights(signal_weight, signal.size, "signal_weight")
        bw = _weights(background_weight, background.size, "background_weight")
        return cls(
            int(np.count_nonzero(signal)), signal.size,
            int(np.count_nonzero(background)), background.size,
            float(sw[signal].sum()), float(sw.sum()),
            float(bw[background].sum()), float(bw.sum()),
        )


def _weights(value: ArrayLike | None, size: int, name: str) -> np.ndarray:
    if value is None:
        return np.ones(size, dtype=np.float64)
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    if np.any(~np.isfinite(result)) or np.any(result < 0):
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
