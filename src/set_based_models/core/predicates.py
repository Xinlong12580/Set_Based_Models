"""Slot-level cuts and event-level aggregation rules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from .data import ObservableKind, ObservableSpec, SetInput


class Comparison(str, Enum):
    GREATER = ">"
    GREATER_EQUAL = ">="
    LESS = "<"
    LESS_EQUAL = "<="


@dataclass(frozen=True, slots=True)
class ThresholdCut:
    observable_index: int
    comparison: Comparison | str
    threshold: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "comparison", Comparison(self.comparison))
        if self.observable_index < 0 or not np.isfinite(self.threshold):
            raise ValueError("observable_index must be non-negative and threshold finite")

    def validate(self, observables: tuple[ObservableSpec, ...]) -> None:
        try:
            spec = observables[self.observable_index]
        except IndexError as exc:
            raise ValueError(f"Unknown observable index {self.observable_index}") from exc
        if not spec.allows_threshold:
            raise TypeError(f"Observable {spec.name!r} does not allow threshold cuts")

    def evaluate(self, data: SetInput) -> NDArray[np.bool_]:
        self.validate(data.observables)
        values = data.X[..., self.observable_index]
        if self.comparison is Comparison.GREATER:
            result = values > self.threshold
        elif self.comparison is Comparison.GREATER_EQUAL:
            result = values >= self.threshold
        elif self.comparison is Comparison.LESS:
            result = values < self.threshold
        else:
            result = values <= self.threshold
        return data.value_mask[..., self.observable_index] & result

    def to_text(self, observables: tuple[ObservableSpec, ...]) -> str:
        self.validate(observables)
        return f"{observables[self.observable_index].name} {self.comparison.value} {self.threshold:g}"


@dataclass(frozen=True, slots=True, init=False)
class CategoryCut:
    observable_index: int
    accepted_codes: frozenset[int]

    def __init__(self, observable_index: int, accepted_codes: set[int] | frozenset[int]) -> None:
        codes = frozenset(accepted_codes)
        if observable_index < 0 or not codes or any(
            isinstance(code, bool) or not isinstance(code, int) for code in codes
        ):
            raise ValueError("Category cuts require a valid index and one or more integer codes")
        object.__setattr__(self, "observable_index", observable_index)
        object.__setattr__(self, "accepted_codes", codes)

    def validate(self, observables: tuple[ObservableSpec, ...]) -> None:
        try:
            spec = observables[self.observable_index]
        except IndexError as exc:
            raise ValueError(f"Unknown observable index {self.observable_index}") from exc
        if not spec.allows_membership:
            raise TypeError(f"Observable {spec.name!r} does not allow category cuts")
        allowed = spec.allowed_codes
        assert allowed is not None
        if self.accepted_codes - allowed:
            raise ValueError(f"Unknown category code for observable {spec.name!r}")

    def evaluate(self, data: SetInput) -> NDArray[np.bool_]:
        self.validate(data.observables)
        values = data.X[..., self.observable_index]
        return data.value_mask[..., self.observable_index] & np.isin(
            values, tuple(self.accepted_codes)
        )

    def to_text(self, observables: tuple[ObservableSpec, ...]) -> str:
        self.validate(observables)
        spec = observables[self.observable_index]
        if spec.kind is ObservableKind.CATEGORICAL:
            assert spec.categories is not None
            labels = [spec.categories[code] for code in sorted(self.accepted_codes)]
        else:
            labels = [str(bool(code)) for code in sorted(self.accepted_codes)]
        return f"{spec.name} in {{{', '.join(labels)}}}"


SlotCut = ThresholdCut | CategoryCut


@dataclass(frozen=True, slots=True)
class CountAggregation:
    minimum_count: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.minimum_count, bool) or not isinstance(self.minimum_count, int):
            raise TypeError("minimum_count must be an integer")
        if self.minimum_count < 1:
            raise ValueError("minimum_count must be positive")

    def evaluate(self, mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
        if mask.ndim != 2:
            raise ValueError("A slot mask must have shape (n_samples, n_slots)")
        return np.count_nonzero(mask, axis=1) >= self.minimum_count

    def to_text(self) -> str:
        return "exists" if self.minimum_count == 1 else f"count >= {self.minimum_count}"
