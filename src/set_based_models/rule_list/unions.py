"""Executable unions of fitted existential rules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..core.data import ObservableSpec, SetInput
from .rules import ExistentialRule


@dataclass(frozen=True, slots=True)
class RuleUnion:
    """Logical OR of zero or more independently aggregated rules."""

    observables: tuple[ObservableSpec, ...]
    rules: tuple[ExistentialRule, ...]

    def __post_init__(self) -> None:
        if not self.observables:
            raise ValueError("observables must not be empty")
        rule_ids = [rule.rule_id for rule in self.rules]
        if len(set(rule_ids)) != len(rule_ids):
            raise ValueError("A rule union cannot contain duplicate rule IDs")
        for rule in self.rules:
            if rule.observables != self.observables:
                raise ValueError("Every rule in a union must use the same observable schema")

    @property
    def rule_ids(self) -> tuple[int, ...]:
        return tuple(rule.rule_id for rule in self.rules)

    @property
    def n_rules(self) -> int:
        return len(self.rules)

    def matches_slot(self, slot: ArrayLike) -> bool:
        values = np.asarray(slot)
        if values.shape != (len(self.observables),):
            raise ValueError(f"slot must have shape ({len(self.observables)},)")
        return any(rule.matches_slot(values) for rule in self.rules)

    def matches_slots(self, slots: ArrayLike) -> NDArray[np.bool_]:
        values = np.asarray(slots)
        if values.ndim < 2 or values.shape[-1] != len(self.observables):
            raise ValueError("slots must have shape (..., n_observables)")
        result = np.zeros(values.shape[:-1], dtype=np.bool_)
        for rule in self.rules:
            result |= rule.matches_slots(values)
        return result

    def slot_mask(
        self, X: ArrayLike, *, slot_mask: ArrayLike | None = None
    ) -> NDArray[np.bool_]:
        data = SetInput.from_arrays(X, self.observables, slot_mask=slot_mask)
        result = np.zeros(data.slot_mask.shape, dtype=np.bool_)
        for rule in self.rules:
            result |= rule.slot_mask(data.X, slot_mask=data.slot_mask)
        return result

    def predict(
        self, X: ArrayLike, *, slot_mask: ArrayLike | None = None
    ) -> NDArray[np.bool_]:
        data = SetInput.from_arrays(X, self.observables, slot_mask=slot_mask)
        result = np.zeros(data.n_samples, dtype=np.bool_)
        # Aggregate each rule independently before applying OR. Counting the
        # combined slot mask would be incorrect when minimum_count > 1.
        for rule in self.rules:
            result |= rule.predict(data.X, slot_mask=data.slot_mask)
        return result

    def __call__(
        self, X: ArrayLike, *, slot_mask: ArrayLike | None = None
    ) -> NDArray[np.bool_]:
        return self.predict(X, slot_mask=slot_mask)

    def to_text(self) -> str:
        if not self.rules:
            return "FALSE"
        return "\nOR\n".join(
            f"Rule {rule.rule_id}: {rule.to_text()}" for rule in self.rules
        )
