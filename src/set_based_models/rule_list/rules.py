"""Executable existential conjunction rules."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import ArrayLike
from ..core.data import ObservableSpec, SetInput
from ..core.predicates import SlotCut
from .statistics import RuleStatistics

@dataclass(frozen=True, slots=True)
class RuleStep:
    depth: int
    cut: SlotCut
    statistics: RuleStatistics
    generation_score: float

@dataclass(frozen=True, slots=True)
class ExistentialRule:
    rule_id: int
    observables: tuple[ObservableSpec, ...]
    steps: tuple[RuleStep, ...]
    minimum_count: int = 1
    signal_selection: np.ndarray | None = field(default=None, repr=False, compare=False)
    background_selection: np.ndarray | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self.rule_id < 0 or self.minimum_count < 1 or not self.steps:
            raise ValueError("A rule needs a non-negative ID, cuts, and a positive count")
        if tuple(s.depth for s in self.steps) != tuple(range(1, len(self.steps) + 1)):
            raise ValueError("Rule-step depths must be consecutive from one")
        for name in ("signal_selection", "background_selection"):
            selection = getattr(self, name)
            if selection is None:
                continue
            selection = np.asarray(selection, dtype=np.bool_)
            if selection.ndim != 1:
                raise ValueError(f"{name} must be a one-dimensional event mask")
            selection = selection.copy()
            selection.flags.writeable = False
            object.__setattr__(self, name, selection)
    @property
    def depth(self): return len(self.steps)
    @property
    def cuts(self): return tuple(step.cut for step in self.steps)
    @property
    def statistics(self): return self.steps[-1].statistics
    @property
    def generation_score(self): return self.steps[-1].generation_score
    def cut_chain(self, depth=None):
        depth = self.depth if depth is None else depth
        if not 0 <= depth <= self.depth: raise ValueError("Invalid rule depth")
        return self.cuts[:depth]
    def matches_slots(self, slots: ArrayLike, *, depth=None):
        values = np.asarray(slots)
        if values.ndim < 2 or values.shape[-1] != len(self.observables):
            raise ValueError("slots must have shape (..., n_observables)")
        shape = values.shape[:-1]
        data = SetInput.from_arrays(values.reshape(-1, 1, values.shape[-1]), self.observables)
        result = data.slot_mask.copy()
        for cut in self.cut_chain(depth): result &= cut.evaluate(data)
        return result[:, 0].reshape(shape)
    def matches_slot(self, slot: ArrayLike, *, depth=None):
        values = np.asarray(slot)
        if values.shape != (len(self.observables),):
            raise ValueError(f"slot must have shape ({len(self.observables)},)")
        return bool(self.matches_slots(values[None, :], depth=depth)[0])
    def slot_mask(self, X, *, slot_mask=None, depth=None):
        data = SetInput.from_arrays(X, self.observables, slot_mask=slot_mask)
        result = data.slot_mask.copy()
        for cut in self.cut_chain(depth): result &= cut.evaluate(data)
        return result
    def predict(self, X, *, slot_mask=None, depth=None):
        return np.count_nonzero(self.slot_mask(X, slot_mask=slot_mask, depth=depth), axis=1) >= self.minimum_count
    def __call__(self, X, *, slot_mask=None): return self.predict(X, slot_mask=slot_mask)
    def to_text(self, *, depth=None):
        prefix = "EXISTS" if self.minimum_count == 1 else f"COUNT >= {self.minimum_count}"
        cuts = " AND ".join(c.to_text(self.observables) for c in self.cut_chain(depth))
        return f"{prefix} slot: {cuts or 'TRUE'}"
