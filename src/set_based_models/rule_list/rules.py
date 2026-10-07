"""Executable existential conjunction rules."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import ArrayLike
from ..core.data import ObservableSpec, SetInput, _validate_event_observables, _validate_validity_columns
from ..core.predicates import Cut, EventCut
from .statistics import RuleStatistics

@dataclass(frozen=True, slots=True)
class RuleStep:
    depth: int
    cut: Cut
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
    event_observables: tuple[ObservableSpec, ...] = ()

    def __post_init__(self):
        if self.rule_id < 0 or self.minimum_count < 1 or not self.steps:
            raise ValueError("A rule needs a non-negative ID, cuts, and a positive count")
        if tuple(s.depth for s in self.steps) != tuple(range(1, len(self.steps) + 1)):
            raise ValueError("Rule-step depths must be consecutive from one")
        object.__setattr__(self, "event_observables",
                           _validate_event_observables(self.event_observables))
        if not self.observables and not self.event_observables:
            raise ValueError("A rule requires object or event observables")
        _validate_validity_columns(self.observables)
        for step in self.steps:
            step.cut.validate(self.event_observables if isinstance(step.cut, EventCut)
                              else self.observables)
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
    def matches_slots(self, slots: ArrayLike, *, depth=None, event_features=None):
        """Evaluate each slot's cuts and event conditions, without counting.

        Event features may be a single vector or have leading dimensions
        matching the events in ``slots``; they are broadcast over slots.
        """
        if not self.observables:
            raise ValueError("Event-only rules have no object slots; use predict(event_features=...)")
        values = np.asarray(slots)
        if values.ndim < 2 or values.shape[-1] != len(self.observables):
            raise ValueError("slots must have shape (..., n_observables)")
        shape = values.shape[:-1]
        events = None
        if event_features is not None:
            events = np.asarray(event_features)
            if events.ndim < 1 or events.ndim > values.ndim:
                raise ValueError("event_features must align with the slots' leading dimensions")
            if events.shape[-1] != len(self.event_observables):
                raise ValueError("event_features has the wrong number of event observables")
            broadcast_shape = (events.shape[:-1]
                               + (1,) * (values.ndim - events.ndim)
                               + (events.shape[-1],))
            try:
                events = np.broadcast_to(events.reshape(broadcast_shape),
                                         shape + (events.shape[-1],))
            except ValueError as exc:
                raise ValueError("event_features must align with the slots' leading dimensions") from exc
            events = events.reshape(values.reshape(-1, values.shape[-1]).shape[0],
                                    len(self.event_observables))
        result = self.slot_mask(values.reshape(-1, 1, values.shape[-1]),
                                event_features=events, depth=depth)
        return result[:, 0].reshape(shape)
    def matches_slot(self, slot: ArrayLike, *, depth=None, event_features=None):
        values = np.asarray(slot)
        if values.shape != (len(self.observables),):
            raise ValueError(f"slot must have shape ({len(self.observables)},)")
        return bool(self.matches_slots(values[None, :], depth=depth,
                                       event_features=event_features)[0])

    def _evaluate(self, X, slot_mask, event_features, depth):
        cuts = self.cut_chain(depth)
        if X is None and any(not isinstance(cut, EventCut) for cut in cuts):
            raise ValueError("X is required to evaluate object-level cuts")
        data = SetInput.from_arrays(
            X, self.observables, slot_mask=slot_mask,
            event_features=event_features, event_observables=self.event_observables,
        )
        active = data.slot_mask.copy()
        selected = np.ones(data.n_samples, dtype=bool)
        has_slot_cut = False
        for cut in cuts:
            if isinstance(cut, EventCut):
                selected &= cut.evaluate(data)
            else:
                active &= cut.evaluate(data)
                has_slot_cut = True
        return selected, active, has_slot_cut or (not cuts and data.n_slots > 0)

    def slot_mask(self, X=None, *, slot_mask=None, depth=None, event_features=None):
        """Return slots satisfying slot cuts, gated by the event conditions.

        This does not apply ``minimum_count``. An event-only rule returns all
        valid slots of passing events, which may be empty; use ``predict``
        for the event-level result.
        """
        selected, active, _ = self._evaluate(X, slot_mask, event_features, depth)
        return active & selected[:, None]

    def predict(self, X=None, *, slot_mask=None, depth=None, event_features=None):
        """Select events; omit X when the evaluated cuts are all event-level."""
        selected, active, needs_slots = self._evaluate(X, slot_mask, event_features, depth)
        if needs_slots:
            selected &= np.count_nonzero(active, axis=1) >= self.minimum_count
        return selected

    def __call__(self, X=None, *, slot_mask=None, event_features=None):
        return self.predict(X, slot_mask=slot_mask, event_features=event_features)

    def to_text(self, *, depth=None):
        prefix = "EXISTS" if self.minimum_count == 1 else f"COUNT >= {self.minimum_count}"
        cuts = self.cut_chain(depth)
        if not cuts and not self.observables:
            return "TRUE"
        events = [c.to_text(self.event_observables) for c in cuts if isinstance(c, EventCut)]
        slots = [c.to_text(self.observables) for c in cuts if not isinstance(c, EventCut)]
        slot_text = f"{prefix} slot: {' AND '.join(slots) or 'TRUE'}"
        if not events:
            return slot_text
        if slots:
            events.append(f"({slot_text})")
        return " AND ".join(events)
