"""Input definitions and validation for set-based decision trees.

The canonical input is a dense, three-dimensional NumPy array with axes
``(sample, slot, observable)``.  All values are stored as ``float64``;
``ObservableSpec`` records whether each observable is continuous, ordinal,
categorical, or boolean so that the tree can generate valid predicates.

Padding and missing measurements have distinct representations:

* ``slot_mask[i, j]`` says whether slot ``j`` exists in sample ``i``.
* ``value_mask[i, j, k]`` says whether observable ``k`` was measured for an
  existing slot.  If omitted, every value in every valid slot must be finite.

Values in padded slots and explicitly missing measurements are ignored and
need not be finite.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


class ObservableKind(str, Enum):
    """The predicates that may be generated for an observable."""

    CONTINUOUS = "continuous"
    ORDINAL = "ordinal"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"


@dataclass(frozen=True, slots=True)
class ObservableSpec:
    """Meaning and allowed values of one column on the observable axis.

    Parameters
    ----------
    name:
        Unique human-readable name used when exporting tree rules.
    kind:
        Continuous and ordinal observables permit threshold predicates.
        Categorical observables permit equality/membership predicates.
        Boolean observables permit true/false predicates.
    categories:
        For categorical observables, a non-empty mapping from integer codes to
        display labels. Codes must be compact non-negative integers beginning
        at zero. It must be omitted for all other kinds.
    """

    name: str
    kind: ObservableKind | str
    categories: Mapping[int, str] | None = None

    def __post_init__(self) -> None:
        name = self.name.strip()
        if not name:
            raise ValueError("Observable name must not be empty")
        object.__setattr__(self, "name", name)

        try:
            kind = ObservableKind(self.kind)
        except ValueError as exc:
            allowed = ", ".join(kind.value for kind in ObservableKind)
            raise ValueError(f"Unknown observable kind {self.kind!r}; expected {allowed}") from exc
        object.__setattr__(self, "kind", kind)

        if kind is ObservableKind.CATEGORICAL:
            if not self.categories:
                raise ValueError(
                    f"Categorical observable {name!r} requires a non-empty categories mapping"
                )
            categories = dict(self.categories)
            if any(isinstance(code, bool) or not isinstance(code, int) for code in categories):
                raise TypeError("Category codes must be integers, not booleans or floats")
            expected = set(range(len(categories)))
            if set(categories) != expected:
                raise ValueError(
                    f"Category codes for {name!r} must be consecutive integers from 0; "
                    f"received {sorted(categories)}"
                )
            if any(not isinstance(label, str) or not label for label in categories.values()):
                raise ValueError("Category labels must be non-empty strings")
            object.__setattr__(self, "categories", MappingProxyType(categories))
        elif self.categories is not None:
            raise ValueError("categories may only be supplied for categorical observables")

    @property
    def allows_threshold(self) -> bool:
        """Whether tree nodes may compare this observable to a threshold."""

        return self.kind in (ObservableKind.CONTINUOUS, ObservableKind.ORDINAL)

    @property
    def allows_membership(self) -> bool:
        """Whether tree nodes may use equality or set-membership predicates."""

        return self.kind in (ObservableKind.CATEGORICAL, ObservableKind.BOOLEAN)

    @property
    def allowed_codes(self) -> frozenset[int] | None:
        """Return valid discrete codes, or ``None`` for non-discrete data."""

        if self.kind is ObservableKind.BOOLEAN:
            return frozenset((0, 1))
        if self.kind is ObservableKind.CATEGORICAL:
            assert self.categories is not None
            return frozenset(self.categories)
        return None


@dataclass(frozen=True, slots=True)
class SetInput:
    """Validated inputs for fitting or evaluating a set-based tree.

    Construction converts ``X`` to a C-contiguous ``float64`` array and masks
    to Boolean arrays. Validation is intentionally strict: discrete values must
    already be exact integer codes and are never rounded implicitly.

    ``y`` is optional so the same class can represent prediction input. When
    present, it must contain binary labels encoded as 0 and 1. Sample weights
    must be finite and non-negative.
    """

    X: NDArray[np.float64]
    observables: tuple[ObservableSpec, ...]
    slot_mask: NDArray[np.bool_]
    value_mask: NDArray[np.bool_]
    y: NDArray[np.int64] | None = None
    sample_weight: NDArray[np.float64] | None = None

    @classmethod
    def from_arrays(
        cls,
        X: ArrayLike,
        observables: Sequence[ObservableSpec],
        *,
        y: ArrayLike | None = None,
        slot_mask: ArrayLike | None = None,
        value_mask: ArrayLike | None = None,
        sample_weight: ArrayLike | None = None,
    ) -> SetInput:
        """Convert and validate array-like inputs.

        If ``slot_mask`` is omitted, all slots are considered real. If
        ``value_mask`` is omitted, all observables of real slots are considered
        present. A supplied value mask cannot mark a value present in a padded
        slot.
        """

        try:
            x_array = np.asarray(X, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise TypeError("X must be convertible to a numeric float64 array") from exc
        if x_array.ndim != 3:
            raise ValueError(
                "X must have shape (n_samples, n_slots, n_observables); "
                f"received {x_array.shape}"
            )
        n_samples, n_slots, n_observables = x_array.shape
        if n_slots == 0 or n_observables == 0:
            raise ValueError("X must contain at least one slot and one observable")

        specs = tuple(observables)
        if len(specs) != n_observables:
            raise ValueError(
                f"Expected {n_observables} observable specifications, received {len(specs)}"
            )
        if not all(isinstance(spec, ObservableSpec) for spec in specs):
            raise TypeError("Every observable definition must be an ObservableSpec")
        names = [spec.name for spec in specs]
        if len(set(names)) != len(names):
            raise ValueError("Observable names must be unique")

        if slot_mask is None:
            slots = np.ones((n_samples, n_slots), dtype=np.bool_)
        else:
            slots = _as_bool_mask(slot_mask, (n_samples, n_slots), "slot_mask")

        if value_mask is None:
            values = np.broadcast_to(slots[..., None], x_array.shape).copy()
        else:
            values = _as_bool_mask(value_mask, x_array.shape, "value_mask")
            if np.any(values & ~slots[..., None]):
                raise ValueError("value_mask cannot mark values present in padded slots")

        if np.any(~np.isfinite(x_array[values])):
            raise ValueError("Present values in valid slots must be finite")

        for index, spec in enumerate(specs):
            observed = x_array[..., index][values[..., index]]
            _validate_observable_values(observed, spec)

        labels = None if y is None else _validate_labels(y, n_samples)
        weights = (
            None
            if sample_weight is None
            else _validate_sample_weight(sample_weight, n_samples)
        )

        return cls(
            X=np.ascontiguousarray(x_array),
            observables=specs,
            slot_mask=np.ascontiguousarray(slots),
            value_mask=np.ascontiguousarray(values),
            y=labels,
            sample_weight=weights,
        )

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_slots(self) -> int:
        return self.X.shape[1]

    @property
    def n_observables(self) -> int:
        return self.X.shape[2]

    def observable_index(self, name: str) -> int:
        """Return an observable's third-axis index by name."""

        for index, spec in enumerate(self.observables):
            if spec.name == name:
                return index
        raise KeyError(f"Unknown observable {name!r}")


def _as_bool_mask(value: ArrayLike, shape: tuple[int, ...], name: str) -> NDArray[np.bool_]:
    raw = np.asarray(value)
    if raw.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; received {raw.shape}")
    if raw.dtype.kind != "b":
        raise TypeError(f"{name} must have Boolean dtype")
    return np.asarray(raw, dtype=np.bool_)


def _validate_observable_values(values: NDArray[np.float64], spec: ObservableSpec) -> None:
    if spec.kind is ObservableKind.CONTINUOUS:
        return

    if np.any(values != np.floor(values)):
        raise ValueError(f"Observable {spec.name!r} requires exact integer-valued data")

    codes = values.astype(np.int64)
    if spec.kind is ObservableKind.ORDINAL:
        return

    allowed = spec.allowed_codes
    assert allowed is not None
    invalid = sorted(set(np.unique(codes).tolist()) - allowed)
    if invalid:
        raise ValueError(
            f"Observable {spec.name!r} contains unknown codes {invalid}; "
            f"allowed codes are {sorted(allowed)}"
        )


def _validate_labels(value: ArrayLike, n_samples: int) -> NDArray[np.int64]:
    raw = np.asarray(value)
    if raw.shape != (n_samples,):
        raise ValueError(f"y must have shape ({n_samples},); received {raw.shape}")
    if raw.dtype.kind not in "biuf" or np.any(~np.isfinite(raw)):
        raise TypeError("y must contain finite numeric or Boolean binary labels")
    if np.any((raw != 0) & (raw != 1)):
        raise ValueError("y must contain only binary labels 0 and 1")
    return np.ascontiguousarray(raw, dtype=np.int64)


def _validate_sample_weight(value: ArrayLike, n_samples: int) -> NDArray[np.float64]:
    try:
        weights = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("sample_weight must be numeric") from exc
    if weights.shape != (n_samples,):
        raise ValueError(
            f"sample_weight must have shape ({n_samples},); received {weights.shape}"
        )
    if np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("sample_weight must be finite and non-negative")
    return np.ascontiguousarray(weights)

