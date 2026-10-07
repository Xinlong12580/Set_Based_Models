"""Candidate predicates shared by the tree and rule search."""

from __future__ import annotations

import numpy as np

from .data import ObservableSpec, SetInput
from .predicates import CategoryCut, Comparison, EventCut, ThresholdCut


def observable_cuts(index: int, spec: ObservableSpec, values, max_thresholds):
    if not spec.allows_threshold:
        assert spec.allowed_codes is not None
        return tuple(CategoryCut(index, {code}) for code in sorted(spec.allowed_codes))
    unique = np.unique(values)
    if unique.size < 2:
        return ()
    thresholds = unique[:-1] + np.diff(unique) / 2.0
    if max_thresholds is not None and thresholds.size > max_thresholds:
        positions = np.linspace(0, thresholds.size - 1, max_thresholds, dtype=int)
        thresholds = thresholds[np.unique(positions)]
    return tuple(
        ThresholdCut(index, comparison, float(threshold))
        for threshold in thresholds
        for comparison in (Comparison.GREATER, Comparison.LESS_EQUAL)
    )


def event_candidate_cuts(signal: SetInput, background: SetInput,
                         signal_rows, background_rows, max_thresholds, *, excluded_columns=()):
    """Generate cuts using only the events currently eligible for extension."""
    result = []
    for index, spec in enumerate(signal.event_observables):
        if spec.name in excluded_columns:
            continue
        assert signal.event_features is not None and background.event_features is not None
        arrays = []
        for data, rows in ((signal, signal_rows), (background, background_rows)):
            observed = data.event_features[rows, index]
            if data.event_value_mask is not None:
                observed = observed[data.event_value_mask[rows, index]]
            arrays.append(observed)
        values = np.concatenate(arrays)
        result.extend(EventCut(cut) for cut in observable_cuts(
            index, spec, values, max_thresholds
        ))
    return tuple(result)
