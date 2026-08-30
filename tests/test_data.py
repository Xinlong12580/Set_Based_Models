import numpy as np
import pytest

from set_based_dt.data import ObservableKind, ObservableSpec, SetInput


SPECS = (
    ObservableSpec("pt", "continuous"),
    ObservableSpec("n_tracks", "ordinal"),
    ObservableSpec("flavor", "categorical", {0: "light", 1: "bottom"}),
    ObservableSpec("tagged", "boolean"),
)


def test_constructs_canonical_float64_input() -> None:
    X = np.array([[[40, 3, 0, 0], [80, 5, 1, 1]]], dtype=np.float32)
    data = SetInput.from_arrays(X, SPECS, y=[1])

    assert data.X.dtype == np.float64
    assert data.X.flags.c_contiguous
    assert data.X.shape == (1, 2, 4)
    assert data.slot_mask.all()
    assert data.value_mask.all()
    assert data.observables[2].allows_membership
    assert not data.observables[2].allows_threshold


def test_rejects_fractional_discrete_value() -> None:
    X = np.array([[[40, 3, 0.5, 0]]])

    with pytest.raises(ValueError, match="exact integer-valued"):
        SetInput.from_arrays(X, SPECS)


def test_rejects_unknown_category_and_boolean_codes() -> None:
    X = np.array([[[40, 3, 2, 0]]])
    with pytest.raises(ValueError, match="unknown codes"):
        SetInput.from_arrays(X, SPECS)

    X = np.array([[[40, 3, 0, 2]]])
    with pytest.raises(ValueError, match="unknown codes"):
        SetInput.from_arrays(X, SPECS)


def test_ignores_values_in_padded_slots() -> None:
    X = np.array([[[40, 3, 0, 1], [np.nan, np.nan, np.nan, np.nan]]])
    data = SetInput.from_arrays(X, SPECS, slot_mask=np.array([[True, False]]))

    assert data.slot_mask.tolist() == [[True, False]]
    assert not data.value_mask[0, 1].any()


def test_value_mask_distinguishes_missing_measurement_from_padding() -> None:
    X = np.array([[[40, 3, np.nan, 1]]])
    value_mask = np.ones_like(X, dtype=bool)
    value_mask[0, 0, 2] = False

    data = SetInput.from_arrays(X, SPECS, value_mask=value_mask)
    assert data.slot_mask[0, 0]
    assert not data.value_mask[0, 0, 2]


def test_requires_boolean_masks() -> None:
    X = np.zeros((1, 1, 4))
    with pytest.raises(TypeError, match="Boolean dtype"):
        SetInput.from_arrays(X, SPECS, slot_mask=[[1]])


def test_category_codes_must_be_compact() -> None:
    with pytest.raises(ValueError, match="consecutive integers"):
        ObservableSpec("flavor", ObservableKind.CATEGORICAL, {1: "light", 2: "bottom"})


def test_validates_labels_and_weights() -> None:
    X = np.zeros((2, 1, 4))
    data = SetInput.from_arrays(X, SPECS, y=[0, 1], sample_weight=[1.0, 2.0])
    assert data.y is not None and data.y.dtype == np.int64

    with pytest.raises(ValueError, match="binary labels"):
        SetInput.from_arrays(X, SPECS, y=[0, 2])
    with pytest.raises(ValueError, match="non-negative"):
        SetInput.from_arrays(X, SPECS, sample_weight=[1, -1])
