"""Run the EDT with object-level and event-level inputs.

The input ``.pt`` files must contain the original ``obj``, ``label``, and
``obj_columns`` entries plus:

* ``event_features`` with shape ``(n_events, n_event_features)``; and
* ``event_feature_columns`` naming the columns of that tensor.

All event features are interpreted as non-negative integer-valued ordinal
observables.  This is the format produced by
``scripts/add_event_multiplicities.py``.  At every internal node, event-level
and object-level cuts compete using the same Gini-gain objective.

Run, for example::

    python casual_tests/stale/run_EDT_with_event_features.py \
        --signal data/with_event_multiplicities/anomalous.pt \
        --background data/with_event_multiplicities/reference.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from set_based_models import (  # noqa: E402
    ObservableSpec,
    gini_gain,
)


OBJECT_COLUMNS = ["pt", "eta", "phi", "type_id"]
OBSERVABLES = (
    ObservableSpec("pt", "continuous"),
    ObservableSpec("eta", "continuous"),
    ObservableSpec("phi", "continuous"),
    ObservableSpec(
        "type_id",
        "categorical",
        {0: "jet", 1: "muon", 2: "electron", 3: "photon", 4: "met"},
    ),
)


def load_dataset(path: str | Path):
    """Load one augmented dataset and return NumPy model inputs."""
    path = Path(path).expanduser().resolve()
    raw = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "obj",
        "label",
        "obj_columns",
        "event_features",
        "event_feature_columns",
    }
    if not isinstance(raw, dict) or not required <= raw.keys():
        missing = required - raw.keys() if isinstance(raw, dict) else required
        raise ValueError(f"{path} is missing required keys: {sorted(missing)}")
    if raw["obj_columns"] != OBJECT_COLUMNS:
        raise ValueError(f"Unexpected object columns in {path}: {raw['obj_columns']!r}")

    obj = raw["obj"]
    label = raw["label"]
    event_features = raw["event_features"]
    event_columns = raw["event_feature_columns"]
    if not isinstance(obj, torch.Tensor) or obj.ndim != 3 or obj.shape[1:] != (23, 4):
        shape = tuple(obj.shape) if isinstance(obj, torch.Tensor) else type(obj).__name__
        raise ValueError(f"Expected obj shape (events, 23, 4) in {path}, got {shape}")
    if not isinstance(label, torch.Tensor) or label.ndim != 1 or label.shape[0] != obj.shape[0]:
        raise ValueError(f"label must contain one entry per obj row in {path}.")
    if not isinstance(event_features, torch.Tensor) or event_features.ndim != 2:
        raise ValueError(f"event_features must be a two-dimensional tensor in {path}.")
    if event_features.shape[0] != obj.shape[0]:
        raise ValueError(f"event_features must contain one row per obj row in {path}.")
    if (
        not isinstance(event_columns, list)
        or not event_columns
        or not all(isinstance(name, str) and name.strip() for name in event_columns)
        or len(event_columns) != event_features.shape[1]
    ):
        raise ValueError(
            f"event_feature_columns must name every event_features column in {path}."
        )
    if len(set(event_columns)) != len(event_columns):
        raise ValueError(f"event_feature_columns contains duplicate names in {path}.")
    if not bool(torch.isfinite(event_features).all()):
        raise ValueError(f"event_features must contain only finite values in {path}.")
    if not bool((event_features == torch.floor(event_features)).all()):
        raise ValueError(f"event_features must contain integer-valued multiplicities in {path}.")
    if not bool((event_features >= 0).all()):
        raise ValueError(f"event_features must contain non-negative multiplicities in {path}.")

    obj = torch.nan_to_num(obj, nan=0.0, posinf=0.0, neginf=0.0)
    slot_mask = obj[..., 0] > 0
    return (
        obj.numpy(),
        slot_mask.numpy(),
        event_features.numpy(),
        label,
        tuple(event_columns),
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--signal",
        type=Path,
        default=ROOT / "data/with_event_multiplicities/anomalous.pt",
        help="Signal/mixture augmented .pt file.",
    )
    parser.add_argument(
        "--background",
        type=Path,
        default=ROOT / "data/with_event_multiplicities/reference.pt",
        help="Reference/background augmented .pt file.",
    )
    args = parser.parse_args(argv)
    from set_based_models import ExistentialDecisionTree

    signal_path = args.signal.expanduser().resolve()
    background_path = args.background.expanduser().resolve()

    try:
        signal, signal_slot_mask, signal_events, signal_labels, signal_event_columns = (
            load_dataset(signal_path)
        )
        (
            background,
            background_slot_mask,
            background_events,
            background_labels,
            background_event_columns,
        ) = load_dataset(background_path)
        if signal_event_columns != background_event_columns:
            raise ValueError(
                "Signal and background event_feature_columns must match exactly; "
                f"received {list(signal_event_columns)!r} and "
                f"{list(background_event_columns)!r}."
            )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    event_observables = tuple(
        ObservableSpec(name, "ordinal") for name in signal_event_columns
    )

    print(f"Signal input: {signal_path}")
    print(f"Background input: {background_path}")
    print(f"Event observables: {', '.join(signal_event_columns)}")
    print(
        f"Signal sample: {len(signal)} events; stored labels:",
        torch.unique(signal_labels, return_counts=True),
    )
    print(
        f"Reference background: {len(background)} events; stored labels:",
        torch.unique(background_labels, return_counts=True),
    )
    print("All rows in the signal input are treated as signal, regardless of stored process labels.")

    tree = ExistentialDecisionTree(
        OBSERVABLES,
        event_observables=event_observables,
        max_depth=6,
        gain_function=gini_gain,
        min_samples_leaf=20,
    )
    tree.fit(
        signal,
        background,
        signal_slot_mask=signal_slot_mask,
        background_slot_mask=background_slot_mask,
        signal_event_features=signal_events,
        background_event_features=background_events,
    )

    leaves = [node for node in tree.nodes_ if node.is_leaf]
    ranked = sorted(
        leaves,
        key=lambda node: node.score if node.score is not None else float("-inf"),
        reverse=True,
    )
    for selected_leaf in ranked:
        print(f"\nSelected leaf: {selected_leaf.node_id}")
        score = "n/a" if selected_leaf.score is None else f"{selected_leaf.score:.3f}"
        print(f"Signal score: {score}")
        print("\nExact root-to-leaf event path:")
        print(tree.cut_chain_text(selected_leaf.node_id))
        print("\nPositive-branch slot cut chain:")
        slot_chain = tree.slot_cut_chain(selected_leaf.node_id)
        if slot_chain:
            for cut in slot_chain:
                print(f"  {cut.to_text(tree.observables)}")
        else:
            print("  (no positive slot cuts)")


if __name__ == "__main__":
    main()
