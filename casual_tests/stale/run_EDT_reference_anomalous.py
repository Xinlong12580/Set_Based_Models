"""Run with ``python casual_tests/stale/run_EDT_reference_anomalous.py``.

Fits the original EDT settings to all rows in the signal and background inputs.
Use --signal and --background to choose .pt files; explicit relative paths are
resolved from the shell's current directory. Defaults are the repository's
data/anomalous.pt and data/reference.pt, independent of the current directory.
"""

import argparse
from pathlib import Path
import sys

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from set_based_models import ObservableSpec, gini_gain  # noqa: E402


OBJECT_COLUMNS = ["pt", "eta", "phi", "type_id"]
OBSERVABLES = (
    ObservableSpec("pt", "continuous"),
    ObservableSpec("eta", "continuous"),
    ObservableSpec("phi", "continuous"),
    ObservableSpec("type_id", "categorical", {0: "jet", 1: "muon", 2: "electron", 3: "photon", 4: "met"}),
)


def load_obj_dataset(path: str | Path):
    path = Path(path).expanduser().resolve()
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict) or not {"obj", "label", "obj_columns"} <= raw.keys():
        raise ValueError(f"{path} must contain obj, label, and obj_columns.")
    if raw["obj_columns"] != OBJECT_COLUMNS:
        raise ValueError(f"Unexpected object columns in {path}: {raw['obj_columns']!r}")
    obj = raw["obj"]
    label = raw["label"]
    if obj.ndim != 3 or obj.shape[1:] != (23, 4):
        raise ValueError(f"Expected obj shape (events, 23, 4) in {path}, got {tuple(obj.shape)}")
    if label.ndim != 1 or label.shape[0] != obj.shape[0]:
        raise ValueError(f"label must contain one entry per obj row in {path}.")
    obj = torch.nan_to_num(obj, nan=0.0, posinf=0.0, neginf=0.0)
    slot_mask = obj[..., 0] > 0
    return obj.numpy(), slot_mask.numpy(), label


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--signal", type=Path, default=ROOT / "data/anomalous.pt",
        help="Signal/mixture .pt file (default: repository data/anomalous.pt).",
    )
    parser.add_argument(
        "--background", type=Path, default=ROOT / "data/reference.pt",
        help="Reference/background .pt file (default: repository data/reference.pt).",
    )
    args = parser.parse_args(argv)
    from set_based_models import ExistentialDecisionTree

    signal_path = args.signal.expanduser().resolve()
    background_path = args.background.expanduser().resolve()
    signal, signal_slot_mask, signal_labels = load_obj_dataset(signal_path)
    background, background_slot_mask, background_labels = load_obj_dataset(background_path)

    print(f"Signal input: {signal_path}")
    print(f"Background input: {background_path}")
    print(f"Signal sample: {len(signal)} events; stored labels:",
          torch.unique(signal_labels, return_counts=True))
    print(f"Reference background: {len(background)} events; stored labels:",
          torch.unique(background_labels, return_counts=True))
    print("All rows in the signal input are treated as signal, regardless of stored process labels.")

    tree = ExistentialDecisionTree(
        OBSERVABLES,
        max_depth=6,
        gain_function=gini_gain,
        min_samples_leaf=20,
    )
    tree.fit(
        signal,
        background,
        signal_slot_mask=signal_slot_mask,
        background_slot_mask=background_slot_mask,
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
