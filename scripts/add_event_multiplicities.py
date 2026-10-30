"""Add object multiplicities as an aligned event-level tensor to two .pt files.

Run from the repository root, for example:

  python scripts/add_event_multiplicities.py \
      --reference data/minbias_ggHgammagamma/reference.pt \
      --anomalous data/minbias_ggHgammagamma/anomalous.pt \
      --output-dir data/minbias_ggHgammagamma_with_multiplicities

Each output preserves the complete input dictionary and adds:

* ``event_features``: an ``[n_events, 4]`` int64 tensor containing electron,
  muon, photon, and jet multiplicities; and
* ``event_feature_columns``: names for those four columns.

An object exists when its ``pt`` value is finite and strictly positive, which
matches the slot-mask convention used by the EDT and ERL runners. Type IDs use
the production object schema: jet=0, muon=1, electron=2, photon=3, and MET=4.
MET is validated but is not included in the event tensor. Existing outputs and
inputs that already define either event-level key are rejected.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
EVENT_FEATURE_COLUMNS = ["n_electrons", "n_muons", "n_photons", "n_jets"]
TYPE_IDS = {
    "n_electrons": 2,
    "n_muons": 1,
    "n_photons": 3,
    "n_jets": 0,
}
KNOWN_TYPE_IDS = frozenset((*TYPE_IDS.values(), 4))
EVENT_KEYS = {"event_features", "event_feature_columns"}


def add_event_multiplicities(dataset: dict, *, source_name: str = "dataset") -> dict:
    """Return a shallow copy with multiplicities derived from ``dataset['obj']``."""
    if not isinstance(dataset, dict):
        raise ValueError(f"{source_name} must contain a dictionary.")
    missing = {"obj", "obj_columns", "label"} - dataset.keys()
    if missing:
        raise ValueError(f"{source_name} is missing required keys: {sorted(missing)}.")
    existing = EVENT_KEYS & dataset.keys()
    if existing:
        raise ValueError(
            f"{source_name} already contains event-level keys {sorted(existing)}; "
            "refusing to replace them."
        )

    obj = dataset["obj"]
    labels = dataset["label"]
    columns = dataset["obj_columns"]
    if not isinstance(obj, torch.Tensor) or obj.ndim != 3:
        raise ValueError(f"{source_name}: obj must be a three-dimensional tensor.")
    if obj.device.type != "cpu":
        raise ValueError(f"{source_name}: obj must be stored on CPU.")
    if not isinstance(labels, torch.Tensor) or labels.ndim != 1:
        raise ValueError(f"{source_name}: label must be a one-dimensional tensor.")
    if len(labels) != obj.shape[0]:
        raise ValueError(f"{source_name}: label must contain one entry per obj row.")
    if not isinstance(columns, list) or len(columns) != obj.shape[-1]:
        raise ValueError(f"{source_name}: obj_columns does not match the obj tensor.")
    if len(set(columns)) != len(columns):
        raise ValueError(f"{source_name}: obj_columns contains duplicate names.")
    for required_column in ("pt", "type_id"):
        if required_column not in columns:
            raise ValueError(f"{source_name}: obj_columns must contain {required_column!r}.")

    pt = obj[..., columns.index("pt")]
    type_id = obj[..., columns.index("type_id")]
    if not bool(torch.isfinite(pt).all()) or not bool(torch.isfinite(type_id).all()):
        raise ValueError(f"{source_name}: pt and type_id must contain only finite values.")

    valid_slots = pt > 0
    valid_types = type_id[valid_slots]
    if valid_types.numel():
        integral_types = valid_types == torch.round(valid_types)
        if not bool(integral_types.all()):
            raise ValueError(f"{source_name}: valid slots contain non-integer type_id values.")
        observed = {int(value) for value in torch.unique(valid_types).tolist()}
        unknown = observed - KNOWN_TYPE_IDS
        if unknown:
            raise ValueError(f"{source_name}: valid slots contain unknown type IDs {sorted(unknown)}.")

    multiplicities = torch.stack(
        [
            torch.count_nonzero(valid_slots & (type_id == TYPE_IDS[name]), dim=1)
            for name in EVENT_FEATURE_COLUMNS
        ],
        dim=1,
    ).to(dtype=torch.int64)

    result = dict(dataset)
    result["event_features"] = multiplicities
    result["event_feature_columns"] = EVENT_FEATURE_COLUMNS.copy()
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference", type=Path, default=ROOT / "data/reference.pt",
        help="Reference/background .pt input (default: repository data/reference.pt).",
    )
    parser.add_argument(
        "--anomalous", type=Path, default=ROOT / "data/anomalous.pt",
        help="Anomalous/mixture .pt input (default: repository data/anomalous.pt).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data/with_event_multiplicities",
        help="Directory for new reference.pt and anomalous.pt files.",
    )
    args = parser.parse_args(argv)

    inputs = [args.reference.expanduser().resolve(), args.anomalous.expanduser().resolve()]
    output_dir = args.output_dir.expanduser().resolve()
    outputs = [output_dir / "reference.pt", output_dir / "anomalous.pt"]
    if inputs[0] == inputs[1]:
        parser.error("Reference and anomalous inputs must be different files.")
    if any(source == target for source in inputs for target in outputs):
        parser.error("Output files must be different from both input files.")
    for path in outputs:
        if path.exists() or path.is_symlink():
            parser.error(f"Refusing to overwrite {path}; choose a new --output-dir.")

    try:
        datasets = [
            torch.load(path, map_location="cpu", weights_only=True) for path in inputs
        ]
        converted = [
            add_event_multiplicities(dataset, source_name=str(path))
            for path, dataset in zip(inputs, datasets)
        ]
        if converted[0]["obj_columns"] != converted[1]["obj_columns"]:
            raise ValueError("Reference and anomalous inputs have different object schemas.")
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    output_dir.mkdir(parents=True, exist_ok=True)
    for path, dataset in zip(outputs, converted):
        with path.open("xb") as handle:
            torch.save(dataset, handle)
        shape = tuple(dataset["event_features"].shape)
        print(f"Saved {path} with event_features shape {shape}")
    print(f"Columns: {EVENT_FEATURE_COLUMNS}")


if __name__ == "__main__":
    main()
