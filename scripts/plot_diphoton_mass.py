"""Plot stored diphoton masses from a converted mixture PT file.

Example:
    python scripts/plot_diphoton_mass.py data/ggHgammagamma_mixture/anomalous.pt \
        --output reports/ggHgammagamma_diphoton_mass.png

Requires pair_obj (conversion with --with-pair-tensor). By default every valid
photon pair contributes, as in ERL; events with >2 photons can contribute more
than once. --pairs leading uses only the two highest-PT stored photons per
event. Stored process labels are overlaid without guessing their meaning.
"""

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def _photon_code(sample, name):
    schema = sample.get("pair_obj_feature_schema")
    if schema is None:
        return 3  # Standard converter type IDs: jet, muon, electron, photon.
    if not isinstance(schema, list) or not all(isinstance(spec, dict) for spec in schema):
        raise ValueError("pair_obj_feature_schema must be a list of column definitions.")
    for spec in schema:
        if spec.get("name") == name:
            categories = spec.get("categories", {})
            if not isinstance(categories, dict):
                raise ValueError(f"The pair schema must identify photon category codes in {name}.")
            for code, label in categories.items():
                if str(label).lower() in {"photon", "photons"}:
                    return int(code)
    raise ValueError(f"The pair schema must identify photons in {name}.")


def diphoton_masses(sample, *, pairs="all", leading_pt_min=0,
                    subleading_pt_min=0, sum_pt_min=0):
    """Return (masses in GeV, source event indices), excluding padding/invalids.

    Leading-pair selection precedes mass validity and PT cuts: an invalid
    leading pair does not fall back to a different pair in that event.
    """
    if not isinstance(sample, dict):
        raise ValueError("Input must be a converted PT dictionary.")
    tensor = sample.get("pair_obj")
    if (not isinstance(tensor, torch.Tensor) or tensor.ndim != 3
            or tensor.layout != torch.strided or tensor.is_complex()
            or any(size == 0 for size in tensor.shape)):
        raise ValueError("A nonempty pair_obj tensor is required; convert with --with-pair-tensor.")
    columns = sample.get("pair_obj_columns")
    if (not isinstance(columns, list) or len(columns) != tensor.shape[-1]
            or not all(isinstance(name, str) for name in columns)
            or len(set(columns)) != len(columns)):
        raise ValueError("pair_obj_columns must uniquely name every pair_obj column.")
    required = {"pt_1", "pt_2", "type_id_1", "type_id_2", "pair_mass", "pair_mass_valid"}
    if not required.issubset(columns):
        raise ValueError(f"Missing pair columns: {sorted(required - set(columns))}")
    mask = sample.get("pair_slot_mask")
    if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool
            or mask.layout != torch.strided or mask.shape != tensor.shape[:2]):
        raise ValueError("pair_slot_mask must be a Boolean tensor aligned with pair_obj.")
    if pairs not in {"all", "leading"}:
        raise ValueError("pairs must be 'all' or 'leading'.")
    if any(not np.isfinite(value) or value < 0
           for value in (leading_pt_min, subleading_pt_min, sum_pt_min)):
        raise ValueError("PT thresholds must be finite and nonnegative.")

    values = tensor.detach().cpu().numpy()
    indices = {name: columns.index(name) for name in required}
    pt1 = values[..., indices["pt_1"]].astype(np.float64)
    pt2 = values[..., indices["pt_2"]].astype(np.float64)
    selected = (mask.cpu().numpy()
                & (values[..., indices["type_id_1"]] == _photon_code(sample, "type_id_1"))
                & (values[..., indices["type_id_2"]] == _photon_code(sample, "type_id_2"))
                & np.isfinite(pt1) & np.isfinite(pt2) & (pt1 > 0) & (pt2 > 0))
    if pairs == "leading":
        # The two highest-PT photons maximize the pair's scalar PT sum.
        row = np.flatnonzero(selected.any(axis=1))
        slot = np.argmax(np.where(selected[row], pt1[row] + pt2[row], -np.inf), axis=1)
        selected = np.zeros_like(selected)
        selected[row, slot] = True

    mass = values[..., indices["pair_mass"]]
    selected &= ((values[..., indices["pair_mass_valid"]] == 1)
                 & np.isfinite(mass) & (mass >= 0)
                 & (np.maximum(pt1, pt2) > leading_pt_min)
                 & (np.minimum(pt1, pt2) > subleading_pt_min)
                 & (pt1 + pt2 > sum_pt_min))
    row, slot = np.nonzero(selected)
    return mass[row, slot].astype(np.float64), row


def _nonnegative_float(text):
    value = float(text)
    if not np.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="Converted mixture PT file containing pair_obj.")
    parser.add_argument("--output", type=Path, help="PNG/PDF/SVG output; default: beside the input PT file.")
    parser.add_argument("--pairs", choices=("all", "leading"), default="all")
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--mass-range", type=float, nargs=2, default=(0, 250), metavar=("LOW", "HIGH"))
    parser.add_argument("--leading-pt-min", type=_nonnegative_float, default=0)
    parser.add_argument("--subleading-pt-min", type=_nonnegative_float, default=0)
    parser.add_argument("--sum-pt-min", type=_nonnegative_float, default=0)
    parser.add_argument("--log-y", action="store_true")
    args = parser.parse_args(argv)
    if args.bins < 1:
        parser.error("--bins must be positive.")
    low, high = args.mass_range
    if not np.isfinite([low, high]).all() or low < 0 or high <= low:
        parser.error("--mass-range requires finite values with 0 <= LOW < HIGH.")
    output = args.output or args.input.with_suffix(".diphoton_mass.png")
    if output.suffix.lower() not in {".png", ".pdf", ".svg"}:
        parser.error("--output must end in .png, .pdf, or .svg.")
    try:
        # Memory mapping avoids loading unrelated PF/event/object tensors into RAM.
        sample = torch.load(args.input, map_location="cpu", weights_only=True, mmap=True)
        masses, rows = diphoton_masses(sample, pairs=args.pairs,
                                      leading_pt_min=args.leading_pt_min,
                                      subleading_pt_min=args.subleading_pt_min,
                                      sum_pt_min=args.sum_pt_min)
        labels = sample.get("label")
        if labels is not None:
            if (not isinstance(labels, torch.Tensor) or labels.ndim != 1
                    or labels.layout != torch.strided
                    or labels.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}
                    or len(labels) != sample["pair_obj"].shape[0]):
                raise ValueError("label must be a one-dimensional integer tensor aligned with events.")
            labels = labels.cpu().numpy()
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))
    if not len(masses):
        parser.error("No valid diphoton pairs pass the selection.")

    groups = [] if labels is None else [(f"Stored label {label}", masses[labels[rows] == label])
                                       for label in np.unique(labels[rows])]
    bins = np.linspace(low, high, args.bins + 1)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(masses, bins=bins, histtype="stepfilled", color="0.7", alpha=.45,
            label=f"Mixture ({len(masses)} entries)")
    for name, group in groups:
        ax.hist(group, bins=bins, histtype="step", linewidth=1.8,
                label=f"{name} ({len(group)} entries)")
    ax.set(xlabel=r"Diphoton invariant mass $m_{\gamma\gamma}$ [GeV]",
           ylabel=f"{'Pair' if args.pairs == 'all' else 'Event'} entries per {bins[1] - bins[0]:g} GeV",
           title=f"{args.input.parent.name}/{args.input.name}: {args.pairs} photon pairs",
           xlim=(low, high))
    if args.log_y:
        ax.set_yscale("log")
        ax.set_ylim(bottom=.5)
    else:
        ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        print(f"Warning: overwriting {output}", file=sys.stderr)
    fig.savefig(output, dpi=180)
    plt.close(fig)

    print(f"Saved {output}")
    print(f"Selected {len(masses)} pairs from {len(np.unique(rows))} / {sample['pair_obj'].shape[0]} events.")
    print(f"Cuts: leading PT > {args.leading_pt_min:g}, subleading PT > {args.subleading_pt_min:g}, "
          f"sum PT > {args.sum_pt_min:g} GeV; pair_mass_valid=True.")
    for name, group in [("Total", masses), *groups]:
        print(f"{name}: {len(group)} entries; below range={np.count_nonzero(group < low)}; "
              f"above range={np.count_nonzero(group > high)}.")
    if args.pairs == "all":
        print("All-pair counts can include multiple pairs per event; labels are diagnostic only.")


if __name__ == "__main__":
    main()
