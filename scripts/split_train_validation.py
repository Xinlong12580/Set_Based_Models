"""Split reference.pt and anomalous.pt into disjoint, shuffled train/validation files.

Example: python scripts/split_train_validation.py --input-dir data/mixture --train-fraction 0.8
Separate fractions: --reference-train-fraction 0.5 --anomalous-train-fraction 0.8
All tensors in these mixture files are event-aligned; other metadata is preserved.
"""

import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--train-fraction", type=float, default=0.8, help="Default training fraction for both datasets (default: 0.8).")
    parser.add_argument("--reference-train-fraction", type=float, help="Override the reference training fraction.")
    parser.add_argument("--anomalous-train-fraction", type=float, help="Override the anomalous training fraction.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generator = torch.Generator().manual_seed(args.seed)

    for name in ("reference", "anomalous"):
        sample = torch.load(args.input_dir / f"{name}.pt", map_location="cpu", weights_only=True)
        count = len(sample["label"])
        order = torch.randperm(count, generator=generator)
        fraction = getattr(args, f"{name}_train_fraction")
        split = int(count * (args.train_fraction if fraction is None else fraction))
        for part, indices in (("train", order[:split]), ("validation", order[split:])):
            subset = {key: value.index_select(0, indices) if isinstance(value, torch.Tensor) else value
                      for key, value in sample.items()}
            output = args.input_dir / f"{name}_{part}.pt"
            if output.exists():
                print(f"Warning: overwriting {output}")
            torch.save(subset, output)
            print(f"Saved {len(indices)} events to {output}")


if __name__ == "__main__":
    main()
