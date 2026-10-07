"""Mix any compatible background and signal .pt datasets (requires PyTorch).

Example, run from the repository root:
  python scripts/make_reference_anomalous.py \\
      --background data/minbias_10000.pt --signal data/ggHgammagamma.pt \\
      --output-dir data/ggHgammagamma_mixture

By default, reference.pt contains 2,000 background events; anomalous.pt
contains 1,000 different background events and 100 signal events, shuffled.
Sampling is without replacement. Disjoint rows avoid background event reuse;
statistical independence also assumes the source events were generated
independently and that the two input files do not share events.

Use --background and --signal for the input paths. The old --minbias and --zz
options remain aliases. Default paths still point to data/minbias_10000.pt
and data/ZZ_leptonic.pt, resolved relative to this repository.

Inputs must have matching keys, tensor shapes after the event dimension,
dtypes, and non-tensor metadata (including pf_columns and obj_columns).
All aligned event tensors and original process labels are preserved; labels
are not used to select events or infer which input is signal. Added
``source_id`` (0=background input, 1=signal input) and ``source_index`` identify
each output event's input .pt row, even when the inputs share process labels.
``source_files`` lists the two input paths in that order; ``sampling_seed``
records the seed. Source indices do not refer to raw Parquet rows.
Existing outputs are overwritten by default, with a warning for each file.
Use --no-overwrite to refuse replacement. Input files and output symlinks
are protected; both outputs are serialized before replacing existing files.
"""

import argparse
import os
from pathlib import Path
import tempfile
import warnings

import torch


ROOT = Path(__file__).resolve().parents[1]
PROVENANCE_KEYS = {"source_id", "source_index", "source_files", "sampling_seed"}


def make_datasets(background, signal, *, reference_count=2000, background_count=1000,
                  signal_count=100, seed=42):
    """Return aligned datasets with disjoint background rows and unchanged labels."""
    for count in (reference_count, background_count, signal_count):
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError("All event counts must be positive integers.")
    if not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63).")

    required = {"pf", "obj", "label", "pf_columns", "obj_columns"}
    for name, sample in (("Background", background), ("Signal", signal)):
        if not isinstance(sample, dict) or not required <= sample.keys():
            raise ValueError(f"{name} must contain {sorted(required)}.")
        if PROVENANCE_KEYS & sample.keys():
            raise ValueError(f"{name} already contains reserved provenance keys.")
        labels = sample["label"]
        if not isinstance(labels, torch.Tensor) or labels.ndim != 1:
            raise ValueError(f"{name}: label must be a one-dimensional tensor.")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                if value.ndim == 0 or value.shape[0] != len(labels):
                    raise ValueError(f"{name}: {key} is not an aligned event tensor.")
                if value.device.type != "cpu":
                    raise ValueError(f"{name}: {key} must be on CPU.")
        for key in ("pf", "obj"):
            tensor = sample[key]
            columns = sample[key + "_columns"]
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
                raise ValueError(f"{name}: {key} must be a three-dimensional tensor.")
            if not isinstance(columns, list) or len(columns) != tensor.shape[-1]:
                raise ValueError(f"{name}: {key}_columns does not match {key}.")

    if background.keys() != signal.keys():
        raise ValueError("The input dictionaries must have matching keys.")
    tensor_keys = []
    metadata = {}
    for key, value in background.items():
        other = signal[key]
        if isinstance(value, torch.Tensor):
            if (not isinstance(other, torch.Tensor)
                    or value.shape[1:] != other.shape[1:]
                    or value.dtype != other.dtype):
                raise ValueError(f"Input tensor shapes/dtypes differ for {key}.")
            tensor_keys.append(key)
        else:
            if isinstance(other, torch.Tensor) or value != other:
                raise ValueError(f"Input metadata differs for {key}.")
            metadata[key] = value

    if len(background["label"]) < reference_count + background_count:
        raise ValueError("Not enough background events for disjoint samples.")
    if len(signal["label"]) < signal_count:
        raise ValueError("Not enough signal events.")

    rng = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(len(background["label"]), generator=rng)
    reference_indices = order[:reference_count]
    background_indices = order[reference_count:reference_count + background_count]
    signal_indices = torch.randperm(len(signal["label"]), generator=rng)[:signal_count]
    mixture_order = torch.randperm(background_count + signal_count, generator=rng)

    reference = {**metadata, "sampling_seed": seed}
    anomalous = {**metadata, "sampling_seed": seed}
    for key in tensor_keys:
        reference[key] = background[key].index_select(0, reference_indices)
        mixture = torch.cat((background[key].index_select(0, background_indices),
                             signal[key].index_select(0, signal_indices)), dim=0)
        anomalous[key] = mixture.index_select(0, mixture_order)
    reference["source_id"] = torch.zeros(reference_count, dtype=torch.int64)
    reference["source_index"] = reference_indices.clone()
    anomalous["source_id"] = torch.cat((
        torch.zeros(background_count, dtype=torch.int64),
        torch.ones(signal_count, dtype=torch.int64),
    )).index_select(0, mixture_order)
    anomalous["source_index"] = torch.cat((background_indices, signal_indices)).index_select(0, mixture_order)
    return reference, anomalous


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--background", "--minbias", dest="background", type=Path,
        default=ROOT / "data/minbias_10000.pt",
        help="Background .pt input (default: repository data/minbias_10000.pt).",
    )
    parser.add_argument(
        "--signal", "--zz", dest="signal", type=Path,
        default=ROOT / "data/ZZ_leptonic.pt",
        help="Signal .pt input, e.g. ggHgammagamma.pt (default: repository data/ZZ_leptonic.pt).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data",
        help="Directory for reference.pt and anomalous.pt; existing files are overwritten with warnings.",
    )
    parser.add_argument("--no-overwrite", action="store_true",
                        help="Refuse to replace existing output files.")
    parser.add_argument("--reference-count", type=int, default=2000)
    parser.add_argument("--background-count", type=int, default=1000)
    parser.add_argument("--signal-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    inputs = [args.background.expanduser().resolve(), args.signal.expanduser().resolve()]
    output_dir = args.output_dir.expanduser().resolve()
    outputs = [output_dir / "reference.pt", output_dir / "anomalous.pt"]
    if inputs[0] == inputs[1] or (all(path.exists() for path in inputs) and inputs[0].samefile(inputs[1])):
        parser.error("Background and signal inputs must be different files.")
    for path in outputs:
        if path.is_symlink():
            parser.error(f"Output must not be a symbolic link: {path}")
        if any(path == source or (path.exists() and source.exists() and path.samefile(source))
               for source in inputs):
            parser.error(f"Output must not overwrite an input file: {path}")
        if path.exists():
            if not path.is_file():
                parser.error(f"Output must be a regular file: {path}")
            if args.no_overwrite:
                parser.error(f"Refusing to overwrite {path}; choose a new --output-dir.")
            warnings.warn(f"Overwriting existing file: {path}", UserWarning, stacklevel=2)
    try:
        samples = [torch.load(path, map_location="cpu", weights_only=True) for path in inputs]
        reference, anomalous = make_datasets(
            *samples, reference_count=args.reference_count,
            background_count=args.background_count, signal_count=args.signal_count,
            seed=args.seed,
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    output_dir.mkdir(parents=True, exist_ok=True)
    # Serialize both datasets before publication, so a serialization failure
    # cannot truncate or replace either previous output. Stage on the same disk.
    with tempfile.TemporaryDirectory(prefix=".reference_anomalous_", dir=output_dir) as staging:
        staged = [Path(staging) / path.name for path in outputs]
        for path, sample in zip(staged, (reference, anomalous)):
            sample["source_files"] = [str(source) for source in inputs]
            torch.save(sample, path)
        for source, target in zip(staged, outputs):
            if args.no_overwrite:
                os.link(source, target)  # Exclusive publication, including races.
            else:
                os.replace(source, target)
    for path, sample in zip(outputs, (reference, anomalous)):
        print(f"Saved {len(sample['label'])} events to {path}")
    print(f"Background input: {inputs[0]}")
    print(f"Signal input: {inputs[1]}")
    print(f"Seed: {args.seed}; background overlap: 0 events")


if __name__ == "__main__":
    main()
