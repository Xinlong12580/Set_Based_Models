"""Mix compatible background and signal .pt files (requires PyTorch).

Example, run from the repository root:
  python scripts/make_reference_anomalous.py \\
      --background 'data/SIGNAL/train_minbias_*.pt' \\
      --signal 'data/SIGNAL/train_ggHgammagamma_*.pt' \\
      --output-dir data/ggHgammagamma_mixture

By default, reference.pt contains 2,000 background events; anomalous.pt
contains 1,000 different background events and 100 signal events, shuffled.
Sampling is without replacement. Disjoint rows avoid background event reuse;
statistical independence also assumes the source events were generated
independently and that the input files do not share events.

Use --background and --signal for one or more input paths or wildcard patterns.
Quoted patterns are expanded here; shell-expanded lists also work. Matches
are sorted within each pattern, and duplicate files within a class are used
only once. Background and signal pools must not share files. Sampling is
uniform over events, not files. Only one input file is held in RAM at a time,
in addition to the requested outputs; inputs are scanned once to check schemas
and count events, then files supplying selected events are loaded again.
The old --minbias and --zz options remain aliases. Default paths still point
to data/minbias_10000.pt and data/ZZ_leptonic.pt, relative to this repository.

Inputs must have matching keys, tensor shapes after the event dimension,
dtypes, and non-tensor metadata (including pf_columns and obj_columns).
All aligned event tensors and original process labels are preserved; labels
are not used to select events or infer which input is signal. Added
``source_id`` (0=background input, 1=signal input) and ``source_index`` identify
each output event's input .pt row, even when the inputs share process labels.
``source_files`` lists background paths followed by signal paths. For multi-file
inputs, ``source_file_index`` identifies the actual path in that list;
``source_index`` remains the row within that file. Single-file inputs retain
the original format (source_id also indexes the two source_files).
``sampling_seed`` records the seed. Source indices do not refer to raw Parquet rows.
Patterns include every matching file: avoid stale chunks from earlier runs.
Existing outputs are overwritten by default, with a warning for each file.
Use --no-overwrite to refuse replacement. Input files and output symlinks
are protected; both outputs are serialized before replacing existing files.
"""

import argparse
import glob
import os
from pathlib import Path
import tempfile
import warnings

import torch


ROOT = Path(__file__).resolve().parents[1]
PROVENANCE_KEYS = {"source_id", "source_index", "source_file_index", "source_files", "sampling_seed"}


def _validate_options(reference_count, background_count, signal_count, seed):
    for count in (reference_count, background_count, signal_count):
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError("All event counts must be positive integers.")
    if not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63).")


def _describe_sample(sample, name):
    """Validate a dataset and retain only tensor layouts and small metadata."""
    required = {"pf", "obj", "label", "pf_columns", "obj_columns"}
    if not isinstance(sample, dict) or not required <= sample.keys():
        raise ValueError(f"{name} must contain {sorted(required)}.")
    if PROVENANCE_KEYS & sample.keys():
        raise ValueError(f"{name} already contains reserved provenance keys.")
    labels = sample["label"]
    if not isinstance(labels, torch.Tensor) or labels.ndim != 1:
        raise ValueError(f"{name}: label must be a one-dimensional tensor.")
    tensors, metadata = {}, {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            if value.ndim == 0 or value.shape[0] != len(labels):
                raise ValueError(f"{name}: {key} is not an aligned event tensor.")
            if value.device.type != "cpu":
                raise ValueError(f"{name}: {key} must be on CPU.")
            tensors[key] = (value.shape[1:], value.dtype)
        else:
            metadata[key] = value
    for key in ("pf", "obj"):
        tensor = sample[key]
        columns = sample[key + "_columns"]
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
            raise ValueError(f"{name}: {key} must be a three-dimensional tensor.")
        if not isinstance(columns, list) or len(columns) != tensor.shape[-1]:
            raise ValueError(f"{name}: {key}_columns does not match {key}.")
    return tensors, metadata


def _check_schema(sample, schema, name):
    tensors, metadata = _describe_sample(sample, name)
    expected_tensors, expected_metadata = schema
    if (tensors.keys() | metadata.keys()) != (expected_tensors.keys() | expected_metadata.keys()):
        raise ValueError("The input dictionaries must have matching keys.")
    for key, layout in expected_tensors.items():
        if tensors.get(key) != layout:
            raise ValueError(f"Input tensor shapes/dtypes differ for {key}: {name}.")
    for key, value in expected_metadata.items():
        if key not in metadata or metadata[key] != value:
            raise ValueError(f"Input metadata differs for {key}: {name}.")


def _draw_indices(background_size, signal_size, reference_count, background_count, signal_count, seed):
    if background_size < reference_count + background_count:
        raise ValueError("Not enough background events for disjoint samples.")
    if signal_size < signal_count:
        raise ValueError("Not enough signal events.")

    rng = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(background_size, generator=rng)
    reference_indices = order[:reference_count]
    background_indices = order[reference_count:reference_count + background_count]
    signal_indices = torch.randperm(signal_size, generator=rng)[:signal_count]
    mixture_order = torch.randperm(background_count + signal_count, generator=rng)
    return reference_indices, background_indices, signal_indices, mixture_order


def make_datasets(background, signal, *, reference_count=2000, background_count=1000,
                  signal_count=100, seed=42):
    """Return aligned datasets with disjoint background rows and unchanged labels."""
    _validate_options(reference_count, background_count, signal_count, seed)
    schema = _describe_sample(background, "Background")
    _check_schema(signal, schema, "Signal")
    tensor_keys, metadata = schema
    reference_indices, background_indices, signal_indices, mixture_order = _draw_indices(
        len(background["label"]), len(signal["label"]),
        reference_count, background_count, signal_count, seed,
    )

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


def _expand_inputs(expressions):
    """Expand sorted local globs, keeping explicit order and no file aliases."""
    files, seen = [], set()
    for expression in expressions:
        pattern = str(Path(expression).expanduser())
        matches = sorted(glob.glob(pattern)) if glob.has_magic(pattern) else [pattern]
        if not matches:
            raise ValueError(f"No input files match: {expression}")
        for match in matches:
            path = Path(match).resolve()
            stat = path.stat()
            identity = (stat.st_dev, stat.st_ino)
            if identity not in seen:
                files.append(path)
                seen.add(identity)
    return files


def _mix_files(background_files, signal_files, *, reference_count, background_count, signal_count, seed):
    """Two passes: count/check one file at a time, then copy selected rows only."""
    _validate_options(reference_count, background_count, signal_count, seed)
    inputs = background_files + signal_files
    schema, counts = None, []
    for path in inputs:
        sample = torch.load(path, map_location="cpu", weights_only=True)
        if schema is None:
            schema = _describe_sample(sample, str(path))
        else:
            _check_schema(sample, schema, str(path))
        counts.append(len(sample["label"]))
        del sample

    background_size = sum(counts[:len(background_files)])
    signal_size = sum(counts[len(background_files):])
    reference_indices, background_indices, signal_indices, mixture_order = _draw_indices(
        background_size, signal_size, reference_count, background_count, signal_count, seed,
    )
    # Global indices are used only for drawing rows; saved source_index values
    # remain local to the actual PT file, as in the single-file interface.
    mixture_indices = torch.cat((background_indices, signal_indices + background_size))
    selections = (reference_indices, mixture_indices.index_select(0, mixture_order))
    tensors, metadata = schema
    results = []
    for indices in selections:
        result = {**metadata, "sampling_seed": seed}
        result.update({key: torch.empty((len(indices), *shape), dtype=dtype)
                       for key, (shape, dtype) in tensors.items()})
        result["source_id"] = (indices >= background_size).to(torch.int64)
        result["source_index"] = torch.empty(len(indices), dtype=torch.int64)
        if len(inputs) > 2:
            result["source_file_index"] = torch.empty(len(indices), dtype=torch.int64)
        results.append(result)

    offset = 0
    for file_index, (path, count) in enumerate(zip(inputs, counts)):
        positions = [torch.where((indices >= offset) & (indices < offset + count))[0]
                     for indices in selections]
        if any(len(rows) for rows in positions):
            sample = torch.load(path, map_location="cpu", weights_only=True)
            _check_schema(sample, schema, str(path))
            if len(sample["label"]) != count:
                raise ValueError(f"Input event count changed while mixing: {path}")
            for result, indices, rows in zip(results, selections, positions):
                local_rows = indices.index_select(0, rows) - offset
                for key in tensors:
                    result[key].index_copy_(0, rows, sample[key].index_select(0, local_rows))
                result["source_index"].index_copy_(0, rows, local_rows)
                if len(inputs) > 2:
                    result["source_file_index"].index_fill_(0, rows, file_index)
            del sample
        offset += count
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--background", "--minbias", dest="background", nargs="+",
        default=[ROOT / "data/minbias_10000.pt"],
        help="Background .pt paths or wildcard patterns (default: repository data/minbias_10000.pt).",
    )
    parser.add_argument(
        "--signal", "--zz", dest="signal", nargs="+",
        default=[ROOT / "data/ZZ_leptonic.pt"],
        help="Signal .pt paths or wildcard patterns (default: repository data/ZZ_leptonic.pt).",
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

    try:
        background_files, signal_files = (_expand_inputs(patterns)
                                          for patterns in (args.background, args.signal))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    inputs = background_files + signal_files
    output_dir = args.output_dir.expanduser().resolve()
    outputs = [output_dir / "reference.pt", output_dir / "anomalous.pt"]
    background_ids = {(path.stat().st_dev, path.stat().st_ino) for path in background_files}
    signal_ids = {(path.stat().st_dev, path.stat().st_ino) for path in signal_files}
    if background_ids & signal_ids:
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
        reference, anomalous = _mix_files(
            background_files, signal_files, reference_count=args.reference_count,
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
    print(f"Background inputs ({len(background_files)}): " + ", ".join(map(str, background_files)))
    print(f"Signal inputs ({len(signal_files)}): " + ", ".join(map(str, signal_files)))
    print(f"Seed: {args.seed}; background overlap: 0 events")


if __name__ == "__main__":
    main()
