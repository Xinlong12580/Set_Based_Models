"""Train an ERL grid, validate a saved scan, or run both stages."""

import argparse
from glob import escape, glob
from itertools import product
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from casual_tests import train_ERL as training
from casual_tests import validate_ERL as validation
from casual_tests import _quantitative_reporting as reporting
from casual_tests.erl_scores import validation_score_settings


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    training.add_input_arguments(result, model_outputs=False)
    training.add_training_arguments(result, scan=True)
    mode = result.add_mutually_exclusive_group()
    mode.add_argument("--train-only", action="store_true", help="Save the scan without running validation.")
    mode.add_argument("--validate-only", action="store_true",
                      help="Infer saved scan models from the output prefix and skip training.")
    result.add_argument("--validation-signal", type=Path)
    result.add_argument("--validation-background", type=Path)
    validation.add_pruning_arguments(result, ("--validation-reference-penalty", "--pruning-reference-penalty"),
                                     "validation_reference_penalty", score_prefix="validation")
    return result


def train_scan(args):
    prefix = training.output_prefix(args)
    data = training.load_inputs(args.signal, args.background, args)
    grid = [getattr(args, name) for name in training.SEARCH_PARAMETERS]
    count = math.prod(len(values) for values in grid)
    runs = []
    for index, values in enumerate(product(*grid)):
        parameters = dict(zip(training.SEARCH_PARAMETERS, values))
        print(f"\nSEARCH {index + 1}/{count}: {parameters}", flush=True)
        run_args = argparse.Namespace(**{**vars(args), **parameters, "output_prefix": Path(f"{prefix}.run{index:03d}"),
                                         "model": None, "report": None})
        model_path, report_path = training.train(run_args, data=data)
        runs.append({"model": str(model_path), "report": str(report_path), "search_parameters": parameters})
    manifest = Path(f"{prefix}.scan.json")
    reporting._write_report(manifest, {"format": "set_based_models_erl_scan", "schema_version": 1,
                            "signal": str(data[0].path), "background": str(data[1].path), "runs": runs},
                            overwrite=args.overwrite_output)
    print(f"Scan manifest saved to: {manifest}")
    return manifest


def validation_path(path, override, validate_only):
    if override is not None:
        return override
    if validate_only and not path.stem.endswith("_train"):
        return path
    return path.with_name(path.stem.removesuffix("_train") + "_validation.pt")


def main(argv=None):
    args = parser().parse_args(argv)
    prefix = training.output_prefix(args)
    models = None
    if args.validate_only:
        manifest = Path(f"{prefix}.scan.json")
        if manifest.is_file():
            print(f"Validation only: using saved scan {manifest}", flush=True)
        else:
            manifest = None
            models = sorted(glob(escape(str(prefix)) + ".run[0-9]*.training.model.json"))
            if not models:
                raise FileNotFoundError(f"No saved scan manifest or training models found for prefix {prefix}")
            print(f"Validation only: found {len(models)} saved models for {prefix}", flush=True)
    else:
        manifest = train_scan(args)
        if args.train_only:
            return manifest
    validation_args = argparse.Namespace(**vars(args))
    validation_args.signal = validation_path(args.signal, args.validation_signal, args.validate_only)
    validation_args.background = validation_path(args.background, args.validation_background, args.validate_only)
    validation_args.output_prefix = prefix
    validation_args.reference_penalty = args.validation_reference_penalty
    validation_args.score, validation_args.score_delta = validation_score_settings(args, "validation")
    validation_args.manifest = manifest
    validation_args.models = models
    validation_args.model = validation_args.report = None
    return validation.validate(validation_args)


if __name__ == "__main__":
    main()
