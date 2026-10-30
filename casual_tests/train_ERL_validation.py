"""Train one ERL configuration, then recalculate and prune on validation."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from casual_tests import train_ERL as training
from casual_tests import validate_ERL as validation
from casual_tests.erl_scores import validation_score_settings


def parser():
    result = training.parser()
    result.description = __doc__
    result.add_argument("--validation-signal", type=Path)
    result.add_argument("--validation-background", type=Path)
    validation.add_pruning_arguments(result, ("--pruning-reference-penalty", "--validation-reference-penalty"),
                                     "pruning_reference_penalty", score_prefix="pruning")
    result.set_defaults(beam_signal_similarity_threshold=None, beam_background_similarity_threshold=None)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    prefix = training.output_prefix(args)
    training_args = argparse.Namespace(**vars(args))
    training_args.output_prefix = prefix
    if training_args.beam_signal_similarity_threshold is None:
        training_args.beam_signal_similarity_threshold = args.signal_similarity_threshold
    if training_args.beam_background_similarity_threshold is None:
        training_args.beam_background_similarity_threshold = args.background_similarity_threshold
    model_path, _ = training.train(training_args)

    validation_args = argparse.Namespace(**vars(args))
    validation_args.signal = args.validation_signal or args.signal.with_name(args.signal.stem.removesuffix("_train") + "_validation.pt")
    validation_args.background = args.validation_background or args.background.with_name(args.background.stem.removesuffix("_train") + "_validation.pt")
    validation_args.output_prefix = prefix
    validation_args.beam_reference_penalty = args.reference_penalty if args.score == "gini" else None
    validation_args.beam_score_name = args.score
    validation_args.beam_score_delta = args.score_delta if args.score != "gini" else None
    validation_args.reference_penalty = args.pruning_reference_penalty
    validation_args.score, validation_args.score_delta = validation_score_settings(args, "pruning")
    validation_args.manifest, validation_args.models = None, [str(model_path)]
    validation_args.model = validation_args.report = None
    return validation.validate(validation_args)


if __name__ == "__main__":
    main()
