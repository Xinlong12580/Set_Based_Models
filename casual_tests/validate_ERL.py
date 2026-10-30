"""Merge ERL candidates, optionally shorten on validation, then prune the pool."""

import argparse
from glob import glob
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from set_based_models import ExistentialRuleList
from casual_tests import train_ERL as training
from casual_tests import _quantitative_reporting as reporting
from casual_tests.erl_scores import DEFAULT_DELTA, SCORE_NAMES, make_score


def add_pruning_arguments(result, reference_options=("--reference-penalty", "--pruning-reference-penalty"),
                          reference_dest="reference_penalty", *, score_prefix=None):
    result.add_argument(*reference_options, dest=reference_dest, type=float, default=1.,
                        help="Common reference penalty for validation scoring.")
    if score_prefix is None:
        result.add_argument("--score", "--pruning-score", choices=SCORE_NAMES, default="gini")
        result.add_argument("--score-delta", "--delta", "--pruning-score-delta", type=float, default=DEFAULT_DELTA)
    else:
        result.add_argument("--validation-score", "--pruning-score", dest=f"{score_prefix}_score", choices=SCORE_NAMES,
                            help="Common validation score; defaults to the uniform beam score, or gini for a mixed grid.")
        result.add_argument("--validation-score-delta", "--pruning-score-delta", dest=f"{score_prefix}_score_delta", type=float,
                            help="Validation delta; defaults to the uniform beam delta, or 0.001 for a mixed grid.")
    result.add_argument("--stricter-score-improvement", type=float, default=0.)
    result.add_argument("--self-prune", action="store_true", help="Shorten each rule on validation before final pruning.")
    result.add_argument("--self-prune-score-improvement", type=float,
                        help="Local shortening threshold; defaults to --stricter-score-improvement.")
    result.add_argument("--max-trigger-rate-khz", type=float)
    result.add_argument("--input-rate-mhz", type=float, default=40.)
    result.add_argument("--require-complete-priority", action="store_true")
    result.add_argument("--final-similarity-filter", action="store_true")
    result.add_argument("--final-signal-similarity-threshold", "--signal-similarity-threshold",
                        dest="signal_similarity_threshold", type=float, default=reporting.SIGNAL_SIMILARITY_THRESHOLD)
    result.add_argument("--final-background-similarity-threshold", "--background-similarity-threshold",
                        dest="background_similarity_threshold", type=float, default=reporting.BACKGROUND_SIMILARITY_THRESHOLD)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    training.add_input_arguments(result)
    source = result.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--models", nargs="+", help="Model paths or quoted wildcard patterns.")
    add_pruning_arguments(result)
    return result


def validate(args):
    scorer = make_score(getattr(args, "score", "gini"), reference_penalty=args.reference_penalty,
                        delta=getattr(args, "score_delta", DEFAULT_DELTA))
    if args.manifest is not None:
        scan = json.loads(args.manifest.read_text())
        paths = [run["model"] for run in scan["runs"]]
        args.source_scan_runs = scan["runs"]
        prefix = args.output_prefix or Path(str(args.manifest).removesuffix(".scan.json"))
    else:
        paths = [Path(path) for pattern in args.models
                 for path in sorted(glob(str(Path(pattern).expanduser()))) or [str(Path(pattern).expanduser())]]
        prefix = args.output_prefix or Path("training_outputs/ERL_combined")
    model = ExistentialRuleList.load_many(paths, scoring_function=scorer)
    signal, background, inputs = training.load_inputs(args.signal, args.background, args)
    for dataset in (signal, background):
        if model.observables != dataset.object_observables or model.event_observables != dataset.event_observables:
            raise ValueError("Validation feature schemas must match the saved models")
    max_acceptance = None if args.max_trigger_rate_khz is None else args.max_trigger_rate_khz / (1000 * args.input_rate_mhz)
    max_acceptance = None if max_acceptance is None else min(1., max_acceptance)
    if args.self_prune:
        improvement = args.stricter_score_improvement if args.self_prune_score_improvement is None else args.self_prune_score_improvement
        print(f"Self-pruning {len(model.candidate_rules_)} candidates from {len(paths)} models on validation...", flush=True)
        model = model.self_prune_candidates(**inputs, stricter_score_improvement=improvement,
                                            max_reference_acceptance=max_acceptance,
                                            require_complete_priority=args.require_complete_priority)
        print(f"Self-pruning produced {len(model.candidate_rules_)} pooled candidates before final pruning.", flush=True)
    else:
        print(f"Recalculating {len(model.candidate_rules_)} candidates from {len(paths)} models on validation...", flush=True)
        model.recalculate_candidates(**inputs)
    model.prune_candidates(require_complete_priority=args.require_complete_priority,
                           stricter_score_improvement=args.stricter_score_improvement,
                           max_reference_acceptance=max_acceptance,
                           similarity_filter=args.final_similarity_filter,
                           signal_similarity_threshold=args.signal_similarity_threshold,
                           background_similarity_threshold=args.background_similarity_threshold)
    return training.save_outputs(model, signal, background, args, "validation", prefix.expanduser().resolve())


def main(argv=None):
    return validate(parser().parse_args(argv))


if __name__ == "__main__":
    main()
