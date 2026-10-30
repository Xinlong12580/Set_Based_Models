"""Train one ERL configuration and save its unpruned model and report."""

import argparse
from contextlib import nullcontext
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from set_based_models import ExistentialRuleList
from set_based_models.rule_list.generation import _object_priority_progress
from casual_tests import _erl_reporting as reporting_tools
from casual_tests._erl_reporting import load_dataset
from casual_tests import _quantitative_reporting as reporting
from casual_tests.erl_scores import DEFAULT_DELTA, SCORE_NAMES, make_score, score_settings
from casual_tests._erl_progress import SearchProgress


SEARCH_PARAMETERS = {
    "max_rule_depth": (int, 4), "beam_width": (int, 40),
    "max_candidate_rules": (int, 300), "max_thresholds": (int, 32),
    "minimum_count": (int, 1), "min_signal_efficiency": (float, .01),
    "reference_penalty": (float, 1.), "beam_novelty_weight": (float, 0.),
    "score": (str, "gini"), "score_delta": (float, DEFAULT_DELTA),
    "beam_signal_similarity_threshold": (float, reporting.SIGNAL_SIMILARITY_THRESHOLD),
    "beam_background_similarity_threshold": (float, reporting.BACKGROUND_SIMILARITY_THRESHOLD),
}


def add_input_arguments(result, *, model_outputs=True):
    result.add_argument("--signal", type=Path, required=True)
    result.add_argument("--background", type=Path, required=True)
    result.add_argument("--no-object-features", "--event-only", action="store_true")
    result.add_argument("--no-event-features", "--object-only", action="store_true")
    result.add_argument("--basic-object-features", action="store_true")
    result.add_argument("--pair-tensor", "--use-pair-tensor", action="store_true")
    result.add_argument("--output-prefix", type=Path)
    if model_outputs:
        result.add_argument("--model", type=Path, help="Override the saved model path.")
        result.add_argument("--training-report", "--report", dest="report", type=Path, help="Override the report path.")
    else:
        result.set_defaults(model=None, report=None)
    result.add_argument("--overwrite-output", "--overwrite-report", dest="overwrite_output", action="store_true")
    result.add_argument("--print-top", type=int, default=10)
    result.add_argument("--print-reference-top", type=int, default=5)
    result.add_argument("--print-pruned-top", type=int, default=20)


def add_training_arguments(result, *, scan=False):
    for name, (kind, default) in SEARCH_PARAMETERS.items():
        flags = ["--" + name.replace("_", "-")]
        if name == "beam_novelty_weight":
            flags.append("--novelty-beta")
        if name == "reference_penalty":
            flags.append("--beam-reference-penalty")
        if name == "score":
            flags.append("--beam-score")
        if name == "score_delta":
            flags.extend(("--delta", "--beam-score-delta"))
        result.add_argument(*flags, type=kind, nargs="+" if scan else None,
                            default=[default] if scan else default,
                            **({"choices": SCORE_NAMES} if name == "score" else {}),
                            help=f"{'Values to scan' if scan else 'Value'} (default: {default}).")
    result.add_argument("--beam-similarity-filter", action="store_true")
    result.add_argument("--object-cut-priority", nargs="+", default=[])
    result.add_argument("--allow-validity-cuts", action="store_true")
    result.add_argument("--progress", action="store_true")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    add_input_arguments(result)
    add_training_arguments(result)
    return result


def load_inputs(signal_path, background_path, args):
    signal, background = [load_dataset(path, use_objects=not args.no_object_features,
                          use_events=not args.no_event_features, basic_object_features=args.basic_object_features,
                          use_pairs=args.pair_tensor) for path in (signal_path, background_path)]
    inputs = dict(signal=None if args.no_object_features else signal.objects,
                  background=None if args.no_object_features else background.objects,
                  signal_slot_mask=None if args.no_object_features else signal.slot_mask,
                  background_slot_mask=None if args.no_object_features else background.slot_mask,
                  signal_event_features=signal.event_features, background_event_features=background.event_features)
    return signal, background, inputs


def output_prefix(args):
    mode = "event_only" if args.no_object_features else "object_only" if args.no_event_features else "combined"
    suffix = ("_pairs" if args.pair_tensor else "") + ("_basic_objects" if args.basic_object_features else "")
    prefix = args.output_prefix
    if prefix is None and args.report is not None:
        name = args.report.name
        ending = ".training.report.json" if name.endswith(".training.report.json") else ".report.json"
        prefix = args.report.with_name(name.removesuffix(ending)) if name.endswith(ending) else args.report.with_suffix("")
    return (prefix or Path("training_outputs") / f"ERL_{args.signal.parent.name}{suffix}_{mode}").expanduser().resolve()


def save_outputs(model, signal, background, args, stage, prefix):
    model_path = args.model or Path(f"{prefix}.{stage}.model.json")
    report_path = args.report or Path(f"{prefix}.{stage}.report.json")
    model_path, report_path = [path.expanduser().resolve() for path in (model_path, report_path)]
    if model_path == report_path or any(path in (signal.path, background.path) for path in (model_path, report_path)):
        raise ValueError("Model and report paths must be distinct and separate from input tensors")
    options = argparse.Namespace(**{
        "stricter_score_improvement": 0., "max_trigger_rate_khz": None, "input_rate_mhz": 40.,
        "require_complete_priority": False, "final_similarity_filter": False,
        "progress": False,
        "beam_similarity_filter": model.beam_similarity_filter,
        "signal_similarity_threshold": model.beam_signal_similarity_threshold,
        "background_similarity_threshold": model.beam_background_similarity_threshold,
        **vars(args),
    })
    mode = "event_only" if args.no_object_features else "object_only" if args.no_event_features else "combined"
    max_acceptance = None if options.max_trigger_rate_khz is None else options.max_trigger_rate_khz / (1000 * options.input_rate_mhz)
    report = reporting_tools._training_report(model, signal, background, options, mode, max_acceptance, report_path, model_path)
    report.update(stage=stage, trainer=f"casual_tests/{'train_ERL' if stage == 'training' else 'validate_ERL'}.py",
                  statistics_source=model.statistics_source_,
                  outputs={"model": str(model_path), f"{stage}_report": str(report_path)})
    report["evaluation_note"] = ("Unpruned training candidates." if stage == "training" else
                                 "Validation used for selection; not an independent test.")
    configuration = report["configuration"]
    active_score = score_settings(getattr(args, "score", "gini"), reference_penalty=args.reference_penalty,
                                  delta=getattr(args, "score_delta", DEFAULT_DELTA))
    configuration.update(pruning_stage="not_applied" if stage == "training" else "after_validation_recalculation",
                         final_similarity_pruning=stage == "validation" and options.final_similarity_filter,
                         beam_signal_similarity_threshold=model.beam_signal_similarity_threshold,
                         beam_background_similarity_threshold=model.beam_background_similarity_threshold,
                         final_signal_similarity_threshold=options.signal_similarity_threshold if stage == "validation" else None,
                         final_background_similarity_threshold=options.background_similarity_threshold if stage == "validation" else None,
                         scoring_stage="beam_search" if stage == "training" else "validation_pruning",
                         beam_reference_penalty=active_score["reference_penalty"] if stage == "training" else getattr(args, "beam_reference_penalty", None),
                         pruning_reference_penalty=active_score["reference_penalty"] if stage == "validation" else None,
                         beam_score_name=active_score["score_name"] if stage == "training" else getattr(args, "beam_score_name", None),
                         beam_score_delta=active_score["score_delta"] if stage == "training" else getattr(args, "beam_score_delta", None),
                         rate_estimation=f"{stage}_reference_acceptance_times_input_rate_no_confidence_bound")
    if stage == "training":
        configuration.update(stricter_score_improvement=None, pruning_policy=None)
    configuration["self_pruning_applied"] = model.self_pruning_provenance_ is not None
    if model.self_pruning_provenance_ is not None:
        provenance = model.self_pruning_provenance_
        report["self_pruning_provenance"] = provenance
        configuration.update(pruning_stage="after_validation_self_pruning",
                             self_prune_score_improvement=provenance["configuration"]["stricter_score_improvement"])
        report["summary"].update(self_pruning_source_rule_count=len(provenance["source_rule_ids"]),
                                 self_pruning_evaluated_variant_count=provenance["evaluated_variant_count"],
                                 self_pruning_locally_retained_variant_count=provenance["locally_retained_variant_count"])
        for record in report["results"]["rules"]:
            record["self_pruning_parent_rule_ids"] = provenance["rules"][record["rule_id"]]["parent_rule_ids"]
    if model.merge_provenance_ is not None:
        report["merge_provenance"] = model.merge_provenance_
        report["source_scan_runs"] = getattr(args, "source_scan_runs", None)
        configuration["penalty_applied_during_search"] = None
        configuration["priority_completion_scope"] = "each_candidate_source_priority"
        for name in (*SEARCH_PARAMETERS, "beam_similarity_filter", "allow_validity_cuts", "object_cut_priority"):
            if name != "reference_penalty":
                configuration.pop(name, None)
        sources = model.merge_provenance_["sources"]
        report["validity_flags_are_guard_only"] = not any(s["configuration"]["allow_validity_cuts"] for s in sources)
        for record in report["results"]["rules"]:
            origin = model.merge_provenance_["rules"][record["rule_id"]]
            source = sources[origin["source_index"]]
            priority = source["configuration"]["object_cut_priority"]
            rule = model.get_rule(record["rule_id"])
            progress = _object_priority_progress(rule.cuts, priority, model.observables)
            record.update(source={**origin, "model": source["name"]}, object_cut_priority=priority,
                          object_cut_priority_remaining=priority[progress:], object_cut_priority_complete=progress == len(priority))
    configuration.update(active_score)
    if stage == "training":
        configuration["penalty_applied_during_search"] = active_score["score_name"] == "gini"
    model.save(model_path, overwrite=args.overwrite_output)
    reporting._write_report(report_path, report, overwrite=args.overwrite_output)
    reporting._print_header(report, report_path)
    print(f"STAGE: {stage}; candidates={len(model.candidate_rules_)}; selected={len(model.selected_rules_)}")
    formula = (f"Gini gain - {args.reference_penalty:g} * reference acceptance" if active_score["score_name"] == "gini"
               else f"{active_score['score_formula']}; delta={active_score['score_delta']:g}")
    print(f"{'BEAM' if stage == 'training' else 'PRUNING'} SCORE: {formula}.")
    rules = report["results"]["rules"]
    reporting._print_section("MIXTURE-ENRICHED CANDIDATES")
    reporting_tools._print_rules([r for r in rules if r["direction"] == "mixture-enriched"], args.print_top)
    if args.print_reference_top:
        reporting._print_section("REFERENCE-ENRICHED CANDIDATES")
        reporting_tools._print_rules([r for r in rules if r["direction"] == "reference-enriched"], args.print_reference_top)
    if stage == "validation":
        reporting._print_section("VALIDATION-SELECTED RULES")
        reporting_tools._print_rules([r for r in rules if r["retained_after_final_pruning"]], args.print_pruned_top)
    print(f"Model saved to: {model_path}\nReport saved to: {report_path}")
    return model_path, report_path


def train(args, data=None):
    """Train one scalar configuration; callers may reuse a loaded input tuple."""
    scorer = make_score(getattr(args, "score", "gini"), reference_penalty=args.reference_penalty,
                        delta=getattr(args, "score_delta", DEFAULT_DELTA))
    signal, background, inputs = load_inputs(args.signal, args.background, args) if data is None else data
    if signal.object_observables != background.object_observables or signal.event_observables != background.event_observables:
        raise ValueError("Training inputs must have matching feature schemas")
    configuration = {name: getattr(args, name) for name in SEARCH_PARAMETERS
                     if name not in ("reference_penalty", "score", "score_delta")}
    model = ExistentialRuleList(signal.object_observables, event_observables=signal.event_observables,
                               scoring_function=scorer,
                               beam_similarity_filter=args.beam_similarity_filter, allow_validity_cuts=args.allow_validity_cuts,
                               object_cut_priority=args.object_cut_priority, **configuration)
    with SearchProgress() if args.progress else nullcontext() as progress:
        model.fit(**inputs, progress_callback=progress)
    return save_outputs(model, signal, background, args, "training", output_prefix(args))


def main(argv=None):
    return train(parser().parse_args(argv))


if __name__ == "__main__":
    main()
