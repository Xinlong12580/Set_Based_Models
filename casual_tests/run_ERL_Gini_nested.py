"""Penalized-Gini ERL using both saved tensors, or either tensor alone.

Score/pruning defaults follow run_ERL_Gini_nested_event_only.py. Logical-subset pruning
supports event, object, and mixed rules, preserving same-slot/count semantics.
Schema-declared validity flags automatically guard object and event cuts.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from set_based_models import EventCut, ExistentialRuleList, ObservableSpec, SetInput
from casual_tests import _event_only_reporting as event_reporting
from casual_tests import _quantitative_reporting as reporting
from casual_tests.run_ERL_Gini_nested_event_only import _positive_float
from casual_tests.run_ERL_Gini_penalized_event_only import (
    ReferencePenalizedGini, _nonnegative_float, _similarity_threshold,
)

BASIC_OBJECT_COLUMNS = ("pt", "eta", "phi", "type_id")


@dataclass(frozen=True)
class Dataset(reporting.LoadedDataset):
    object_observables: tuple
    event_observables: tuple
    object_schema: list | None
    feature_schema: list | None


def _schema(raw, prefix, width, path):
    columns = raw.get(prefix + "_columns")
    if (not isinstance(columns, list) or len(columns) != width
            or not all(isinstance(name, str) and name.strip() for name in columns)):
        raise ValueError(f"{prefix}_columns must name every column in {path}.")
    columns = tuple(name.strip() for name in columns)
    if len(set(columns)) != len(columns):
        raise ValueError(f"Duplicate {prefix}_columns in {path}.")
    key = "obj_feature_schema" if prefix == "obj" else "event_feature_schema"
    schema = raw.get(key)
    if schema is None:
        if prefix == "obj":
            if list(columns) != reporting.OBJECT_COLUMNS:
                raise ValueError(f"{key} is required for non-legacy object columns in {path}.")
            specs = reporting.OBSERVABLES
        else:
            # The established schema-less event format contains multiplicities.
            specs = tuple(ObservableSpec(name, "ordinal") for name in columns)
    else:
        if (not isinstance(schema, list) or len(schema) != width
                or not all(isinstance(item, dict) for item in schema)):
            raise ValueError(f"{key} must describe every column in {path}.")
        specs = []
        for name, item in zip(columns, schema):
            if item.get("name") != name or "kind" not in item:
                raise ValueError(f"{key} names/order must match columns in {path}.")
            specs.append(reporting._observable_from_schema(name, item))
        specs = tuple(specs)
        json.dumps(schema, allow_nan=False)
    return columns, specs, schema


def _tensor(raw, key, ndim, rows, path):
    value = raw.get(key)
    if (not isinstance(value, torch.Tensor) or value.ndim != ndim
            or value.layout != torch.strided or value.is_complex()
            or any(size == 0 for size in value.shape) or value.shape[0] != rows):
        raise ValueError(f"{key} must be a nonempty real {ndim}D tensor with {rows} rows in {path}.")
    return value.detach().to(dtype=torch.float64).numpy()


def load_dataset(path, *, use_objects=True, use_events=True, basic_object_features=False):
    """Validate only enabled modalities; never replace NaNs in active features."""
    if not use_objects and not use_events:
        raise ValueError("Enable at least one of object or event features.")
    if basic_object_features and not use_objects:
        raise ValueError("Basic object features require object inputs to be enabled.")
    path = Path(path).expanduser().resolve()
    raw = torch.load(path, map_location="cpu", weights_only=True)
    labels = raw.get("label") if isinstance(raw, dict) else None
    if (not isinstance(labels, torch.Tensor) or labels.ndim != 1 or not len(labels)
            or labels.layout != torch.strided or labels.dtype not in {
                torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}):
        raise ValueError(f"label must be a nonempty 1D integer process-label tensor in {path}.")

    objects = mask = events = None
    object_specs = event_specs = event_columns = ()
    object_schema = event_schema = None
    if use_objects:
        objects = _tensor(raw, "obj", 3, len(labels), path)
        columns, object_specs, object_schema = _schema(raw, "obj", objects.shape[2], path)
        if basic_object_features:
            missing = [name for name in BASIC_OBJECT_COLUMNS if name not in columns]
            if missing:
                raise ValueError(f"Basic object features require columns {missing} in {path}.")
            indices = [columns.index(name) for name in BASIC_OBJECT_COLUMNS]
            objects = objects[..., indices]
            object_specs = tuple(object_specs[index] for index in indices)
            if object_schema is not None:
                object_schema = [object_schema[index] for index in indices]
            columns = BASIC_OBJECT_COLUMNS
        if "pt" not in columns:
            raise ValueError(f"obj_columns must contain pt to identify padded slots in {path}.")
        mask = objects[..., columns.index("pt")] > 0
    if use_events:
        events = _tensor(raw, "event_features", 2, len(labels), path)
        event_columns, event_specs, event_schema = _schema(
            raw, "event_feature", events.shape[1], path)
    validated = SetInput.from_arrays(
        objects, object_specs, slot_mask=mask,
        event_features=events, event_observables=event_specs,
    )
    for index, name in enumerate(event_columns):
        is_count = event_schema is None or event_schema[index].get("units") == "count"
        if is_count and np.any(events[:, index][validated.event_value_mask[:, index]] < 0):
            raise ValueError(f"Multiplicity {name!r} must be nonnegative in {path}.")
    return Dataset(path, validated.X, validated.slot_mask, labels.numpy(),
                   validated.event_features, event_columns,
                   object_specs, event_specs, object_schema, event_schema)


def _load_pair(parser, args, mode):
    try:
        signal, background = (load_dataset(path, use_objects=not args.no_object_features,
                                           use_events=not args.no_event_features,
                                           basic_object_features=args.basic_object_features)
                              for path in (args.signal, args.background))
        if signal.path.samefile(background.path):
            raise ValueError("Signal and background inputs must be different files.")
        if (signal.object_observables != background.object_observables
                or signal.object_schema != background.object_schema
                or signal.objects.shape[1:] != background.objects.shape[1:]
                or signal.event_observables != background.event_observables
                or signal.feature_schema != background.feature_schema):
            raise ValueError("Enabled signal/background shapes, columns, and schemas must match.")
        requested = (args.report.expanduser().absolute() if args.report is not None else
                     Path.cwd() / "reports" /
                     f"ERL_Gini_nested_flexible_{signal.path.parent.name}_{signal.path.stem}"
                     f"{'_basic_objects' if args.basic_object_features else ''}_{mode}.txt")
        target = requested.parent.resolve() / requested.name
        if target.is_symlink():
            raise ValueError("Refusing to write through a report symlink.")
        if (target in (signal.path, background.path) or
                (target.exists() and any(target.samefile(path) for path in (signal.path, background.path)))):
            raise ValueError("The report must not overwrite either input tensor.")
        if target.exists() and not target.is_file():
            raise ValueError("The report output must be a file, not a directory.")
        reporting._check_report_target(parser, target, args.overwrite_report)
        return signal, background, target
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))


def _rule_records(model, signal, background, result, args, max_acceptance):
    ranks = {rule.rule_id: rank for rank, rule in enumerate(result.retained, 1)}
    records = []
    for rank, rule in enumerate(model.candidate_rules_, 1):
        used_objects = {model.observables[cut.observable_index].name for cut in rule.cuts
                        if not isinstance(cut, EventCut)}
        remaining_priority = [name for name in model.object_cut_priority if name not in used_objects]
        normalized = result.normalized_cuts[rule.rule_id]
        parts = [reporting._predicate_text(cut, model.event_observables, include_scope=True)
                 for cut in normalized if isinstance(cut, EventCut)]
        slots = [cut for cut in normalized if not isinstance(cut, EventCut)]
        if slots:
            prefix = ("EXISTS same slot" if rule.minimum_count == 1
                      else f"COUNT >= {rule.minimum_count} matching slots")
            predicates = " AND ".join(reporting._predicate_text(cut, model.observables)
                                      for cut in slots)
            parts.append(f"{prefix}: {predicates}")
        simplified, effective_depth = " AND ".join(parts), len(normalized)
        steps = []
        for step in rule.steps:
            is_event = isinstance(step.cut, EventCut)
            schema = model.event_observables if is_event else model.observables
            stats = step.statistics
            steps.append({
                "depth": step.depth, "scope": "event" if is_event else "slot",
                "cut": reporting._cut_record(step.cut, schema),
                "display": reporting._predicate_text(step.cut, schema, include_scope=is_event),
                "prefix_score": step.generation_score,
                "prefix_gini_gain": reporting.gini_rule_score(stats),
                "prefix_reference_penalty_term": args.reference_penalty * stats.false_positive_rate,
                "prefix_signal_accepted": stats.signal_accepted,
                "prefix_background_accepted": stats.background_accepted,
                "prefix_signal_acceptance": stats.true_positive_rate,
                "prefix_reference_acceptance": stats.false_positive_rate,
            })
        metrics = reporting._selection_metrics(rule.signal_selection, rule.background_selection,
                                               signal.labels, background.labels)
        retained = rule.rule_id in ranks
        records.append({
            "rank": rank, "rule_id": rule.rule_id, "depth": rule.depth,
            "effective_depth_after_simplification": effective_depth,
            "logical_pruning_applicable": True, "minimum_count": rule.minimum_count,
            "object_cut_priority_remaining": remaining_priority,
            "score": rule.generation_score, "gini_gain": steps[-1]["prefix_gini_gain"],
            "reference_penalty_term": steps[-1]["prefix_reference_penalty_term"],
            "direction": metrics["direction"], "metrics": metrics, "steps": steps,
            "raw_rule": rule.to_text(), "simplified_rule": simplified,
            "normalized_cuts": [reporting._cut_record(
                cut, model.event_observables if isinstance(cut, EventCut) else model.observables)
                for cut in normalized],
            "estimated_reference_rate_khz": args.input_rate_mhz * 1000 * rule.statistics.false_positive_rate,
            "passes_reference_rate_limit": (max_acceptance is None or
                                            rule.statistics.false_positive_rate <= max_acceptance),
            "retained_after_final_pruning": retained,
            "retained_after_nested_pruning": retained,
            "retained_after_similarity_pruning": retained,
            "pruned_rank": ranks.get(rule.rule_id), "pruning": result.decisions[rule.rule_id],
        })
    return records


def _print_rules(records, limit):
    if not records:
        print("(none)")
        return
    shown = records if limit == 0 else records[:limit]
    for rank, record in enumerate(shown, 1):
        print(f"\n[{rank:03d}] global_rank={record['rank']} rule={record['rule_id']} "
              f"score={record['score']:.6g} gini_gain={record['gini_gain']:.6g} "
              f"reference_penalty_term={record['reference_penalty_term']:.6g} "
              f"depth={record['depth']} effective_depth={record['effective_depth_after_simplification']} "
              f"pruned_rank={record['pruned_rank']} logical_pruning={record['logical_pruning_applicable']}")
        print(f"  rule:       {record['simplified_rule']}")
        if record["object_cut_priority_remaining"]:
            print(f"  object priority remaining: {' -> '.join(record['object_cut_priority_remaining'])}")
        if record["raw_rule"] != record["simplified_rule"]:
            print(f"  raw rule:   {record['raw_rule']}")
        reporting._print_metrics(record["metrics"])
        print("  prefix history:")
        for step in record["steps"]:
            print(f"    {step['depth']}. [{step['scope']}] {step['display']} | "
                  f"score={step['prefix_score']:.6g} gain={step['prefix_gini_gain']:.6g} "
                  f"reference_penalty_term={step['prefix_reference_penalty_term']:.6g} | "
                  f"mixture={step['prefix_signal_accepted']} reference={step['prefix_background_accepted']}")
    if len(shown) < len(records):
        print(f"\n... {len(records) - len(shown)} additional rules are stored in the report.")


def main(argv=None):
    parser = event_reporting._parser(__doc__, model="ERL")
    parser.add_argument("--no-object-features", "--event-only", action="store_true",
                        help="Ignore obj and its schema; use native event-only ERL.")
    parser.add_argument("--no-event-features", "--object-only", action="store_true",
                        help="Ignore event_features and its schema; use object-only ERL.")
    parser.add_argument("--basic-object-features", action="store_true",
                        help="Use only pt, eta, phi, type_id from obj; event inputs are unchanged.")
    parser.add_argument("--object-cut-priority", nargs="+", default=[], metavar="FEATURE",
                        help="Compulsory first-use order of object features, e.g. type_id pt. "
                             "Incomplete rules remain eligible; event cuts are unrestricted.")
    parser.add_argument("--reference-penalty", type=_nonnegative_float, default=1.0,
                        help="Lambda in Gini - lambda * reference acceptance (default: 1; 0 = pure Gini).")
    parser.add_argument("--stricter-score-improvement", type=_nonnegative_float, default=0.0,
                        help="Required strict score improvement for event/object/mixed rules (default: 0).")
    parser.add_argument("--max-trigger-rate-khz", type=_nonnegative_float, default=None,
                        help="Optional per-rule rate budget, applied before logical pruning.")
    parser.add_argument("--input-rate-mhz", type=_positive_float, default=40.0,
                        help="Reference input-stream rate in MHz (default: 40).")
    parser.add_argument("--beam-similarity-filter", action="store_true",
                        help="Opt into beam Jaccard filtering; prefixes can be lost before final pruning.")
    parser.add_argument("--final-similarity-filter", action="store_true",
                        help="Opt into final Jaccard pruning for all rule types (default: off).")
    parser.add_argument("--signal-similarity-threshold", type=_similarity_threshold,
                        default=reporting.SIGNAL_SIMILARITY_THRESHOLD,
                        help="Mixture Jaccard threshold for enabled similarity stages; higher prunes less.")
    parser.add_argument("--background-similarity-threshold", type=_similarity_threshold,
                        default=reporting.BACKGROUND_SIMILARITY_THRESHOLD,
                        help="Reference Jaccard threshold for enabled similarity stages; higher prunes less.")
    parser.add_argument("--print-reference-top", type=reporting._nonnegative_int, default=5)
    parser.add_argument("--print-pruned-top", type=reporting._nonnegative_int, default=20)
    args = parser.parse_args(argv)
    if args.no_object_features and args.no_event_features:
        parser.error("Cannot disable both object and event features.")
    if args.basic_object_features and args.no_object_features:
        parser.error("--basic-object-features cannot be combined with --event-only/--no-object-features.")
    if args.object_cut_priority and args.no_object_features:
        parser.error("--object-cut-priority requires object features; cannot use --event-only.")
    mode = ("event_only" if args.no_object_features else
            "object_only" if args.no_event_features else "combined")
    max_acceptance = (None if args.max_trigger_rate_khz is None else
                      (args.max_trigger_rate_khz / 1000) / args.input_rate_mhz)
    if (not math.isfinite(args.input_rate_mhz * 1000) or
            (max_acceptance is not None and not math.isfinite(max_acceptance))):
        parser.error("rate values are too large or small for finite rate calculations")
    signal, background, target = _load_pair(parser, args, mode)
    try:
        model = ExistentialRuleList(
            signal.object_observables, event_observables=signal.event_observables,
            max_rule_depth=3, beam_width=50, max_candidate_rules=500, max_thresholds=32,
            minimum_count=1, min_signal_efficiency=0.01,
            allow_validity_cuts=args.allow_validity_cuts,
            object_cut_priority=args.object_cut_priority,
            scoring_function=ReferencePenalizedGini(args.reference_penalty),
            beam_similarity_filter=args.beam_similarity_filter,
            beam_signal_similarity_threshold=args.signal_similarity_threshold,
            beam_background_similarity_threshold=args.background_similarity_threshold,
        )
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    model.fit(
        None if args.no_object_features else signal.objects,
        None if args.no_object_features else background.objects,
        signal_slot_mask=None if args.no_object_features else signal.slot_mask,
        background_slot_mask=None if args.no_object_features else background.slot_mask,
        signal_event_features=signal.event_features, background_event_features=background.event_features,
    )
    model.prune_candidates(
        stricter_score_improvement=args.stricter_score_improvement,
        max_reference_acceptance=None if max_acceptance is None else min(1.0, max_acceptance),
        similarity_filter=args.final_similarity_filter,
        signal_similarity_threshold=args.signal_similarity_threshold,
        background_similarity_threshold=args.background_similarity_threshold,
    )
    records = _rule_records(model, signal, background, model.pruning_result_, args, max_acceptance)
    report = reporting._base_report("ERL_Gini_nested_flexible", signal, background, model.event_observables)
    report.update({
        "feature_mode": mode, "native_event_only": mode == "event_only", "dummy_object_input": None,
        "object_observables": [reporting._observable_record(spec) for spec in model.observables],
        "obj_feature_schema": signal.object_schema, "event_feature_schema": signal.feature_schema,
        "native_event_info_used": False, "validity_flags_are_automatic_guards": True,
        "validity_flags_are_guard_only": not model.allow_validity_cuts,
        "validity_guard_policy": "explicit_schema_metadata; invalid values never satisfy associated cuts",
        "evaluation_note": "Acceptances describe training inputs, not independent validation.",
        "ranking_criterion": "score descending (not raw Gini gain)",
        "retention_field_note": "Legacy nested/similarity flags alias final retention; inspect pruning.status.",
        "configuration": {
            "scoring_function": "gini_gain_minus_reference_acceptance_penalty",
            "score_formula": "gini_gain - reference_penalty * reference_acceptance",
            "reference_penalty": args.reference_penalty, "penalty_applied_during_search": True,
            "pruning_stage": "after_candidate_generation",
            "pruning_implementation": "ExistentialRuleList.prune_candidates",
            "stricter_score_improvement": args.stricter_score_improvement,
            "stricter_filter_stage": "after_rate_filter_and_equivalence_merging",
            "stricter_filter_scope": "event_object_and_mixed_rules_with_same_slot_and_minimum_count_semantics",
            "stricter_filter_order": "logically_looser_first_then_final_score_ranking",
            "stricter_filter_comparison": "improvement_must_be_strictly_greater_with_1e-12_roundoff_tolerance",
            "containment_method": "same_column_bounds_and_category_sets_not_training_masks",
            "complexity_measure": "number_of_display_simplified_predicates",
            "max_trigger_rate_khz": args.max_trigger_rate_khz, "input_rate_mhz": args.input_rate_mhz,
            "max_reference_acceptance": max_acceptance,
            "rate_filter_stage": "before_equivalence_merging_and_stricter_pruning",
            "rate_limit_scope": "each_rule_not_combined_OR",
            "rate_estimation": "training_reference_acceptance_times_input_rate_no_confidence_bound",
            "pruning_policy": "rate_then_scoped_equivalence_and_stricter_filter_then_optional_similarity",
            "final_similarity_pruning": args.final_similarity_filter,
            "max_rule_depth": model.max_rule_depth, "beam_width": model.beam_width,
            "max_candidate_rules": model.max_candidate_rules, "max_thresholds": model.max_thresholds,
            "minimum_count": model.minimum_count, "min_signal_efficiency": model.min_signal_efficiency,
            "beam_similarity_filter": args.beam_similarity_filter,
            "beam_signal_similarity_threshold": args.signal_similarity_threshold,
            "beam_background_similarity_threshold": args.background_similarity_threshold,
            "final_signal_similarity_threshold": args.signal_similarity_threshold,
            "final_background_similarity_threshold": args.background_similarity_threshold,
            "object_features_enabled": not args.no_object_features,
            "basic_object_features": args.basic_object_features,
            "object_cut_priority": list(model.object_cut_priority),
            "object_cut_priority_policy": "next_unused_object_feature_only; "
                                          "event_cuts_unrestricted; incomplete_rules_eligible",
            "selected_object_columns": [spec.name for spec in model.observables],
            "allow_validity_cuts": model.allow_validity_cuts,
            "object_validity_guards": {spec.name: spec.validity_column for spec in model.observables
                                       if spec.validity_column is not None},
            "event_validity_guards": {spec.name: spec.validity_column for spec in model.event_observables
                                      if spec.validity_column is not None},
            "event_features_enabled": not args.no_event_features,
            "object_slot_mask_policy": "named pt column > 0; schema-declared per-feature validity guards",
        },
        "summary": {
            "candidate_rule_count": len(records), "pruned_rule_count": len(model.selected_rules_),
            "mixture_enriched_rule_count": sum(r["direction"] == "mixture-enriched" for r in records),
            "reference_enriched_rule_count": sum(r["direction"] == "reference-enriched" for r in records),
            "pruned_rule_ids": [r.rule_id for r in model.selected_rules_],
            "logical_pruning_applicable_rule_count": sum(r["logical_pruning_applicable"] for r in records),
            "equivalent_rules_removed": sum(r["pruning"]["status"] == "equivalent_rule" for r in records),
            "rate_rules_removed": sum(r["pruning"]["status"] == "reference_rate_limit" for r in records),
            "stricter_rules_removed": sum(r["pruning"]["status"] == "stricter_score_improvement" for r in records),
            "similarity_rules_removed": sum(r["pruning"]["status"] == "similarity" for r in records),
        },
        "results": {"rules": records},
    })
    reporting._print_header(report, target)
    print(f"FEATURE MODE: {mode}; object columns={len(model.observables)}; "
          f"event columns={len(model.event_observables)}; no dummy object tensor.")
    if args.basic_object_features:
        print(f"OBJECT FEATURES: basic only ({', '.join(BASIC_OBJECT_COLUMNS)}).")
    if model.object_cut_priority:
        print(f"OBJECT CUT PRIORITY: {' -> '.join(model.object_cut_priority)} -> unrestricted; "
              "incomplete rules eligible; event cuts unrestricted.")
    print("NOTE: native_event_info/pf are not inputs; schema-declared validity flags automatically guard cuts.")
    print("VALIDITY FLAGS: " + ("standalone cuts allowed (explicit opt-in)." if model.allow_validity_cuts
                                else "guard-only; standalone True/False cuts disabled."))
    print(f"SCORING: Gini gain - {args.reference_penalty:g} * reference acceptance (fraction).")
    if max_acceptance is not None:
        print(f"RATE LIMIT: per rule <= {args.max_trigger_rate_khz:g} kHz at {args.input_rate_mhz:g} MHz; "
              f"reference acceptance <= {max_acceptance:.6g}.")
    print("NOTE: rates use training reference acceptance, without confidence bounds or an OR-menu budget.")
    print(f"LOGICAL PRUNING: event/object/mixed rules require improvement > {args.stricter_score_improvement:g}; "
          "same-slot predicates and minimum counts are respected. It can discard lower-rate operating points.")
    print(f"SIMILARITY: beam={args.beam_similarity_filter}; final={args.final_similarity_filter}.")
    print("NOTE: zero observed reference acceptance does not establish zero background rate.")
    enriched = [r for r in records if r["direction"] == "mixture-enriched"]
    reporting._print_section("TOP MIXTURE-ENRICHED CANDIDATES BEFORE PRUNING")
    _print_rules(enriched, args.print_top)
    if args.print_reference_top:
        reporting._print_section("TOP REFERENCE-ENRICHED CANDIDATES (NOT ANOMALY SELECTIONS)")
        _print_rules([r for r in records if r["direction"] == "reference-enriched"], args.print_reference_top)
    reporting._print_section("TOP MIXTURE-ENRICHED RULES AFTER FINAL PRUNING")
    _print_rules([r for r in enriched if r["retained_after_final_pruning"]], args.print_pruned_top)
    reporting._print_section("PRUNING AUDIT")
    by_id = {r["rule_id"]: r for r in records}
    removed = [r for r in records if not r["retained_after_final_pruning"]]
    for record in removed:
        decision = record["pruning"]
        print(f"  rule={record['rule_id']} reason={decision['status']} "
              f"witness_rule={decision['pruned_by_rule_id']} "
              f"retained_witness_rule={decision['retained_witness_rule_id']} "
              f"score_gap={decision['score_gap']} "
              f"estimated_rate_khz={record['estimated_reference_rate_khz']:.6g}")
        if decision["status"] == "stricter_score_improvement":
            print(f"    score_improvement={decision['score_improvement']:.6g}; "
                  f"required_above={args.stricter_score_improvement:g}")
        print(f"    discarded: {record['simplified_rule']}")
        if decision["pruned_by_rule_id"] is not None:
            print(f"    witness:   {by_id[decision['pruned_by_rule_id']]['simplified_rule']}")
    if not removed:
        print("(none)")
    reporting._write_report(target, report, overwrite=args.overwrite_report)
    print(f"\nStructured report saved to: {target}")


if __name__ == "__main__":
    main()
