"""Shared ERL tensor loading and quantitative report construction."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

from set_based_models import EventCut, ObservableSpec, SetInput
from set_based_models.rule_list.generation import _object_priority_progress
from casual_tests import _quantitative_reporting as reporting

BASIC_OBJECT_COLUMNS = ("pt", "eta", "phi", "type_id")
BASIC_PAIR_COLUMNS = tuple(f"{name}_{member}" for member in (1, 2) for name in BASIC_OBJECT_COLUMNS)


@dataclass(frozen=True)
class Dataset(reporting.LoadedDataset):
    object_observables: tuple
    event_observables: tuple
    object_schema: list | None
    feature_schema: list | None
    object_tensor_key: str | None


def _schema(raw, prefix, width, path):
    columns = raw.get(prefix + "_columns")
    if (not isinstance(columns, list) or len(columns) != width
            or not all(isinstance(name, str) and name.strip() for name in columns)):
        raise ValueError(f"{prefix}_columns must name every column in {path}.")
    columns = tuple(name.strip() for name in columns)
    if len(set(columns)) != len(columns):
        raise ValueError(f"Duplicate {prefix}_columns in {path}.")
    key = {"obj": "obj_feature_schema", "pair_obj": "pair_obj_feature_schema",
           "event_feature": "event_feature_schema"}[prefix]
    schema = raw.get(key)
    if schema is None:
        if prefix == "pair_obj":
            raise ValueError(f"{key} is required for particle pairs in {path}.")
        elif prefix == "obj":
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


def load_dataset(path, *, use_objects=True, use_events=True, basic_object_features=False,
                 use_pairs=False):
    """Validate only enabled modalities; never replace NaNs in active features."""
    if not use_objects and not use_events:
        raise ValueError("Enable at least one of object or event features.")
    if basic_object_features and not use_objects:
        raise ValueError("Basic object features require object inputs to be enabled.")
    if use_pairs and not use_objects:
        raise ValueError("Particle pairs require object inputs to be enabled.")
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
    object_key = ("pair_obj" if use_pairs else "obj") if use_objects else None
    if use_objects:
        objects = _tensor(raw, object_key, 3, len(labels), path)
        columns, object_specs, object_schema = _schema(raw, object_key, objects.shape[2], path)
        if use_pairs:
            saved_mask = raw.get("pair_slot_mask")
            if (not isinstance(saved_mask, torch.Tensor) or saved_mask.dtype != torch.bool
                    or saved_mask.layout != torch.strided or saved_mask.shape != objects.shape[:2]):
                raise ValueError(f"pair_slot_mask must be a Boolean tensor with shape {objects.shape[:2]} in {path}.")
            mask = saved_mask.numpy()
            for name in ("pt_1", "pt_2"):
                if name not in columns:
                    raise ValueError(f"pair_obj_columns must contain {name} in {path}.")
                pt = objects[..., columns.index(name)][mask]
                if np.any(~np.isfinite(pt) | (pt <= 0)):
                    raise ValueError(f"Active particle pairs require finite positive {name} in {path}.")
        if basic_object_features:
            selected = BASIC_PAIR_COLUMNS if use_pairs else BASIC_OBJECT_COLUMNS
            missing = [name for name in selected if name not in columns]
            if missing:
                raise ValueError(f"Basic object features require columns {missing} in {path}.")
            indices = [columns.index(name) for name in selected]
            # Keep guards required by selected measurements, but not unrelated extras.
            for index in tuple(indices):
                guard = object_specs[index].validity_column
                if guard is not None:
                    if guard not in columns:
                        raise ValueError(f"Missing validity column {guard!r} in {path}.")
                    guard_index = columns.index(guard)
                    if guard_index not in indices:
                        indices.append(guard_index)
            objects = objects[..., indices]
            object_specs = tuple(object_specs[index] for index in indices)
            if object_schema is not None:
                object_schema = [object_schema[index] for index in indices]
            columns = tuple(columns[index] for index in indices)
        if not use_pairs:
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
                   object_specs, event_specs, object_schema, event_schema, object_key)


def _rule_records(model, signal, background, result, args, max_acceptance):
    ranks = {} if result is None else {rule.rule_id: rank for rank, rule in enumerate(result.retained, 1)}
    is_gini = getattr(args, "score", "gini") == "gini"
    records = []
    candidates = sorted(model.candidate_rules_, key=lambda rule: (-rule.generation_score, rule.rule_id))
    for rank, rule in enumerate(candidates, 1):
        priority_progress = _object_priority_progress(rule.cuts, model.object_cut_priority, model.observables)
        remaining_priority = list(model.object_cut_priority[priority_progress:])
        normalized = rule.cuts if result is None else result.normalized_cuts[rule.rule_id]
        parts = [reporting._predicate_text(cut, model.event_observables, include_scope=True)
                 for cut in normalized if isinstance(cut, EventCut)]
        slots = [cut for cut in normalized if not isinstance(cut, EventCut)]
        if slots:
            slot_name = "pair" if args.pair_tensor else "slot"
            prefix = (f"EXISTS same {slot_name}" if rule.minimum_count == 1
                      else f"COUNT >= {rule.minimum_count} matching {slot_name}s")
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
                "prefix_reference_penalty_term": args.reference_penalty * stats.false_positive_rate if is_gini else None,
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
            "object_cut_priority_complete": not remaining_priority,
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
            "pruned_rank": ranks.get(rule.rule_id),
            "pruning": ({"status": "not_pruned", "pruned_by_rule_id": None,
                         "score_gap": None, "retained_witness_rule_id": None}
                        if result is None else result.decisions[rule.rule_id]),
        })
    return records


def _print_rules(records, limit):
    if not records:
        print("(none)")
        return
    shown = records if limit == 0 else records[:limit]
    for rank, record in enumerate(shown, 1):
        components = (f"gini_gain={record['gini_gain']:.6g} reference_penalty_term={record['reference_penalty_term']:.6g}"
                      if record["reference_penalty_term"] is not None else f"gini_gain_diagnostic={record['gini_gain']:.6g}")
        print(f"\n[{rank:03d}] global_rank={record['rank']} rule={record['rule_id']} "
              f"score={record['score']:.6g} {components} "
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
            components = (f"gain={step['prefix_gini_gain']:.6g} reference_penalty_term={step['prefix_reference_penalty_term']:.6g}"
                          if step["prefix_reference_penalty_term"] is not None else f"gini_gain_diagnostic={step['prefix_gini_gain']:.6g}")
            print(f"    {step['depth']}. [{step['scope']}] {step['display']} | "
                  f"score={step['prefix_score']:.6g} {components} | "
                  f"mixture={step['prefix_signal_accepted']} reference={step['prefix_background_accepted']}")
    if len(shown) < len(records):
        print(f"\n... {len(records) - len(shown)} additional rules are stored in the report.")


def _training_report(model, signal, background, args, mode, max_acceptance, report_path, model_path):
    records = _rule_records(model, signal, background, model.pruning_result_, args, max_acceptance)
    report = reporting._base_report("ERL", signal, background, model.event_observables)
    report.update({
        "stage": "training", "trainer": "casual_tests/train_ERL.py",
        "outputs": {"model": str(model_path), "training_report": str(report_path)},
        "feature_mode": mode, "native_event_only": mode == "event_only", "dummy_object_input": None,
        "object_tensor_key": signal.object_tensor_key,
        "object_observables": [reporting._observable_record(spec) for spec in model.observables],
        "obj_feature_schema": signal.object_schema if signal.object_tensor_key == "obj" else None,
        "pair_obj_feature_schema": signal.object_schema if args.pair_tensor else None,
        "event_feature_schema": signal.feature_schema,
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
            "pruning_policy": ("priority_completion_then_" if args.require_complete_priority else "")
                              + "rate_then_scoped_equivalence_and_stricter_filter_then_optional_similarity",
            "final_similarity_pruning": args.final_similarity_filter,
            "max_rule_depth": model.max_rule_depth, "beam_width": model.beam_width,
            "max_candidate_rules": model.max_candidate_rules, "max_thresholds": model.max_thresholds,
            "minimum_count": model.minimum_count, "min_signal_efficiency": model.min_signal_efficiency,
            "beam_similarity_filter": args.beam_similarity_filter,
            "beam_novelty_weight": model.beam_novelty_weight,
            "novelty_selection_score": "base_score + beta * (1 - max_signal_event_jaccard)",
            "novelty_selection_scope": "beam_and_truncated_candidate_pool_only; intrinsic_scores_unchanged",
            "beam_signal_similarity_threshold": args.signal_similarity_threshold,
            "beam_background_similarity_threshold": args.background_similarity_threshold,
            "final_signal_similarity_threshold": args.signal_similarity_threshold,
            "final_background_similarity_threshold": args.background_similarity_threshold,
            "object_features_enabled": not args.no_object_features,
            "progress": args.progress, "progress_output": "stderr" if args.progress else None,
            "object_tensor_key": signal.object_tensor_key, "pair_tensor": args.pair_tensor,
            "basic_object_features": args.basic_object_features,
            "object_cut_priority": list(model.object_cut_priority),
            "require_complete_priority": args.require_complete_priority,
            "priority_completion_filter_stage": "before_all_final_pruning" if args.require_complete_priority else None,
            "object_cut_priority_policy": "next_listed_object_feature_only; repeated_entries_require_cuts; "
                                          "optional_comparison_direction; event_cuts_unrestricted; "
                                          + ("incomplete_prefixes_searched; complete_rules_required_for_retention"
                                             if args.require_complete_priority else "incomplete_rules_eligible"),
            "selected_object_columns": [spec.name for spec in model.observables],
            "allow_validity_cuts": model.allow_validity_cuts,
            "object_validity_guards": {spec.name: spec.validity_column for spec in model.observables
                                       if spec.validity_column is not None},
            "event_validity_guards": {spec.name: spec.validity_column for spec in model.event_observables
                                      if spec.validity_column is not None},
            "event_features_enabled": not args.no_event_features,
            "object_slot_mask_policy": ("saved Boolean pair_slot_mask; both members have positive finite PT; "
                                        "schema-declared per-feature validity guards" if args.pair_tensor else
                                        "named pt column > 0; schema-declared per-feature validity guards"),
        },
        "summary": {
            "final_pruning_applied": model.pruning_result_ is not None,
            "candidate_rule_count": len(records), "pruned_rule_count": len(model.selected_rules_),
            "mixture_enriched_rule_count": sum(r["direction"] == "mixture-enriched" for r in records),
            "reference_enriched_rule_count": sum(r["direction"] == "reference-enriched" for r in records),
            "pruned_rule_ids": [r.rule_id for r in model.selected_rules_],
            "logical_pruning_applicable_rule_count": sum(r["logical_pruning_applicable"] for r in records),
            "equivalent_rules_removed": sum(r["pruning"]["status"] == "equivalent_rule" for r in records),
            "rate_rules_removed": sum(r["pruning"]["status"] == "reference_rate_limit" for r in records),
            "stricter_rules_removed": sum(r["pruning"]["status"] == "stricter_score_improvement" for r in records),
            "incomplete_priority_rules_removed": sum(r["pruning"]["status"] == "incomplete_object_priority" for r in records),
            "similarity_rules_removed": sum(r["pruning"]["status"] == "similarity" for r in records),
        },
        "results": {"rules": records},
    })
    for record in report["inputs"].values():
        record["object_tensor_key"] = signal.object_tensor_key
    return report
