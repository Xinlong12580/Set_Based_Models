"""Event-only training with native ERL inputs and legacy EDT dummy slots.

The original quantitative runners retain their object-input behavior. Real
``obj``/``pf`` values are never inspected. Native event measurements with a
separate validity mask are not automatically merged into ``event_features``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

from set_based_models import (
    EventCut, ExistentialDecisionTree, ExistentialRuleList, ObservableSpec,
    SetInput, gini_gain,
)
from casual_tests import _quantitative_reporting as reporting


DUMMY_OBSERVABLES = (ObservableSpec("unused_dummy", "continuous"),)


@dataclass(frozen=True)
class EventOnlyDataset(reporting.LoadedDataset):
    event_observables: tuple[ObservableSpec, ...]
    feature_schema: list[dict] | None


def load_event_dataset(path: str | Path, *, native_event_only=False) -> EventOnlyDataset:
    """Read event inputs; ERL uses zero slots, while EDT keeps legacy dummy slots."""
    path = Path(path).expanduser().resolve()
    raw = torch.load(path, map_location="cpu", weights_only=True)
    required = {"event_features", "event_feature_columns", "label"}
    if not isinstance(raw, dict) or not required <= raw.keys():
        missing = required - raw.keys() if isinstance(raw, dict) else required
        raise ValueError(f"{path} is missing required keys: {sorted(missing)}")
    events, labels, columns = (raw[key] for key in
                               ("event_features", "label", "event_feature_columns"))
    if (not isinstance(events, torch.Tensor) or events.ndim != 2
            or not events.shape[0] or not events.shape[1]
            or events.is_complex() or events.layout != torch.strided):
        raise ValueError(f"event_features must be a nonempty, real, dense 2D tensor in {path}.")
    if (not isinstance(labels, torch.Tensor) or labels.shape != (events.shape[0],)
            or labels.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}
            or labels.layout != torch.strided):
        raise ValueError(f"label must contain one integer process label per event in {path}.")
    if (not isinstance(columns, list) or len(columns) != events.shape[1]
            or not all(isinstance(name, str) and name.strip() for name in columns)):
        raise ValueError(f"event_feature_columns must name every column in {path}.")
    columns = tuple(name.strip() for name in columns)
    if len(set(columns)) != len(columns):
        raise ValueError(f"Duplicate event_feature_columns in {path}.")

    schema = raw.get("event_feature_schema")
    if schema is None:
        # Legacy event files have only nonnegative multiplicities.
        specs = tuple(ObservableSpec(name, "ordinal") for name in columns)
    else:
        if (not isinstance(schema, list) or len(schema) != len(columns)
                or not all(isinstance(item, dict) for item in schema)):
            raise ValueError(f"event_feature_schema must describe every column in {path}.")
        specs = []
        for name, item in zip(columns, schema):
            if item.get("name") != name or "kind" not in item:
                raise ValueError(f"event_feature_schema names/order do not match columns in {path}.")
            specs.append(reporting._observable_from_schema(name, item, validity_guards=native_event_only))
        specs = tuple(specs)
        # Definitions are included in the report and compared across classes.
        json.dumps(schema, allow_nan=False)

    values = events.detach().to(dtype=torch.float64).numpy()
    objects = None if native_event_only else np.zeros((len(values), 1, 1), dtype=np.float64)
    mask = None if native_event_only else np.zeros((len(values), 1), dtype=bool)
    # Reuse the core validation for finite values and ordinal/category codes.
    validated = SetInput.from_arrays(
        objects, () if native_event_only else DUMMY_OBSERVABLES, slot_mask=mask,
        event_features=values, event_observables=specs,
    )
    for index, name in enumerate(columns):
        is_count = schema is None or schema[index].get("units") == "count"
        if is_count and np.any(values[:, index][validated.event_value_mask[:, index]] < 0):
            raise ValueError(f"Multiplicity {name!r} must be nonnegative in {path}.")
    return EventOnlyDataset(
        path=path, objects=validated.X, slot_mask=validated.slot_mask, labels=labels.numpy(),
        event_features=validated.event_features, event_columns=columns,
        event_observables=specs, feature_schema=schema,
    )


def _parser(description, *, model):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--signal", type=Path, required=True, help="Anomalous/mixture .pt input.")
    parser.add_argument("--background", type=Path, required=True, help="Reference/background .pt input.")
    parser.add_argument("--report", type=Path, help="Plot-compatible JSON report, optionally with .txt suffix.")
    parser.add_argument("--overwrite-report", action="store_true")
    parser.add_argument("--print-top", type=reporting._nonnegative_int,
                        default=20 if model == "EDT" else 10,
                        help="Number of ranked results to print; 0 prints all.")
    if model == "ERL":
        parser.add_argument("--allow-validity-cuts", action="store_true",
                            help="Opt into standalone True/False cuts on validity flags (default: guard-only).")
    return parser


def _load_pair(parser, args, model_name, *, native_event_only=False):
    try:
        signal, background = (load_event_dataset(path, native_event_only=native_event_only)
                              for path in (args.signal, args.background))
        if signal.path.samefile(background.path):
            raise ValueError("Signal and background inputs must be different files.")
        if (signal.event_observables != background.event_observables
                or signal.feature_schema != background.feature_schema):
            raise ValueError("Signal and background must have matching event columns, order, and schemas.")
        if args.report is not None:
            requested = args.report.expanduser().absolute()
        else:
            sample = signal.path.parent.name
            requested = (Path.cwd() / "reports" /
                         f"{model_name}_{sample}_{signal.path.stem}_event_only.txt")
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


def _base_report(model_name, signal, background, *, allow_validity_cuts=None):
    report = reporting._base_report(model_name, signal, background, signal.event_observables)
    # The old helper describes its real object schema. Replace that description
    # explicitly; no dummy variable is an enabled physics input.
    report.update({
        "feature_mode": "event_only", "object_observables": [],
        "event_feature_schema": signal.feature_schema,
        "validity_flags_are_automatic_guards": signal.objects.shape[1] == 0,
        "event_validity_guards": {spec.name: spec.validity_column for spec in signal.event_observables
                                  if spec.validity_column is not None},
        "dummy_object_input": {
            "observable": "unused_dummy", "kind": "continuous", "value": 0.0,
            "slots_per_event": 1, "valid_slots_per_event": 0,
            "real_object_features_used": False,
        },
        "native_event_info_used": False,
        "evaluation_note": "Acceptances describe the training inputs, not an independent validation set.",
    })
    if signal.objects.shape[1] == 0:
        report["native_event_only"] = True
        report["dummy_object_input"] = None
    if allow_validity_cuts is not None:
        report["validity_flags_are_guard_only"] = not allow_validity_cuts
    return report


def _print_header(report, target):
    reporting._print_header(report, target)
    if report.get("native_event_only"):
        print("FEATURE MODE: native event-only; real obj/pf ignored; no object tensor supplied to ERL.")
    else:
        print("FEATURE MODE: event-only; real obj/pf ignored; dummy slots all masked out.")
    print("NOTE: separate native_event_info and its validity masks are not model inputs.")
    if report.get("validity_flags_are_automatic_guards"):
        print("VALIDITY: schema-declared flags automatically guard event cuts; guards add no search depth.")
    if "validity_flags_are_guard_only" in report:
        print("VALIDITY FLAGS: " + ("guard-only; standalone True/False cuts disabled."
              if report["validity_flags_are_guard_only"] else "standalone cuts allowed (explicit opt-in)."))


def run_edt_cli(argv=None, *, description):
    parser = _parser(description, model="EDT")
    args = parser.parse_args(argv)
    signal, background, target = _load_pair(parser, args, "EDT")
    tree = ExistentialDecisionTree(
        DUMMY_OBSERVABLES, event_observables=signal.event_observables,
        max_depth=6, gain_function=gini_gain, min_samples_leaf=20,
    )
    tree.fit(
        signal.objects, background.objects,
        signal_slot_mask=signal.slot_mask, background_slot_mask=background.slot_mask,
        signal_event_features=signal.event_features, background_event_features=background.event_features,
    )
    if any(node.cut is not None and not isinstance(node.cut, EventCut) for node in tree.nodes_):
        raise RuntimeError("Unexpected object-level cut in an event-only tree.")
    selected_leaves = [tree.apply(data.objects, slot_mask=data.slot_mask,
                                 event_features=data.event_features) for data in (signal, background)]
    leaves = sorted((node for node in tree.nodes_ if node.is_leaf),
                    key=lambda node: node.score if node.score is not None else -np.inf, reverse=True)
    records = []
    for rank, leaf in enumerate(leaves, start=1):
        steps = []
        for step in tree.cut_chain(leaf.node_id):
            source = tree.get_node(step.source_node_id)
            steps.append({
                "source_node_id": step.source_node_id, "branch": step.branch.value, "scope": "event",
                "cut": reporting._cut_record(step.cut, tree.event_observables),
                "selected_condition": reporting._selected_condition(
                    step.cut, tree.event_observables, step.branch, None),
                "source_split_gain": source.statistics.split_gain if source.statistics else None,
            })
        records.append({
            "rank": rank, "leaf_id": leaf.node_id, "depth": leaf.depth,
            "leaf_signal_fraction": leaf.score,
            "predicted_class_at_default_threshold": int(
                leaf.score is not None and leaf.score >= tree.decision_threshold),
            "metrics": reporting._selection_metrics(
                selected_leaves[0] == leaf.node_id, selected_leaves[1] == leaf.node_id,
                signal.labels, background.labels),
            "path": steps,
        })
    predictions = [tree.predict(data.objects, slot_mask=data.slot_mask,
                                event_features=data.event_features).astype(bool)
                   for data in (signal, background)]
    report = _base_report("EDT", signal, background)
    report.update({
        "configuration": {
            "gain_function": "gini_gain", "max_depth": 6, "min_samples_leaf": 20,
            "decision_threshold": tree.decision_threshold, "minimum_count": tree.aggregation.minimum_count,
            "max_thresholds": tree.max_thresholds, "object_features_enabled": False,
        },
        "summary": {
            "node_count": len(tree.nodes_), "leaf_count": len(leaves),
            "default_prediction": reporting._selection_metrics(
                *predictions, signal.labels, background.labels),
        },
        "results": {"leaves": records},
    })
    _print_header(report, target)
    reporting._print_selection("DEFAULT TREE SELECTION", report["summary"]["default_prediction"])
    reporting._print_section("LEAVES RANKED BY MIXTURE FRACTION")
    displayed = records if args.print_top == 0 else records[:args.print_top]
    for record in displayed:
        reporting._print_edt_leaf(record)
    if len(displayed) < len(records):
        print(f"\n... {len(records) - len(displayed)} additional leaves are stored in the report.")
    reporting._write_report(target, report, overwrite=args.overwrite_report)
    print(f"\nStructured report saved to: {target}")


def run_erl_cli(argv=None, *, description):
    parser = _parser(description, model="ERL")
    parser.add_argument("--print-reference-top", type=reporting._nonnegative_int, default=5,
                        help="Reference-enriched rules to print; 0 prints none.")
    parser.add_argument("--print-pruned-top", type=reporting._nonnegative_int, default=10,
                        help="Pruned mixture-enriched rules to print; 0 prints all.")
    args = parser.parse_args(argv)
    signal, background, target = _load_pair(parser, args, "ERL_Gini", native_event_only=True)
    model = ExistentialRuleList(
        event_observables=signal.event_observables,
        max_rule_depth=3, beam_width=40, max_candidate_rules=200, max_thresholds=32,
        minimum_count=1, min_signal_efficiency=0.01,
        allow_validity_cuts=args.allow_validity_cuts,
        beam_similarity_filter=True, scoring_function=reporting.gini_rule_score,
        beam_signal_similarity_threshold=reporting.SIGNAL_SIMILARITY_THRESHOLD,
        beam_background_similarity_threshold=reporting.BACKGROUND_SIMILARITY_THRESHOLD,
    ).fit(
        signal_event_features=signal.event_features, background_event_features=background.event_features,
    )
    if any(not isinstance(cut, EventCut) for rule in model.candidate_rules_ for cut in rule.cuts):
        raise RuntimeError("Unexpected object-level cut in an event-only rule list.")
    pruned = model.prune_candidates(
        signal_similarity_threshold=reporting.SIGNAL_SIMILARITY_THRESHOLD,
        background_similarity_threshold=reporting.BACKGROUND_SIMILARITY_THRESHOLD,
    )
    pruned_ranks = {rule.rule_id: rank for rank, rule in enumerate(pruned, start=1)}
    records = []
    for rank, rule in enumerate(model.candidate_rules_, start=1):
        metrics = reporting._selection_metrics(rule.signal_selection, rule.background_selection,
                                               signal.labels, background.labels)
        simplified, effective_depth = reporting._simplified_rule_text(rule, model)
        steps = []
        for step in rule.steps:
            stats = step.statistics
            steps.append({
                "depth": step.depth, "scope": "event",
                "cut": reporting._cut_record(step.cut, model.event_observables),
                "display": reporting._predicate_text(step.cut, model.event_observables, include_scope=True),
                "prefix_gini_gain": step.generation_score,
                "prefix_signal_accepted": stats.signal_accepted,
                "prefix_background_accepted": stats.background_accepted,
                "prefix_signal_acceptance": stats.true_positive_rate,
                "prefix_reference_acceptance": stats.false_positive_rate,
            })
        records.append({
            "rank": rank, "rule_id": rule.rule_id, "depth": rule.depth,
            "effective_depth_after_simplification": effective_depth,
            "gini_gain": rule.generation_score, "direction": metrics["direction"],
            "raw_rule": rule.to_text(), "simplified_rule": simplified,
            "retained_after_similarity_pruning": rule.rule_id in pruned_ranks,
            "pruned_rank": pruned_ranks.get(rule.rule_id), "metrics": metrics, "steps": steps,
        })
    report = _base_report("ERL_Gini", signal, background, allow_validity_cuts=args.allow_validity_cuts)
    enriched = [record for record in records if record["direction"] == "mixture-enriched"]
    reference = [record for record in records if record["direction"] == "reference-enriched"]
    report.update({
        "configuration": {
            "scoring_function": "weighted_gini_gain_of_rule_vs_complement",
            "max_rule_depth": 3, "beam_width": 40, "max_candidate_rules": 200,
            "max_thresholds": 32, "minimum_count": 1, "min_signal_efficiency": 0.01,
            "beam_similarity_filter": True,
            "beam_signal_similarity_threshold": reporting.SIGNAL_SIMILARITY_THRESHOLD,
            "beam_background_similarity_threshold": reporting.BACKGROUND_SIMILARITY_THRESHOLD,
            "object_features_enabled": False,
            "allow_validity_cuts": args.allow_validity_cuts,
        },
        "summary": {
            "candidate_rule_count": len(records), "pruned_rule_count": len(pruned),
            "mixture_enriched_rule_count": len(enriched), "reference_enriched_rule_count": len(reference),
            "pruned_rule_ids": [rule.rule_id for rule in pruned],
        },
        "results": {"rules": records},
    })
    _print_header(report, target)
    reporting._print_section("TOP MIXTURE-ENRICHED RULES")
    reporting._print_rule_records(enriched, args.print_top)
    if args.print_reference_top:
        reporting._print_section("TOP REFERENCE-ENRICHED RULES (NOT ANOMALY SELECTIONS)")
        reporting._print_rule_records(reference, args.print_reference_top)
    reporting._print_section("TOP PRUNED MIXTURE-ENRICHED RULES")
    reporting._print_rule_records(
        [record for record in enriched if record["retained_after_similarity_pruning"]], args.print_pruned_top)
    reporting._write_report(target, report, overwrite=args.overwrite_report)
    print(f"\nStructured report saved to: {target}")
