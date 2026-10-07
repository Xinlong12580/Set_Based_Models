"""Shared implementation for the quantitative EDT and ERL runners.

The public entry points are the four ``run_*_quantitative.py`` scripts in this
directory.  Reports use ordinary JSON despite their ``.txt`` extension, so
they remain readable in a text editor and can be loaded with ``json.load``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from set_based_models import (
    Branch,
    CategoryCut,
    Comparison,
    EventCut,
    ExistentialDecisionTree,
    ExistentialRuleList,
    ObservableKind,
    ObservableSpec,
    PartitionStatistics,
    RuleStatistics,
    SplitStatistics,
    ThresholdCut,
    gini_gain,
)


ROOT = Path(__file__).resolve().parent.parent
OBJECT_COLUMNS = ["pt", "eta", "phi", "type_id"]
OBSERVABLES = (
    ObservableSpec("pt", "continuous"),
    ObservableSpec("eta", "continuous"),
    ObservableSpec("phi", "continuous"),
    ObservableSpec(
        "type_id",
        "categorical",
        {0: "jet", 1: "muon", 2: "electron", 3: "photon", 4: "met"},
    ),
)
SIGNAL_SIMILARITY_THRESHOLD = 0.1
BACKGROUND_SIMILARITY_THRESHOLD = 0.1


@dataclass(frozen=True)
class LoadedDataset:
    path: Path
    objects: np.ndarray
    slot_mask: np.ndarray
    labels: np.ndarray
    event_features: np.ndarray | None
    event_columns: tuple[str, ...]


def gini_rule_score(statistics: RuleStatistics) -> float:
    """Adapt ERL pass/fail statistics to the EDT weighted-Gini objective."""
    parent = PartitionStatistics(
        statistics.signal_total,
        statistics.background_total,
        statistics.signal_weight_total,
        statistics.background_weight_total,
    )
    positive = PartitionStatistics(
        statistics.signal_accepted,
        statistics.background_accepted,
        statistics.signal_weight_accepted,
        statistics.background_weight_accepted,
    )
    negative = PartitionStatistics(
        parent.signal_count - positive.signal_count,
        parent.background_count - positive.background_count,
        parent.signal_weight - positive.signal_weight,
        parent.background_weight - positive.background_weight,
    )
    return gini_gain(SplitStatistics(parent, positive, negative, root=parent))


def load_dataset(path: str | Path, *, require_event_features: bool) -> LoadedDataset:
    path = Path(path).expanduser().resolve()
    raw = torch.load(path, map_location="cpu", weights_only=True)
    required = {"obj", "label", "obj_columns"}
    if require_event_features:
        required |= {"event_features", "event_feature_columns"}
    if not isinstance(raw, dict) or not required <= raw.keys():
        missing = required - raw.keys() if isinstance(raw, dict) else required
        raise ValueError(f"{path} is missing required keys: {sorted(missing)}")
    if raw["obj_columns"] != OBJECT_COLUMNS:
        raise ValueError(f"Unexpected object columns in {path}: {raw['obj_columns']!r}")

    objects = raw["obj"]
    labels = raw["label"]
    if (
        not isinstance(objects, torch.Tensor)
        or objects.ndim != 3
        or objects.shape[1:] != (23, 4)
    ):
        shape = tuple(objects.shape) if isinstance(objects, torch.Tensor) else type(objects).__name__
        raise ValueError(f"Expected obj shape (events, 23, 4) in {path}, got {shape}")
    if (
        not isinstance(labels, torch.Tensor)
        or labels.ndim != 1
        or labels.shape[0] != objects.shape[0]
    ):
        raise ValueError(f"label must contain one entry per obj row in {path}.")

    event_features = None
    event_columns: tuple[str, ...] = ()
    if require_event_features:
        events = raw["event_features"]
        columns = raw["event_feature_columns"]
        if not isinstance(events, torch.Tensor) or events.ndim != 2:
            raise ValueError(f"event_features must be a two-dimensional tensor in {path}.")
        if events.shape[0] != objects.shape[0]:
            raise ValueError(f"event_features must contain one row per obj row in {path}.")
        if (
            not isinstance(columns, list)
            or not columns
            or not all(isinstance(name, str) and name.strip() for name in columns)
            or len(columns) != events.shape[1]
        ):
            raise ValueError(
                f"event_feature_columns must name every event_features column in {path}."
            )
        normalized_columns = tuple(name.strip() for name in columns)
        if len(set(normalized_columns)) != len(normalized_columns):
            raise ValueError(f"event_feature_columns contains duplicate names in {path}.")
        if not bool(torch.isfinite(events).all()):
            raise ValueError(f"event_features must contain only finite values in {path}.")
        if not bool((events == torch.floor(events)).all()):
            raise ValueError(f"event_features must contain integer-valued multiplicities in {path}.")
        if not bool((events >= 0).all()):
            raise ValueError(f"event_features must contain non-negative multiplicities in {path}.")
        event_features = events.numpy()
        event_columns = normalized_columns

    objects = torch.nan_to_num(objects, nan=0.0, posinf=0.0, neginf=0.0)
    return LoadedDataset(
        path=path,
        objects=objects.numpy(),
        slot_mask=(objects[..., 0] > 0).numpy(),
        labels=labels.numpy(),
        event_features=event_features,
        event_columns=event_columns,
    )


def run_edt_cli(
    argv=None,
    *,
    require_event_features: bool,
    description: str,
) -> None:
    parser = _parser(description, model="EDT", require_event_features=require_event_features)
    args = parser.parse_args(argv)
    signal, background, event_observables = _load_pair(parser, args, require_event_features)
    report_path = _report_path(args.report, "EDT", signal.path)
    _check_report_target(parser, report_path, args.overwrite_report)

    tree = ExistentialDecisionTree(
        OBSERVABLES,
        event_observables=event_observables,
        max_depth=6,
        gain_function=gini_gain,
        min_samples_leaf=20,
    )
    tree.fit(
        signal.objects,
        background.objects,
        signal_slot_mask=signal.slot_mask,
        background_slot_mask=background.slot_mask,
        signal_event_features=signal.event_features,
        background_event_features=background.event_features,
    )

    signal_leaf = tree.apply(
        signal.objects,
        slot_mask=signal.slot_mask,
        event_features=signal.event_features,
    )
    background_leaf = tree.apply(
        background.objects,
        slot_mask=background.slot_mask,
        event_features=background.event_features,
    )
    leaves = sorted(
        (node for node in tree.nodes_ if node.is_leaf),
        key=lambda node: node.score if node.score is not None else float("-inf"),
        reverse=True,
    )
    leaf_records = []
    for rank, leaf in enumerate(leaves, start=1):
        signal_selection = signal_leaf == leaf.node_id
        background_selection = background_leaf == leaf.node_id
        metrics = _selection_metrics(
            signal_selection,
            background_selection,
            signal.labels,
            background.labels,
        )
        steps = []
        for path_step in tree.cut_chain(leaf.node_id):
            is_event = isinstance(path_step.cut, EventCut)
            schema = tree.event_observables if is_event else tree.observables
            source = tree.get_node(path_step.source_node_id)
            steps.append(
                {
                    "source_node_id": path_step.source_node_id,
                    "branch": path_step.branch.value,
                    "scope": "event" if is_event else "slot",
                    "cut": _cut_record(path_step.cut, schema),
                    "selected_condition": _selected_condition(
                        path_step.cut,
                        schema,
                        path_step.branch,
                        path_step.aggregation.minimum_count
                        if path_step.aggregation is not None
                        else None,
                    ),
                    "source_split_gain": (
                        source.statistics.split_gain if source.statistics is not None else None
                    ),
                }
            )
        leaf_records.append(
            {
                "rank": rank,
                "leaf_id": leaf.node_id,
                "depth": leaf.depth,
                "leaf_signal_fraction": leaf.score,
                "predicted_class_at_default_threshold": int(
                    leaf.score is not None and leaf.score >= tree.decision_threshold
                ),
                "metrics": metrics,
                "path": steps,
            }
        )

    signal_prediction = tree.predict(
        signal.objects,
        slot_mask=signal.slot_mask,
        event_features=signal.event_features,
    ).astype(bool)
    background_prediction = tree.predict(
        background.objects,
        slot_mask=background.slot_mask,
        event_features=background.event_features,
    ).astype(bool)
    report = _base_report("EDT", signal, background, event_observables)
    report.update(
        {
            "configuration": {
                "gain_function": "gini_gain",
                "max_depth": 6,
                "min_samples_leaf": 20,
                "decision_threshold": tree.decision_threshold,
                "minimum_count": tree.aggregation.minimum_count,
                "max_thresholds": tree.max_thresholds,
            },
            "summary": {
                "node_count": len(tree.nodes_),
                "leaf_count": len(leaves),
                "default_prediction": _selection_metrics(
                    signal_prediction,
                    background_prediction,
                    signal.labels,
                    background.labels,
                ),
            },
            "results": {"leaves": leaf_records},
        }
    )

    _print_header(report, report_path)
    _print_selection("DEFAULT TREE SELECTION", report["summary"]["default_prediction"])
    _print_section("LEAVES RANKED BY MIXTURE FRACTION")
    limit = len(leaf_records) if args.print_top == 0 else min(args.print_top, len(leaf_records))
    for record in leaf_records[:limit]:
        _print_edt_leaf(record)
    if limit < len(leaf_records):
        print(f"\n... {len(leaf_records) - limit} additional leaves are stored in the report.")
    _write_report(report_path, report, overwrite=args.overwrite_report)
    print(f"\nStructured report saved to: {report_path}")


def run_erl_cli(
    argv=None,
    *,
    require_event_features: bool,
    description: str,
) -> None:
    parser = _parser(description, model="ERL", require_event_features=require_event_features)
    parser.add_argument(
        "--print-reference-top",
        type=_nonnegative_int,
        default=5,
        help="Number of reference-enriched rules to print; 0 prints none.",
    )
    parser.add_argument(
        "--print-pruned-top",
        type=_nonnegative_int,
        default=10,
        help="Number of pruned mixture-enriched rules to print; 0 prints all.",
    )
    args = parser.parse_args(argv)
    signal, background, event_observables = _load_pair(parser, args, require_event_features)
    report_path = _report_path(args.report, "ERL_Gini", signal.path)
    _check_report_target(parser, report_path, args.overwrite_report)

    model = ExistentialRuleList(
        OBSERVABLES,
        event_observables=event_observables,
        max_rule_depth=3,
        beam_width=40,
        max_candidate_rules=200,
        max_thresholds=32,
        minimum_count=1,
        beam_similarity_filter=True,
        scoring_function=gini_rule_score,
        beam_signal_similarity_threshold=SIGNAL_SIMILARITY_THRESHOLD,
        min_signal_efficiency=0.01,
        beam_background_similarity_threshold=BACKGROUND_SIMILARITY_THRESHOLD,
    )
    model.fit(
        signal.objects,
        background.objects,
        signal_slot_mask=signal.slot_mask,
        background_slot_mask=background.slot_mask,
        signal_event_features=signal.event_features,
        background_event_features=background.event_features,
    )
    pruned_rules = model.prune_candidates(
        signal_similarity_threshold=SIGNAL_SIMILARITY_THRESHOLD,
        background_similarity_threshold=BACKGROUND_SIMILARITY_THRESHOLD,
    )
    pruned_ids = {rule.rule_id for rule in pruned_rules}
    pruned_ranks = {rule.rule_id: rank for rank, rule in enumerate(pruned_rules, start=1)}

    rule_records = []
    for rank, rule in enumerate(model.candidate_rules_, start=1):
        assert rule.signal_selection is not None and rule.background_selection is not None
        steps = []
        for step in rule.steps:
            is_event = isinstance(step.cut, EventCut)
            schema = model.event_observables if is_event else model.observables
            step_stats = step.statistics
            steps.append(
                {
                    "depth": step.depth,
                    "scope": "event" if is_event else "slot",
                    "cut": _cut_record(step.cut, schema),
                    "display": _predicate_text(step.cut, schema, include_scope=is_event),
                    "prefix_gini_gain": step.generation_score,
                    "prefix_signal_accepted": step_stats.signal_accepted,
                    "prefix_background_accepted": step_stats.background_accepted,
                    "prefix_signal_acceptance": step_stats.true_positive_rate,
                    "prefix_reference_acceptance": step_stats.false_positive_rate,
                }
            )
        metrics = _selection_metrics(
            rule.signal_selection,
            rule.background_selection,
            signal.labels,
            background.labels,
        )
        simplified, effective_depth = _simplified_rule_text(rule, model)
        rule_records.append(
            {
                "rank": rank,
                "rule_id": rule.rule_id,
                "depth": rule.depth,
                "effective_depth_after_simplification": effective_depth,
                "gini_gain": rule.generation_score,
                "direction": metrics["direction"],
                "raw_rule": rule.to_text(),
                "simplified_rule": simplified,
                "retained_after_similarity_pruning": rule.rule_id in pruned_ids,
                "pruned_rank": pruned_ranks.get(rule.rule_id),
                "metrics": metrics,
                "steps": steps,
            }
        )

    report = _base_report("ERL_Gini", signal, background, event_observables)
    report.update(
        {
            "configuration": {
                "scoring_function": "weighted_gini_gain_of_rule_vs_complement",
                "max_rule_depth": 3,
                "beam_width": 40,
                "max_candidate_rules": 200,
                "max_thresholds": 32,
                "minimum_count": 1,
                "min_signal_efficiency": 0.01,
                "beam_similarity_filter": True,
                "beam_signal_similarity_threshold": SIGNAL_SIMILARITY_THRESHOLD,
                "beam_background_similarity_threshold": BACKGROUND_SIMILARITY_THRESHOLD,
            },
            "summary": {
                "candidate_rule_count": len(rule_records),
                "pruned_rule_count": len(pruned_rules),
                "mixture_enriched_rule_count": sum(
                    record["direction"] == "mixture-enriched" for record in rule_records
                ),
                "reference_enriched_rule_count": sum(
                    record["direction"] == "reference-enriched" for record in rule_records
                ),
                "pruned_rule_ids": [rule.rule_id for rule in pruned_rules],
            },
            "results": {"rules": rule_records},
        }
    )

    _print_header(report, report_path)
    enriched = [record for record in rule_records if record["direction"] == "mixture-enriched"]
    reference = [record for record in rule_records if record["direction"] == "reference-enriched"]
    pruned_enriched = [
        record for record in enriched if record["retained_after_similarity_pruning"]
    ]
    _print_section("TOP MIXTURE-ENRICHED RULES")
    _print_rule_records(enriched, args.print_top)
    if args.print_reference_top:
        _print_section("TOP REFERENCE-ENRICHED RULES (NOT ANOMALY SELECTIONS)")
        _print_rule_records(reference, args.print_reference_top)
    _print_section("TOP PRUNED MIXTURE-ENRICHED RULES")
    _print_rule_records(pruned_enriched, args.print_pruned_top)
    _write_report(report_path, report, overwrite=args.overwrite_report)
    print(f"\nStructured report saved to: {report_path}")


def _parser(description: str, *, model: str, require_event_features: bool):
    default_dir = ROOT / ("data/with_event_multiplicities" if require_event_features else "data")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--signal",
        type=Path,
        default=default_dir / "anomalous.pt",
        help="Signal/mixture .pt input.",
    )
    parser.add_argument(
        "--background",
        type=Path,
        default=default_dir / "reference.pt",
        help="Reference/background .pt input.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help=(
            "JSON text report path. By default, derives a .txt filename from "
            "the model and signal-input directory."
        ),
    )
    parser.add_argument(
        "--overwrite-report",
        action="store_true",
        help="Permit replacement of an existing report file.",
    )
    parser.add_argument(
        "--print-top",
        type=_nonnegative_int,
        default=20 if model == "EDT" else 10,
        help="Number of ranked results to print; 0 prints all.",
    )
    return parser


def _load_pair(parser, args, require_event_features):
    try:
        signal = load_dataset(args.signal, require_event_features=require_event_features)
        background = load_dataset(args.background, require_event_features=require_event_features)
        if signal.path == background.path:
            raise ValueError("Signal and background inputs must be different files.")
        if signal.event_columns != background.event_columns:
            raise ValueError(
                "Signal and background event_feature_columns must match exactly; "
                f"received {list(signal.event_columns)!r} and "
                f"{list(background.event_columns)!r}."
            )
        event_observables = tuple(
            ObservableSpec(name, "ordinal") for name in signal.event_columns
        )
        return signal, background, event_observables
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))


def _base_report(model_name, signal, background, event_observables):
    return {
        "format": "set_based_models_quantitative_report",
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "class_semantics": {
            "signal": "Every row of the signal/mixture input, regardless of stored label.",
            "background": "Every row of the reference/background input.",
            "purity_warning": (
                "mixture_label_purity is training-class purity, not injected-process purity."
            ),
        },
        "inputs": {
            "signal": _dataset_record(signal),
            "background": _dataset_record(background),
        },
        "object_observables": [_observable_record(spec) for spec in OBSERVABLES],
        "event_observables": [_observable_record(spec) for spec in event_observables],
    }


def _dataset_record(dataset):
    return {
        "path": str(dataset.path),
        "event_count": int(len(dataset.objects)),
        "object_shape": list(dataset.objects.shape),
        "stored_label_counts": _label_counts(dataset.labels),
        "event_feature_shape": (
            None if dataset.event_features is None else list(dataset.event_features.shape)
        ),
        "event_feature_columns": list(dataset.event_columns),
    }


def _observable_record(spec):
    record = {
        "name": spec.name,
        "kind": spec.kind.value,
        "categories": None if spec.categories is None else dict(spec.categories),
    }
    if spec.validity_column is not None:
        record["validity_column"] = spec.validity_column
    return record


def _observable_from_schema(name, item, *, validity_guards=True):
    """Read explicit guard metadata, including the raw-converter event format."""
    definition = item.get("definition")
    definition = definition if isinstance(definition, dict) else {}
    guards = [mapping[key] for mapping in (item, definition)
              for key in ("validity_column", "validity_feature")
              if mapping.get(key) is not None] if validity_guards else []
    if guards and any(guard != guards[0] for guard in guards):
        raise ValueError(f"Conflicting validity metadata for {name!r}")
    return ObservableSpec(name, item["kind"], item.get("categories"),
                          validity_column=guards[0] if guards else None)


def _selection_metrics(signal_selection, background_selection, signal_labels, background_labels):
    signal_selection = np.asarray(signal_selection, dtype=bool)
    background_selection = np.asarray(background_selection, dtype=bool)
    signal_accepted = int(signal_selection.sum())
    background_accepted = int(background_selection.sum())
    signal_total = int(signal_selection.size)
    background_total = int(background_selection.size)
    signal_acceptance = signal_accepted / signal_total if signal_total else 0.0
    background_acceptance = background_accepted / background_total if background_total else 0.0
    accepted_total = signal_accepted + background_accepted
    purity = signal_accepted / accepted_total if accepted_total else 0.0
    if signal_acceptance > background_acceptance:
        direction = "mixture-enriched"
    elif signal_acceptance < background_acceptance:
        direction = "reference-enriched"
    else:
        direction = "equal-acceptance"
    return {
        "signal_accepted": signal_accepted,
        "signal_total": signal_total,
        "background_accepted": background_accepted,
        "background_total": background_total,
        "signal_acceptance": signal_acceptance,
        "background_acceptance": background_acceptance,
        "acceptance_difference": signal_acceptance - background_acceptance,
        "acceptance_ratio": (
            signal_acceptance / background_acceptance if background_acceptance else None
        ),
        "mixture_label_purity": purity,
        "direction": direction,
        "selected_signal_stored_label_counts": _label_counts(signal_labels[signal_selection]),
        "selected_background_stored_label_counts": _label_counts(
            background_labels[background_selection]
        ),
    }


def _label_counts(labels):
    values, counts = np.unique(np.asarray(labels), return_counts=True)
    return {str(value.item()): int(count) for value, count in zip(values, counts)}


def _cut_record(cut, schema):
    scope = "event" if isinstance(cut, EventCut) else "slot"
    inner = cut.cut if isinstance(cut, EventCut) else cut
    spec = schema[inner.observable_index]
    record = {
        "scope": scope,
        "observable": spec.name,
        "observable_kind": spec.kind.value,
        "raw_text": cut.to_text(schema),
        "display_text": _predicate_text(cut, schema, include_scope=scope == "event"),
    }
    if spec.validity_column is not None:
        record["validity_guard"] = {
            "scope": scope, "observable": spec.validity_column, "required_value": True,
        }
    if isinstance(inner, ThresholdCut):
        record.update(
            {
                "cut_type": "threshold",
                "comparison": inner.comparison.value,
                "threshold": inner.threshold,
            }
        )
    else:
        record.update(
            {
                "cut_type": "category",
                "accepted_codes": sorted(inner.accepted_codes),
            }
        )
    return record


def _predicate_text(cut, schema, *, include_scope=False):
    inner = cut.cut if isinstance(cut, EventCut) else cut
    spec = schema[inner.observable_index]
    if isinstance(inner, ThresholdCut) and spec.kind is ObservableKind.ORDINAL:
        value = inner.threshold
        if inner.comparison is Comparison.GREATER:
            text = f"{spec.name} >= {int(np.floor(value)) + 1}"
        elif inner.comparison is Comparison.GREATER_EQUAL:
            text = f"{spec.name} >= {int(np.ceil(value))}"
        elif inner.comparison is Comparison.LESS:
            text = f"{spec.name} <= {int(np.ceil(value)) - 1}"
        else:
            text = f"{spec.name} <= {int(np.floor(value))}"
        if spec.validity_column is not None:
            text = f"{spec.validity_column} == True AND {text}"
    else:
        text = inner.to_text(schema)
    return f"event: {text}" if include_scope else text


def _selected_condition(cut, schema, branch, minimum_count):
    is_event = isinstance(cut, EventCut)
    if is_event:
        if branch is Branch.POSITIVE:
            return _predicate_text(cut, schema, include_scope=True)
        return _negated_event_text(cut, schema)
    aggregation = "EXISTS" if minimum_count == 1 else f"COUNT >= {minimum_count}"
    predicate = _predicate_text(cut, schema)
    condition = f"{aggregation} current slot: {predicate}"
    return condition if branch is Branch.POSITIVE else f"NOT ({condition})"


def _negated_event_text(cut, schema):
    inner = cut.cut
    if schema[inner.observable_index].validity_column is not None:
        return f"NOT ({_predicate_text(cut, schema, include_scope=True)})"
    if isinstance(inner, ThresholdCut):
        inverse = {
            Comparison.GREATER: Comparison.LESS_EQUAL,
            Comparison.GREATER_EQUAL: Comparison.LESS,
            Comparison.LESS: Comparison.GREATER_EQUAL,
            Comparison.LESS_EQUAL: Comparison.GREATER,
        }[inner.comparison]
        negated = EventCut(ThresholdCut(inner.observable_index, inverse, inner.threshold))
        return _predicate_text(negated, schema, include_scope=True)
    return f"NOT ({_predicate_text(cut, schema, include_scope=True)})"


def _simplified_rule_text(rule, model):
    event_cuts = [step.cut.cut for step in rule.steps if isinstance(step.cut, EventCut)]
    slot_cuts = [step.cut for step in rule.steps if not isinstance(step.cut, EventCut)]
    event_effective = _simplify_cut_group(event_cuts)
    slot_effective = _simplify_cut_group(slot_cuts)
    parts = [
        f"event: {_predicate_text(cut, model.event_observables)}"
        for cut in event_effective
    ]
    if slot_effective:
        prefix = (
            "EXISTS same slot"
            if rule.minimum_count == 1
            else f"COUNT >= {rule.minimum_count} matching slots"
        )
        predicates = " AND ".join(
            _predicate_text(cut, model.observables) for cut in slot_effective
        )
        parts.append(f"{prefix}: {predicates}")
    return " AND ".join(parts), len(event_effective) + len(slot_effective)


def _simplify_cut_group(cuts: Sequence[ThresholdCut | CategoryCut]):
    retained: list[ThresholdCut | CategoryCut | None] = []
    positions: dict[tuple[int, str], int] = {}
    for cut in cuts:
        if isinstance(cut, ThresholdCut):
            if cut.comparison in (Comparison.GREATER, Comparison.GREATER_EQUAL):
                family = "lower"
            else:
                family = "upper"
            key = (cut.observable_index, family)
            if key not in positions:
                positions[key] = len(retained)
                retained.append(cut)
                continue
            index = positions[key]
            current = retained[index]
            assert isinstance(current, ThresholdCut)
            if family == "lower" and cut.threshold > current.threshold:
                retained[index] = cut
            elif family == "upper" and cut.threshold < current.threshold:
                retained[index] = cut
        else:
            key = (cut.observable_index, "category")
            if key not in positions:
                positions[key] = len(retained)
                retained.append(cut)
                continue
            index = positions[key]
            current = retained[index]
            assert isinstance(current, CategoryCut)
            intersection = current.accepted_codes & cut.accepted_codes
            retained[index] = CategoryCut(cut.observable_index, intersection)
    return tuple(cut for cut in retained if cut is not None)


def _report_path(value, model_name, signal_path):
    if value is not None:
        return value.expanduser().resolve()
    dataset_name = signal_path.parent.name or "dataset"
    return (Path.cwd() / f"{model_name}_{dataset_name}_quantitative.txt").resolve()


def _check_report_target(parser, path, overwrite):
    if path.exists() and not overwrite:
        parser.error(f"Refusing to overwrite existing report {path}; use --overwrite-report.")


def _write_report(path, report, *, overwrite):
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with path.open(mode, encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=False, allow_nan=False)
        handle.write("\n")


def _print_header(report, report_path):
    print("=" * 88)
    print(f"MODEL: {report['model']}")
    print(f"REPORT: {report_path}")
    print("=" * 88)
    for role in ("signal", "background"):
        record = report["inputs"][role]
        print(
            f"{role.upper():10s} {record['event_count']:6d} events | "
            f"labels={record['stored_label_counts']} | {record['path']}"
        )
    event_names = [item["name"] for item in report["event_observables"]]
    print(f"EVENT OBSERVABLES: {', '.join(event_names) if event_names else '(none)'}")
    print("NOTE: every signal-input row is treated as mixture class, regardless of stored label.")


def _print_section(title):
    print(f"\n{'-' * 88}\n{title}\n{'-' * 88}")


def _print_selection(title, metrics):
    _print_section(title)
    _print_metrics(metrics)


def _print_metrics(metrics):
    ratio = metrics["acceptance_ratio"]
    ratio_text = "undefined (zero reference acceptance)" if ratio is None else f"{ratio:.6g}"
    print(
        "  accepted:   "
        f"mixture {metrics['signal_accepted']}/{metrics['signal_total']} "
        f"({metrics['signal_acceptance']:.6%}) | "
        f"reference {metrics['background_accepted']}/{metrics['background_total']} "
        f"({metrics['background_acceptance']:.6%})"
    )
    print(
        f"  direction:  {metrics['direction']} | "
        f"difference={metrics['acceptance_difference']:.6g} | ratio={ratio_text}"
    )
    print(f"  class purity: {metrics['mixture_label_purity']:.6%}")
    print(
        "  labels:     "
        f"mixture={metrics['selected_signal_stored_label_counts']} | "
        f"reference={metrics['selected_background_stored_label_counts']}"
    )


def _print_edt_leaf(record):
    print(
        f"\n[{record['rank']:03d}] leaf={record['leaf_id']} depth={record['depth']} "
        f"leaf_signal_fraction={record['leaf_signal_fraction']:.6f} "
        f"default_class={record['predicted_class_at_default_threshold']}"
    )
    _print_metrics(record["metrics"])
    print("  path:")
    if not record["path"]:
        print("    ROOT")
    for index, step in enumerate(record["path"], start=1):
        gain = step["source_split_gain"]
        gain_text = "n/a" if gain is None else f"{gain:.6g}"
        print(
            f"    {index}. node {step['source_node_id']} [{step['branch'].upper()}] "
            f"{step['selected_condition']} (split_gini_gain={gain_text})"
        )


def _print_rule_records(records, requested):
    if not records:
        print("(none)")
        return
    limit = len(records) if requested == 0 else min(requested, len(records))
    for section_rank, record in enumerate(records[:limit], start=1):
        pruned_rank = record["pruned_rank"]
        pruned_text = "n/a" if pruned_rank is None else str(pruned_rank)
        print(
            f"\n[{section_rank:03d}] global_rank={record['rank']} rule={record['rule_id']} "
            f"gini_gain={record['gini_gain']:.6g} depth={record['depth']} "
            f"effective_depth={record['effective_depth_after_simplification']} "
            f"pruned_rank={pruned_text}"
        )
        print(f"  rule:       {record['simplified_rule']}")
        if record["raw_rule"] != record["simplified_rule"]:
            print(f"  raw rule:   {record['raw_rule']}")
        _print_metrics(record["metrics"])
        print("  prefix history:")
        for step in record["steps"]:
            print(
                f"    {step['depth']}. [{step['scope']}] {step['display']} | "
                f"gain={step['prefix_gini_gain']:.6g} | "
                f"mixture={step['prefix_signal_accepted']} "
                f"reference={step['prefix_background_accepted']}"
            )
    if limit < len(records):
        print(f"\n... {len(records) - limit} additional rules are stored in the report.")


def _nonnegative_int(value):
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return result
