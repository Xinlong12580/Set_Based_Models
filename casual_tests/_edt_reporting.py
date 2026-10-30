"""EDT-specific fitting/report assembly; imported only by EDT entry points."""

import numpy as np

from set_based_models import EventCut, gini_gain
from casual_tests import _event_only_reporting as event_reporting
from casual_tests import _quantitative_reporting as reporting


def run_object_edt_cli(
    argv=None,
    *,
    require_event_features: bool,
    description: str,
) -> None:
    parser = reporting._parser(description, model="EDT", require_event_features=require_event_features)
    args = parser.parse_args(argv)
    from set_based_models import ExistentialDecisionTree

    signal, background, event_observables = reporting._load_pair(parser, args, require_event_features)
    report_path = reporting._report_path(args.report, "EDT", signal.path)
    reporting._check_report_target(parser, report_path, args.overwrite_report)

    tree = ExistentialDecisionTree(
        reporting.OBSERVABLES,
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
        metrics = reporting._selection_metrics(
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
                    "cut": reporting._cut_record(path_step.cut, schema),
                    "selected_condition": reporting._selected_condition(
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
    report = reporting._base_report("EDT", signal, background, event_observables)
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
                "default_prediction": reporting._selection_metrics(
                    signal_prediction,
                    background_prediction,
                    signal.labels,
                    background.labels,
                ),
            },
            "results": {"leaves": leaf_records},
        }
    )

    reporting._print_header(report, report_path)
    reporting._print_selection("DEFAULT TREE SELECTION", report["summary"]["default_prediction"])
    reporting._print_section("LEAVES RANKED BY MIXTURE FRACTION")
    limit = len(leaf_records) if args.print_top == 0 else min(args.print_top, len(leaf_records))
    for record in leaf_records[:limit]:
        reporting._print_edt_leaf(record)
    if limit < len(leaf_records):
        print(f"\n... {len(leaf_records) - limit} additional leaves are stored in the report.")
    reporting._write_report(report_path, report, overwrite=args.overwrite_report)
    print(f"\nStructured report saved to: {report_path}")


def run_event_edt_cli(argv=None, *, description):
    parser = event_reporting._parser(description, model="EDT")
    args = parser.parse_args(argv)
    from set_based_models import ExistentialDecisionTree

    signal, background, target = event_reporting._load_pair(parser, args, "EDT")
    tree = ExistentialDecisionTree(
        event_reporting.DUMMY_OBSERVABLES, event_observables=signal.event_observables,
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
    report = event_reporting._base_report("EDT", signal, background)
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
    event_reporting._print_header(report, target)
    reporting._print_selection("DEFAULT TREE SELECTION", report["summary"]["default_prediction"])
    reporting._print_section("LEAVES RANKED BY MIXTURE FRACTION")
    displayed = records if args.print_top == 0 else records[:args.print_top]
    for record in displayed:
        reporting._print_edt_leaf(record)
    if len(displayed) < len(records):
        print(f"\n... {len(records) - len(displayed)} additional leaves are stored in the report.")
    reporting._write_report(target, report, overwrite=args.overwrite_report)
    print(f"\nStructured report saved to: {target}")
