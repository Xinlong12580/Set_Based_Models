"""Print saved ERL statistics and top rules without loading input tensors.

    python scripts/print_erl_model.py MODEL.json --top 10
    python scripts/print_erl_model.py MODEL.json --rule-ids 548 --verbosity 2
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from set_based_models import EventCut, ExistentialRuleList
from set_based_models.rule_list.pruning import normalize_rule


def _load(model):
    source = None
    if isinstance(model, (str, Path)):
        source = str(Path(model).expanduser().resolve())
        model = json.loads(Path(source).read_text())
    if isinstance(model, dict):
        cache = model.get("event_jaccard")
        jaccard = None if cache is None else {"size": len(cache["rule_ids"]),
                                             "source": cache["source"], "blocks": cache["blocks"]}
        # Printing needs cache metadata, not a second copy of the dense matrices.
        model = ExistentialRuleList.from_dict({key: value for key, value in model.items() if key != "event_jaccard"})
    else:
        jaccard = None if model.signal_jaccard_ is None else {
            "size": len(model.candidate_rules_), "source": model.jaccard_source_, "blocks": model.jaccard_blocks_}
    return model, source, jaccard


def _rule_text(rule, cuts):
    parts = [cut.to_text(rule.event_observables) for cut in cuts if isinstance(cut, EventCut)]
    slots = [cut.to_text(rule.observables) for cut in cuts if not isinstance(cut, EventCut)]
    if slots:
        prefix = "EXISTS same slot" if rule.minimum_count == 1 else f"COUNT >= {rule.minimum_count} matching slots"
        parts.append(f"({prefix}: {' AND '.join(slots)})")
    return " AND ".join(parts)


def _acceptance(statistics):
    return (f"mixture={statistics.signal_accepted}/{statistics.signal_total} "
            f"({100 * statistics.true_positive_rate:.6g}%), "
            f"reference={statistics.background_accepted}/{statistics.background_total} "
            f"({100 * statistics.false_positive_rate:.6g}%)")


def print_model(model, *, top=10, verbosity=1, all_candidates=False, rule_ids=None, stream=None):
    """Print a model instance, core JSON dictionary, or model path; change nothing.

    Verbosity: 0 summary; 1 top rules; 2 prefix statistics and provenance;
    3 full configuration and feature schemas. top=0 shows the whole chosen pool.
    Default to retained rules after pruning, otherwise all candidates.
    """
    stream = sys.stdout if stream is None else stream
    def emit(text=""):
        print(text, file=stream)
    model, source, jaccard = _load(model)
    candidates = sorted(model.candidate_rules_, key=lambda rule: (-rule.generation_score, rule.rule_id))
    global_ranks = {rule.rule_id: rank for rank, rule in enumerate(candidates, 1)}
    retained = {rule.rule_id for rule in model.selected_rules_}
    pruned = model.pruning_result_ is not None
    if rule_ids is not None:
        pool, pool_name = [model.get_rule(index) for index in rule_ids], "requested rules"
    elif all_candidates or not pruned:
        pool, pool_name = candidates, "all candidates"
    else:
        pool, pool_name = model.selected_rules_, "retained rules"
    pool = sorted(pool, key=lambda rule: (-rule.generation_score, rule.rule_id))

    emit(f"ERL model{': ' + source if source else ''}")
    emit(f"Mode: {model.input_mode_ or 'unfitted'} | Statistics: {model.statistics_source_ or 'unavailable'}")
    selection = f"{len(retained)} retained" if pruned else "not applied"
    emit(f"Candidates: {len(candidates)} | Final selection: {selection}")
    emit(f"Features: {len(model.observables)} object, {len(model.event_observables)} event")
    totals = {(rule.statistics.signal_total, rule.statistics.background_total) for rule in candidates}
    if len(totals) == 1:
        mixture, reference = next(iter(totals))
        emit(f"Stored sample sizes: mixture={mixture}, reference={reference}")
    elif totals:
        emit("Stored sample sizes vary across candidates/sources.")
    if model.statistics_source_ == "merged":
        emit("Scores come from separate sources; compare them after common-data recalculation.")
    if verbosity >= 2:
        emit("Scores are saved values; the scoring callable is not stored in the model.")
        if pruned:
            emit(f"Pruning decisions: {dict(Counter(d['status'] for d in model.pruning_result_.decisions.values()))}")
        if jaccard is not None:
            emit(f"Event Jaccard: mixture and reference {jaccard['size']} x {jaccard['size']}; "
                 f"source={jaccard['source']}; blocks={list(jaccard['blocks'])}")
        provenance = model.self_pruning_provenance_
        if provenance is not None:
            emit(f"Self-pruning: {len(provenance['source_rule_ids'])} source rules; "
                 f"{provenance['evaluated_variant_count']} evaluated variants; "
                 f"{provenance['locally_retained_variant_count']} local survivors")
    if verbosity >= 3:
        configuration = {name: getattr(model, name) for name in (
            "max_rule_depth", "beam_width", "max_candidate_rules", "max_thresholds", "minimum_count",
            "min_signal_efficiency", "beam_similarity_filter", "beam_signal_similarity_threshold",
            "beam_background_similarity_threshold", "beam_novelty_weight", "allow_validity_cuts", "object_cut_priority")}
        emit("\nContainer configuration:")
        emit(json.dumps(configuration, indent=2))
        emit(f"Final pruning configuration: {json.dumps(model.pruning_configuration_)}")
        if model.self_pruning_provenance_ is not None:
            emit(f"Self-pruning configuration: {json.dumps(model.self_pruning_provenance_['configuration'])}")
        if model.merge_provenance_ is not None:
            emit("Source models and their search configurations:")
            for index, item in enumerate(model.merge_provenance_["sources"]):
                emit(f"  source={index} name={item['name']} configuration={json.dumps(item['configuration'])}")
        for scope, schema in (("Object", model.observables), ("Event", model.event_observables)):
            emit(f"{scope} schema:")
            for index, spec in enumerate(schema):
                emit(f"  [{index}] {spec.name}: {spec.kind.value}; guard={spec.validity_column}; categories={spec.categories}")
    if verbosity == 0:
        return

    shown = pool if top == 0 else pool[:top]
    emit(f"\nTop {len(shown)} of {len(pool)} {pool_name}, ordered by stored score (ties: rule ID).")
    emit("Rank is within this pool; global_rank is across all candidates. Counts are unweighted; acceptance uses stored weights.")
    for rank, rule in enumerate(shown, 1):
        statistics = rule.statistics
        decision = None if not pruned else model.pruning_result_.decisions[rule.rule_id]
        status = "unpruned" if not pruned else "retained" if rule.rule_id in retained else decision["status"]
        cuts = normalize_rule(rule)
        emit(f"\n[{rank}] rule={rule.rule_id} global_rank={global_ranks[rule.rule_id]} score={rule.generation_score:.9g} "
             f"status={status} depth={rule.depth} simplified_cuts={len(cuts)}")
        emit(f"  {_rule_text(rule, cuts)}")
        emit(f"  {_acceptance(statistics)}")
        if verbosity >= 2:
            difference = statistics.true_positive_rate - statistics.false_positive_rate
            direction = "mixture-enriched" if difference > 0 else "reference-enriched" if difference < 0 else "equal acceptance"
            emit(f"  Direction: {direction}; acceptance difference={difference:.9g}")
            emit(f"  Accepted/total weights: mixture={statistics.signal_weight_accepted:g}/{statistics.signal_weight_total:g}, "
                 f"reference={statistics.background_weight_accepted:g}/{statistics.background_weight_total:g}")
            if decision is not None:
                emit(f"  Pruning: {json.dumps(decision)}")
            if model.merge_provenance_ is not None:
                origin = model.merge_provenance_["rules"][rule.rule_id]
                item = model.merge_provenance_["sources"][origin["source_index"]]
                emit(f"  Origin: source={origin['source_index']} model={item['name']} original_rule={origin['rule_id']}")
            if model.self_pruning_provenance_ is not None:
                parents = model.self_pruning_provenance_["rules"][rule.rule_id]["parent_rule_ids"]
                displayed = parents if verbosity >= 3 else parents[:8]
                emit(f"  Self-pruning parent IDs: {displayed}" + (f" (+{len(parents) - len(displayed)} more)" if len(displayed) < len(parents) else ""))
            emit("  Prefix history (stored cut order):")
            for step in rule.steps:
                schema = rule.event_observables if isinstance(step.cut, EventCut) else rule.observables
                emit(f"    {step.depth}. {step.cut.to_text(schema)} | score={step.generation_score:.9g} | {_acceptance(step.statistics)}")
    if not pool:
        emit("No rules in this pool. Use --all-candidates to inspect candidates after an empty final selection.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path, help="Core ERL model JSON, not a runner report")
    parser.add_argument("--top", type=int, default=10, help="Number of top rules to show; 0 shows all")
    parser.add_argument("--verbosity", "-v", type=int, choices=(0, 1, 2, 3), default=1,
                        help="0 summary; 1 top rules; 2 prefix statistics/provenance; 3 configurations/schemas")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--all-candidates", action="store_true", help="Include candidates removed by final pruning")
    selection.add_argument("--rule-ids", type=int, nargs="+", help="Inspect specific IDs, ranked by stored score")
    args = parser.parse_args(argv)
    print_model(args.model, top=args.top, verbosity=args.verbosity,
                all_candidates=args.all_candidates, rule_ids=args.rule_ids)


if __name__ == "__main__":
    main()
