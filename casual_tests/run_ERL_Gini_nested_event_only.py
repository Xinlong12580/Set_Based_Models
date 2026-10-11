"""Event-only penalized-Gini ERL with unified logical-subset pruning.

After search, apply an optional per-rule rate budget, normalize cuts, and merge
equivalent rules. A stricter rule must improve a retained looser rule's score by
more than --stricter-score-improvement (default 0), whether it tightens thresholds
or adds AND conditions. All candidates remain in the JSON report with pruning
reasons. No final Jaccard pruning is applied.
"""

import argparse
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from set_based_models import ExistentialRuleList  # noqa: E402
from casual_tests import _event_only_reporting as event_reporting  # noqa: E402
from casual_tests import _quantitative_reporting as reporting  # noqa: E402
from set_based_models.rule_list.pruning import prune_rule_candidates  # noqa: E402
from casual_tests.run_ERL_Gini_penalized_event_only import (  # noqa: E402
    ReferencePenalizedGini, _nonnegative_float, _print_rules, _similarity_threshold,
)


def _finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise argparse.ArgumentTypeError("value must be finite")
    return result


def _positive_float(value):
    result = _finite_float(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _prune_with_rate_limit(rules, *, min_score_improvement=0.0, max_reference_acceptance=None):
    """Compatibility wrapper; the runner itself calls the model directly."""
    return prune_rule_candidates(
        rules, stricter_score_improvement=min_score_improvement,
        max_reference_acceptance=max_reference_acceptance, similarity_filter=False,
    )


def main(argv=None):
    parser = event_reporting._parser(__doc__, model="ERL")
    parser.add_argument("--reference-penalty", type=_nonnegative_float, default=1.0,
                        help="Lambda in Gini - lambda * reference acceptance (default: 1; 0 = pure Gini).")
    parser.add_argument("--stricter-score-improvement", type=_finite_float, default=0.0,
                        help="Require score(stricter)-score(looser) > this value; "
                             "negative values permit bounded score loss (default: 0).")
    parser.add_argument("--max-trigger-rate-khz", type=_nonnegative_float, default=None,
                        help="Optional per-rule reference rate budget in kHz; applied before logical-subset pruning.")
    parser.add_argument("--input-rate-mhz", type=_positive_float, default=40.0,
                        help="Input-stream rate in MHz for reference rate estimates (default: 40).")
    parser.add_argument("--beam-similarity-filter", action="store_true",
                        help="Opt into beam Jaccard filtering; nested alternatives may be lost before final pruning.")
    parser.add_argument("--signal-similarity-threshold", type=_similarity_threshold,
                        default=reporting.SIGNAL_SIMILARITY_THRESHOLD,
                        help="Mixture Jaccard threshold, used only with --beam-similarity-filter.")
    parser.add_argument("--background-similarity-threshold", type=_similarity_threshold,
                        default=reporting.BACKGROUND_SIMILARITY_THRESHOLD,
                        help="Reference Jaccard threshold, used only with --beam-similarity-filter.")
    parser.add_argument("--print-reference-top", type=reporting._nonnegative_int, default=5,
                        help="Reference-enriched rules to print; 0 prints none.")
    parser.add_argument("--print-pruned-top", type=reporting._nonnegative_int, default=20,
                        help="Retained mixture-enriched rules to print; 0 prints all.")
    args = parser.parse_args(argv)
    max_acceptance = (None if args.max_trigger_rate_khz is None else
                      (args.max_trigger_rate_khz / 1000.0) / args.input_rate_mhz)
    input_rate_khz = args.input_rate_mhz * 1000.0
    if not math.isfinite(input_rate_khz) or (max_acceptance is not None and not math.isfinite(max_acceptance)):
        parser.error("rate values are too large or small for finite rate calculations")
    signal, background, target = event_reporting._load_pair(
        parser, args, "ERL_Gini_nested", native_event_only=True)
    model = ExistentialRuleList(
        event_observables=signal.event_observables,
        max_rule_depth=3, beam_width=100, max_candidate_rules=1000, max_thresholds=32,
        minimum_count=1, min_signal_efficiency=0.01,
        allow_validity_cuts=args.allow_validity_cuts,
        scoring_function=ReferencePenalizedGini(args.reference_penalty),
        beam_similarity_filter=args.beam_similarity_filter,
        beam_signal_similarity_threshold=args.signal_similarity_threshold,
        beam_background_similarity_threshold=args.background_similarity_threshold,
    ).fit(
        signal_event_features=signal.event_features, background_event_features=background.event_features,
    )
    model.prune_candidates(
        stricter_score_improvement=args.stricter_score_improvement,
        max_reference_acceptance=None if max_acceptance is None else min(1.0, max_acceptance),
        similarity_filter=False,
    )
    result = model.pruning_result_
    pruned_ranks = {rule.rule_id: rank for rank, rule in enumerate(result.retained, start=1)}
    records = []
    for rank, rule in enumerate(model.candidate_rules_, start=1):
        normalized = result.normalized_cuts[rule.rule_id]
        simplified = " AND ".join(reporting._predicate_text(cut, model.event_observables, include_scope=True)
                                   for cut in normalized)
        steps = []
        for step in rule.steps:
            stats = step.statistics
            steps.append({
                "depth": step.depth, "scope": "event",
                "cut": reporting._cut_record(step.cut, model.event_observables),
                "display": reporting._predicate_text(step.cut, model.event_observables, include_scope=True),
                "prefix_score": step.generation_score, "prefix_gini_gain": reporting.gini_rule_score(stats),
                "prefix_reference_penalty_term": args.reference_penalty * stats.false_positive_rate,
                "prefix_signal_accepted": stats.signal_accepted,
                "prefix_background_accepted": stats.background_accepted,
                "prefix_signal_acceptance": stats.true_positive_rate,
                "prefix_reference_acceptance": stats.false_positive_rate,
            })
        metrics = reporting._selection_metrics(rule.signal_selection, rule.background_selection,
                                               signal.labels, background.labels)
        retained = rule.rule_id in pruned_ranks
        records.append({
            "rank": rank, "rule_id": rule.rule_id, "depth": rule.depth,
            "effective_depth_after_simplification": len(normalized),
            "score": rule.generation_score, "gini_gain": steps[-1]["prefix_gini_gain"],
            "reference_penalty_term": steps[-1]["prefix_reference_penalty_term"],
            "direction": metrics["direction"], "raw_rule": rule.to_text(), "simplified_rule": simplified,
            "metrics": metrics, "steps": steps,
            "normalized_cuts": [reporting._cut_record(cut, model.event_observables) for cut in normalized],
            "estimated_reference_rate_khz": input_rate_khz * rule.steps[-1].statistics.false_positive_rate,
            "passes_reference_rate_limit": (max_acceptance is None or
                                            rule.steps[-1].statistics.false_positive_rate <= max_acceptance),
            "retained_after_final_pruning": retained,
            "retained_after_nested_pruning": retained,
            # Legacy fields alias final retention, not separate nesting/Jaccard stages.
            "retained_after_similarity_pruning": retained,
            "pruned_rank": pruned_ranks.get(rule.rule_id), "pruning": result.decisions[rule.rule_id],
        })
    enriched = [record for record in records if record["direction"] == "mixture-enriched"]
    reference = [record for record in records if record["direction"] == "reference-enriched"]
    report = event_reporting._base_report("ERL_Gini_nested", signal, background,
                                          allow_validity_cuts=args.allow_validity_cuts)
    report.update({
        "ranking_criterion": "score descending (not raw Gini gain)",
        "retention_field_note": "retention flags include rate, equivalence merging and unified stricter-rule filtering; legacy nested/similarity flags alias final retention for plotting",
        "configuration": {
            "allow_validity_cuts": args.allow_validity_cuts,
            "scoring_function": "gini_gain_minus_reference_acceptance_penalty",
            "score_formula": "gini_gain - reference_penalty * reference_acceptance",
            "reference_penalty": args.reference_penalty, "penalty_applied_during_search": True,
            "pruning_stage": "after_candidate_generation",
            "pruning_implementation": "ExistentialRuleList.prune_candidates",
            "stricter_score_improvement": args.stricter_score_improvement,
            "stricter_filter_stage": "after_rate_filter_and_equivalence_merging",
            "stricter_filter_scope": "all_recognized_strict_subsets_including_tighter_thresholds_and_extra_AND_conditions",
            "stricter_filter_order": "logically_looser_first_then_final_score_ranking",
            "stricter_filter_comparison": "improvement_must_be_strictly_greater_with_1e-12_roundoff_tolerance",
            "max_trigger_rate_khz": args.max_trigger_rate_khz, "input_rate_mhz": args.input_rate_mhz,
            "max_reference_acceptance": max_acceptance,
            "rate_filter_stage": "before_equivalence_merging_and_stricter_pruning",
            "rate_limit_scope": "each_rule_not_combined_OR",
            "rate_estimation": "training_reference_acceptance_times_input_rate_no_confidence_bound",
            "pruning_policy": "rate_filter_then_merge_equivalents_then_unified_stricter_score_improvement",
            "complexity_measure": "number_of_normalized_predicates",
            "containment_method": "same_column_bounds_and_category_sets_not_training_masks",
            "final_similarity_pruning": False,
            "max_rule_depth": model.max_rule_depth, "beam_width": model.beam_width,
            "max_candidate_rules": model.max_candidate_rules,
            "max_thresholds": model.max_thresholds, "minimum_count": model.minimum_count,
            "min_signal_efficiency": model.min_signal_efficiency,
            "beam_similarity_filter": args.beam_similarity_filter,
            "beam_signal_similarity_threshold": args.signal_similarity_threshold,
            "beam_background_similarity_threshold": args.background_similarity_threshold,
            "object_features_enabled": False,
        },
        "summary": {
            "candidate_rule_count": len(records), "pruned_rule_count": len(result.retained),
            "mixture_enriched_rule_count": len(enriched), "reference_enriched_rule_count": len(reference),
            "pruned_rule_ids": [rule.rule_id for rule in result.retained],
            "equivalent_rules_removed": sum(record["pruning"]["status"] == "equivalent_rule" for record in records),
            "rate_rules_removed": sum(record["pruning"]["status"] == "reference_rate_limit" for record in records),
            "stricter_rules_removed": sum(record["pruning"]["status"] == "stricter_score_improvement" for record in records),
        },
        "results": {"rules": records},
    })
    event_reporting._print_header(report, target)
    print(f"SCORING: Gini gain - {args.reference_penalty:g} * reference acceptance (fraction, not percent)")
    if max_acceptance is not None:
        print(f"RATE LIMIT: each rule <= {args.max_trigger_rate_khz:g} kHz at {args.input_rate_mhz:g} MHz input; "
              f"reference acceptance <= {max_acceptance:.6g} ({100 * max_acceptance:.6g}%)")
        print("NOTE: rate filtering precedes equivalence merging and logical-subset pruning; search and scores are unchanged.")
        print("NOTE: rates are training estimates assuming a representative reference stream; "
              "no uncertainty bound or combined-OR budget is enforced.")
    print("EQUIVALENCE: merge provably equivalent rules, preferring the shortest representation.")
    print(f"STRICTER FILTER: require score(stricter)-score(looser) > {args.stricter_score_improvement:g}; "
          "covers tighter thresholds and additional AND conditions, compared against retained looser rules.")
    print("NOTE: this score policy can remove lower-rate operating points; it is not an acceptance-dominance filter.")
    print(f"SIMILARITY: beam_filter={args.beam_similarity_filter}; final Jaccard pruning disabled.")
    print("NOTE: logical-subset pruning is post-search; stricter rules never suppress their looser counterparts.")
    print("NOTE: zero observed reference acceptance is not proof of zero background rate; validate independently.")
    reporting._print_section("TOP MIXTURE-ENRICHED CANDIDATES BEFORE PRUNING")
    _print_rules(enriched, args.print_top)
    if args.print_reference_top:
        reporting._print_section("TOP REFERENCE-ENRICHED CANDIDATES (NOT ANOMALY SELECTIONS)")
        _print_rules(reference, args.print_reference_top)
    reporting._print_section("TOP MIXTURE-ENRICHED RULES AFTER "
                             + ("RATE FILTERING AND " if max_acceptance is not None else "") + "UNIFIED STRICTER-RULE PRUNING")
    _print_rules([record for record in enriched if record["retained_after_final_pruning"]], args.print_pruned_top)
    reporting._print_section("PRUNING AUDIT")
    by_id = {record["rule_id"]: record for record in records}
    removed = [record for record in records if not record["retained_after_final_pruning"]]
    for record in removed:
        decision = record["pruning"]
        if decision["status"] == "reference_rate_limit":
            print(f"  rule={record['rule_id']} reason=reference_rate_limit "
                  f"estimated_rate_khz={record['estimated_reference_rate_khz']:.6g} "
                  f"budget_khz={args.max_trigger_rate_khz:g}")
            print(f"    discarded: {record['simplified_rule']}")
            continue
        witness = by_id[decision["pruned_by_rule_id"]]
        print(f"  rule={record['rule_id']} reason={decision['status']} "
              f"witness_rule={witness['rule_id']} retained_witness_rule={decision['retained_witness_rule_id']} "
              f"score_gap={decision['score_gap']:.6g}")
        if decision["status"] == "stricter_score_improvement":
            print(f"    score_improvement={decision['score_improvement']:.6g}; "
                  f"required_above={args.stricter_score_improvement:g}")
        print(f"    discarded: {record['simplified_rule']}")
        print(f"    witness:   {witness['simplified_rule']}")
    if not removed:
        print("(none)")
    reporting._write_report(target, report, overwrite=args.overwrite_report)
    print(f"\nStructured report saved to: {target}")


if __name__ == "__main__":
    main()
