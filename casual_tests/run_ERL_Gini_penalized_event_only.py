"""Run event-only ERL with score = Gini gain - lambda * reference acceptance.

The default lambda is 1.0; --reference-penalty 0 recovers ordinary Gini scoring.
Every signal-file row is the anomalous-mixture class, regardless of its stored
process label. Real objects and separate native_event_info are not used.
Existing runners and model modules are unchanged.
"""

from dataclasses import dataclass
import argparse
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from set_based_models import EventCut, ExistentialRuleList, RuleStatistics  # noqa: E402
from casual_tests import _event_only_reporting as event_reporting  # noqa: E402
from casual_tests import _quantitative_reporting as reporting  # noqa: E402


@dataclass(frozen=True)
class ReferencePenalizedGini:
    """A larger-is-better ERL score, using the full reference acceptance rate.

    The penalty is applied during beam search, not just to final rankings.
    Negative scores remain eligible: do not reject prefixes that can become
    clean after another cut. Process labels never enter the objective.
    """

    reference_penalty: float = 1.0

    def __post_init__(self):
        if not math.isfinite(self.reference_penalty) or self.reference_penalty < 0:
            raise ValueError("reference_penalty must be finite and nonnegative.")

    def __call__(self, statistics: RuleStatistics) -> float:
        return (reporting.gini_rule_score(statistics)
                - self.reference_penalty * statistics.false_positive_rate)


def _nonnegative_float(value):
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError("value must be finite and nonnegative")
    return result


def _similarity_threshold(value):
    result = _nonnegative_float(value)
    if result > 1:
        raise argparse.ArgumentTypeError("similarity threshold must be between 0 and 1")
    return result


def _print_rules(records, requested):
    """Keep the established log layout, distinguishing gain from actual score."""
    if not records:
        print("(none)")
        return
    displayed = records if requested == 0 else records[:requested]
    for section_rank, record in enumerate(displayed, start=1):
        pruned = record["pruned_rank"]
        print(f"\n[{section_rank:03d}] global_rank={record['rank']} rule={record['rule_id']} "
              f"score={record['score']:.6g} gini_gain={record['gini_gain']:.6g} "
              f"reference_penalty_term={record['reference_penalty_term']:.6g} "
              f"depth={record['depth']} "
              f"effective_depth={record['effective_depth_after_simplification']} "
              f"pruned_rank={pruned if pruned is not None else 'n/a'}")
        print(f"  rule:       {record['simplified_rule']}")
        if record["raw_rule"] != record["simplified_rule"]:
            print(f"  raw rule:   {record['raw_rule']}")
        reporting._print_metrics(record["metrics"])
        print("  prefix history:")
        for step in record["steps"]:
            print(f"    {step['depth']}. [event] {step['display']} | "
                  f"score={step['prefix_score']:.6g} "
                  f"gain={step['prefix_gini_gain']:.6g} "
                  f"reference_penalty_term={step['prefix_reference_penalty_term']:.6g} | "
                  f"mixture={step['prefix_signal_accepted']} "
                  f"reference={step['prefix_background_accepted']}")
    if len(displayed) < len(records):
        print(f"\n... {len(records) - len(displayed)} additional rules are stored in the report.")


def main(argv=None):
    parser = event_reporting._parser(__doc__, model="ERL")
    parser.add_argument("--reference-penalty", type=_nonnegative_float, default=1.0,
                        help="Lambda in Gini - lambda * reference acceptance (default: 1.0; 0 = pure Gini).")
    parser.add_argument("--signal-similarity-threshold", type=_similarity_threshold,
                        default=reporting.SIGNAL_SIMILARITY_THRESHOLD,
                        help="Mixture Jaccard threshold for beam and final pruning; higher prunes less.")
    parser.add_argument("--background-similarity-threshold", type=_similarity_threshold,
                        default=reporting.BACKGROUND_SIMILARITY_THRESHOLD,
                        help="Reference Jaccard threshold for beam and final pruning; higher prunes less.")
    parser.add_argument("--no-beam-similarity-filter", action="store_true",
                        help="Disable similarity filtering during beam search; final pruning still runs.")
    parser.add_argument("--print-reference-top", type=reporting._nonnegative_int, default=5,
                        help="Reference-enriched rules to print; 0 prints none.")
    parser.add_argument("--print-pruned-top", type=reporting._nonnegative_int, default=10,
                        help="Pruned mixture-enriched rules to print; 0 prints all.")
    args = parser.parse_args(argv)
    signal, background, target = event_reporting._load_pair(
        parser, args, "ERL_Gini_penalized", native_event_only=True)
    scorer = ReferencePenalizedGini(args.reference_penalty)
    model = ExistentialRuleList(
        event_observables=signal.event_observables,
        max_rule_depth=3, beam_width=40, max_candidate_rules=200, max_thresholds=32,
        minimum_count=1, min_signal_efficiency=0.01, scoring_function=scorer,
        allow_validity_cuts=args.allow_validity_cuts,
        beam_similarity_filter=not args.no_beam_similarity_filter,
        beam_signal_similarity_threshold=args.signal_similarity_threshold,
        beam_background_similarity_threshold=args.background_similarity_threshold,
    ).fit(
        signal_event_features=signal.event_features, background_event_features=background.event_features,
    )
    if any(not isinstance(cut, EventCut) for rule in model.candidate_rules_ for cut in rule.cuts):
        raise RuntimeError("Unexpected object-level cut in an event-only rule list.")
    pruned = model.prune_candidates(
        signal_similarity_threshold=args.signal_similarity_threshold,
        background_similarity_threshold=args.background_similarity_threshold,
    )
    pruned_ranks = {rule.rule_id: rank for rank, rule in enumerate(pruned, start=1)}
    records = []
    for rank, rule in enumerate(model.candidate_rules_, start=1):
        simplified, effective_depth = reporting._simplified_rule_text(rule, model)
        steps = []
        for step in rule.steps:
            stats = step.statistics
            steps.append({
                "depth": step.depth, "scope": "event",
                "cut": reporting._cut_record(step.cut, model.event_observables),
                "display": reporting._predicate_text(step.cut, model.event_observables, include_scope=True),
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
        records.append({
            "rank": rank, "rule_id": rule.rule_id, "depth": rule.depth,
            "effective_depth_after_simplification": effective_depth,
            "score": rule.generation_score, "gini_gain": steps[-1]["prefix_gini_gain"],
            "reference_penalty_term": steps[-1]["prefix_reference_penalty_term"],
            "direction": metrics["direction"], "raw_rule": rule.to_text(),
            "simplified_rule": simplified, "metrics": metrics, "steps": steps,
            "retained_after_similarity_pruning": rule.rule_id in pruned_ranks,
            "pruned_rank": pruned_ranks.get(rule.rule_id),
        })
    enriched = [record for record in records if record["direction"] == "mixture-enriched"]
    reference = [record for record in records if record["direction"] == "reference-enriched"]
    report = event_reporting._base_report("ERL_Gini_penalized", signal, background,
                                          allow_validity_cuts=args.allow_validity_cuts)
    report.update({
        "ranking_criterion": "score descending (not raw Gini gain)",
        "configuration": {
            "allow_validity_cuts": args.allow_validity_cuts,
            "scoring_function": "gini_gain_minus_reference_acceptance_penalty",
            "score_formula": "gini_gain - reference_penalty * reference_acceptance",
            "reference_penalty": args.reference_penalty, "penalty_applied_during_search": True,
            "max_rule_depth": 3, "beam_width": 40, "max_candidate_rules": 200,
            "max_thresholds": 32, "minimum_count": 1, "min_signal_efficiency": 0.01,
            "beam_similarity_filter": not args.no_beam_similarity_filter,
            "beam_signal_similarity_threshold": args.signal_similarity_threshold,
            "beam_background_similarity_threshold": args.background_similarity_threshold,
            "prune_signal_similarity_threshold": args.signal_similarity_threshold,
            "prune_background_similarity_threshold": args.background_similarity_threshold,
            "object_features_enabled": False,
        },
        "summary": {
            "candidate_rule_count": len(records), "pruned_rule_count": len(pruned),
            "mixture_enriched_rule_count": len(enriched), "reference_enriched_rule_count": len(reference),
            "pruned_rule_ids": [rule.rule_id for rule in pruned],
        },
        "results": {"rules": records},
    })
    event_reporting._print_header(report, target)
    print(f"SCORING: Gini gain - {args.reference_penalty:g} * reference acceptance (fraction, not percent)")
    print("RANKING: descending penalized score, including during beam search; Gini gain is reported separately.")
    print(f"SIMILARITY: beam_filter={not args.no_beam_similarity_filter} | "
          f"mixture={args.signal_similarity_threshold:g} reference={args.background_similarity_threshold:g}")
    print("NOTE: zero observed reference acceptance is not proof of zero background rate; validate independently.")
    reporting._print_section("TOP MIXTURE-ENRICHED RULES")
    _print_rules(enriched, args.print_top)
    if args.print_reference_top:
        reporting._print_section("TOP REFERENCE-ENRICHED RULES (NOT ANOMALY SELECTIONS)")
        _print_rules(reference, args.print_reference_top)
    reporting._print_section("TOP PRUNED MIXTURE-ENRICHED RULES")
    _print_rules([record for record in enriched if record["retained_after_similarity_pruning"]],
                 args.print_pruned_top)
    reporting._write_report(target, report, overwrite=args.overwrite_report)
    print(f"\nStructured report saved to: {target}")


if __name__ == "__main__":
    main()
