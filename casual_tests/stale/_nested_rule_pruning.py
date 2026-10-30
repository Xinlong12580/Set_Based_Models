"""Compatibility helpers; production pruning lives in the model package."""

import math

from set_based_models import EventCut
from set_based_models.rule_list.pruning import (
    RulePruningResult as NestedPruningResult,
    merge_equivalent_rules, normalize_event_rule, prune_stricter_rules, rule_implies,
)


def prune_nested_rules(rules, *, score_gap=0.001):
    """Legacy event-only gap policy; not used by the unified ERL runner."""
    if not math.isfinite(score_gap):
        raise ValueError("score_gap must be finite.")
    rules = list(rules)
    if any(not isinstance(cut, EventCut) for rule in rules for cut in rule.cuts):
        raise ValueError("Legacy nested pruning requires event-only rules; use model.prune_candidates instead.")
    result = merge_equivalent_rules(rules)
    normalized, decisions = result.normalized_cuts, result.decisions
    representatives = list(result.retained)
    # Process simpler rules first, so a pruning witness is itself retained.
    representatives.sort(key=lambda rule: (len(normalized[rule.rule_id]), rule.depth,
                                           -rule.generation_score, rule.rule_id))
    retained = []
    for complex_rule in representatives:
        complex_cuts = normalized[complex_rule.rule_id]
        witnesses = []
        for simple in retained:
            simple_cuts = normalized[simple.rule_id]
            gap = simple.generation_score - complex_rule.generation_score
            if (len(simple_cuts) < len(complex_cuts) and rule_implies(complex_cuts, simple_cuts)
                    and gap > score_gap and not math.isclose(gap, score_gap, rel_tol=1e-12, abs_tol=1e-12)):
                witnesses.append(simple)
        if witnesses:
            witness = max(witnesses, key=lambda rule: (rule.generation_score, -rule.depth, -rule.rule_id))
            decisions[complex_rule.rule_id] = {
                "status": "nested_score_gap", "pruned_by_rule_id": witness.rule_id,
                "score_gap": witness.generation_score - complex_rule.generation_score,
            }
        else:
            retained.append(complex_rule)
    # Equivalence witnesses can themselves be gap-pruned; retain both stages in
    # the audit and expose the final retained witness as well.
    for decision in decisions.values():
        witness = decision["pruned_by_rule_id"]
        while witness is not None and decisions[witness]["pruned_by_rule_id"] is not None:
            witness = decisions[witness]["pruned_by_rule_id"]
        decision["retained_witness_rule_id"] = witness
    retained.sort(key=lambda rule: (-rule.generation_score, rule.rule_id))
    return NestedPruningResult(tuple(retained), decisions, normalized)
