"""Final candidate selection and conservative scope-aware logical pruning.

Containment uses predicates, never coincident training selections. Numeric
bounds on the same column and categorical membership are supported. Relations
between different columns (e.g. two photon threshold counts) are not inferred.
"""

from dataclasses import dataclass
import math

import numpy as np

from ..core.data import ObservableKind
from ..core.predicates import CategoryCut, Comparison, EventCut, ThresholdCut


@dataclass
class RulePruningResult:
    """Selected rules, per-candidate decisions, and normalized predicates."""

    retained: tuple
    decisions: dict
    normalized_cuts: dict


def _family(cut):
    if isinstance(cut, CategoryCut):
        return "category"
    return "lower" if cut.comparison in (Comparison.GREATER, Comparison.GREATER_EQUAL) else "upper"


def _closed(cut):
    return cut.comparison in (Comparison.GREATER_EQUAL, Comparison.LESS_EQUAL)


def _normalize_cuts(cuts, observables):
    """Normalize one conjunction within a single event or slot scope."""
    bounds = {}
    for cut in cuts:
        cut.validate(observables)
        index = cut.observable_index
        # Above float64's exact-integer range, floor(t)+1 can round back to t.
        # Keep the original endpoint there rather than infer false equivalence.
        if (isinstance(cut, ThresholdCut) and observables[index].kind is ObservableKind.ORDINAL
                and abs(cut.threshold) < 2**53):
            threshold = cut.threshold
            integer_bound = {
                Comparison.GREATER: math.floor(threshold) + 1,
                Comparison.GREATER_EQUAL: math.ceil(threshold),
                Comparison.LESS: math.ceil(threshold) - 1,
                Comparison.LESS_EQUAL: math.floor(threshold),
            }[cut.comparison]
            comparison = Comparison.GREATER_EQUAL if _family(cut) == "lower" else Comparison.LESS_EQUAL
            cut = ThresholdCut(index, comparison, integer_bound)
        key = (index, _family(cut))
        previous = bounds.get(key)
        if previous is None:
            bounds[key] = cut
        elif isinstance(cut, CategoryCut):
            codes = previous.accepted_codes & cut.accepted_codes
            if not codes:
                raise ValueError("Cannot normalize an unsatisfiable rule.")
            bounds[key] = CategoryCut(index, codes)
        elif (_family(cut) == "lower" and
              (cut.threshold, not _closed(cut)) > (previous.threshold, not _closed(previous))):
            bounds[key] = cut
        elif (_family(cut) == "upper" and
              (cut.threshold, _closed(cut)) < (previous.threshold, _closed(previous))):
            bounds[key] = cut
    for index in {index for index, _ in bounds}:
        lower, upper = bounds.get((index, "lower")), bounds.get((index, "upper"))
        if lower is not None and upper is not None:
            if (lower.threshold > upper.threshold or
                    (lower.threshold == upper.threshold and not (_closed(lower) and _closed(upper)))):
                raise ValueError("Cannot normalize an unsatisfiable rule.")
    return tuple(bounds[key] for key in sorted(bounds))


def normalize_rule(rule):
    """Remove redundant bounds separately for events and same-object cuts."""
    events = _normalize_cuts(
        (cut.cut for cut in rule.cuts if isinstance(cut, EventCut)), rule.event_observables)
    slots = _normalize_cuts(
        (cut for cut in rule.cuts if not isinstance(cut, EventCut)), rule.observables)

    def without_repeated_guards(cuts, schema):
        names = {schema[cut.observable_index].validity_column for cut in cuts}
        return tuple(cut for cut in cuts if not (
            isinstance(cut, CategoryCut) and cut.accepted_codes == {1}
            and schema[cut.observable_index].name in names))

    return (tuple(EventCut(cut) for cut in without_repeated_guards(events, rule.event_observables))
            + without_repeated_guards(slots, rule.observables))


def _logical_cuts(rule, cuts):
    """Expand implicit guards for proofs only, without adding search steps."""
    events, slots = [], []
    for cut in cuts:
        is_event = isinstance(cut, EventCut)
        schema = rule.event_observables if is_event else rule.observables
        target = events if is_event else slots
        inner = cut.cut if is_event else cut
        target.append(inner)
        flag = schema[inner.observable_index].validity_column
        if flag is not None:
            index = next(i for i, spec in enumerate(schema) if spec.name == flag)
            target.append(CategoryCut(index, {1}))
    return (tuple(EventCut(cut) for cut in _normalize_cuts(events, rule.event_observables))
            + _normalize_cuts(slots, rule.observables))


def normalize_event_rule(rule):
    """Event-only compatibility helper; reject object predicates explicitly."""
    if any(not isinstance(cut, EventCut) for cut in rule.cuts):
        raise ValueError("normalize_event_rule requires event-only rules.")
    return normalize_rule(rule)


def _inner(cut):
    return cut.cut if isinstance(cut, EventCut) else cut


def _cut_implies(strong, weak):
    if strong.observable_index != weak.observable_index or _family(strong) != _family(weak):
        return False
    if isinstance(strong, CategoryCut):
        return strong.accepted_codes <= weak.accepted_codes
    if _family(strong) == "lower":
        tighter = strong.threshold > weak.threshold
    else:
        tighter = strong.threshold < weak.threshold
    return tighter or (strong.threshold == weak.threshold and (_closed(weak) or not _closed(strong)))


def rule_implies(strong_cuts, weak_cuts):
    """Compare predicates within scopes; object counts are checked separately."""
    return all(any(isinstance(strong, EventCut) == isinstance(weak, EventCut)
                   and _cut_implies(_inner(strong), _inner(weak)) for strong in strong_cuts)
               for weak in weak_cuts)


def _rule_implies(strong, weak, normalized):
    a, b = normalized[strong.rule_id], normalized[weak.rule_id]
    if any(not isinstance(cut, EventCut) for cut in b):
        if (not any(not isinstance(cut, EventCut) for cut in a)
                or strong.minimum_count < weak.minimum_count):
            return False
    # If the weaker rule is event-only, its minimum_count is not evaluated.
    return rule_implies(a, b)


def _signature(cuts):
    return tuple((isinstance(cut, EventCut), cut.observable_index, "category",
                  tuple(sorted(_inner(cut).accepted_codes)))
                 if isinstance(_inner(cut), CategoryCut) else
                 (isinstance(cut, EventCut), cut.observable_index,
                  _inner(cut).comparison.value, _inner(cut).threshold)
                 for cut in cuts)


def merge_equivalent_rules(rules):
    """Normalize cuts and merge equivalents, without pruning strict subsets."""
    rules = list(rules)
    if len({rule.rule_id for rule in rules}) != len(rules):
        raise ValueError("Rule IDs must be unique.")
    if any(not math.isfinite(rule.generation_score) for rule in rules):
        raise ValueError("Rule scores must be finite.")
    if rules and any((rule.event_observables != rules[0].event_observables
                      or rule.observables != rules[0].observables) for rule in rules):
        raise ValueError("Rules must share the same object and event observable schemas.")
    normalized = {rule.rule_id: normalize_rule(rule) for rule in rules}
    decisions = {rule.rule_id: {"status": "retained", "pruned_by_rule_id": None, "score_gap": None}
                 for rule in rules}
    groups = {}
    for rule in rules:
        count = (rule.minimum_count
                 if any(not isinstance(cut, EventCut) for cut in rule.cuts) else None)
        groups.setdefault((count, _signature(_logical_cuts(rule, normalized[rule.rule_id]))), []).append(rule)
    representatives = []
    for group in groups.values():
        representative = min(group, key=lambda rule: (rule.depth, -rule.generation_score, rule.rule_id))
        representatives.append(representative)
        for rule in group:
            if rule.rule_id != representative.rule_id:
                decisions[rule.rule_id] = {
                    "status": "equivalent_rule", "pruned_by_rule_id": representative.rule_id,
                    "score_gap": representative.generation_score - rule.generation_score,
                }
    for decision in decisions.values():
        decision["retained_witness_rule_id"] = decision["pruned_by_rule_id"]
    representatives.sort(key=lambda rule: (-rule.generation_score, rule.rule_id))
    return RulePruningResult(tuple(representatives), decisions, normalized)


def prune_stricter_rules(result, *, min_score_improvement):
    """Discard strict logical subsets unless they improve a retained looser rule.

    Looser rules are examined first, even when stricter rules score higher.
    Final output is still score-ranked. Equality at the improvement threshold
    is pruned, with 1e-12 roundoff tolerance. Existing pruning audits are kept.
    """
    if not math.isfinite(min_score_improvement) or min_score_improvement < 0:
        raise ValueError("min_score_improvement must be finite and nonnegative.")
    normalized = result.normalized_cuts
    decisions = {rule_id: dict(decision) for rule_id, decision in result.decisions.items()}
    rules = list(result.retained)
    logical = {rule.rule_id: _logical_cuts(rule, normalized[rule.rule_id]) for rule in rules}

    def stricter(strong, weak):
        return (_rule_implies(strong, weak, logical)
                and not _rule_implies(weak, strong, logical))

    # Counting strict ancestors gives a looser-first topological ordering:
    # any strict subset has all its parent's ancestors plus the parent itself.
    ancestors = {rule.rule_id: sum(stricter(rule, other) for other in rules) for rule in rules}
    rules.sort(key=lambda rule: (ancestors[rule.rule_id], -rule.generation_score, rule.rule_id))
    retained = []
    for rule in rules:
        witnesses = []
        for previous in retained:
            improvement = rule.generation_score - previous.generation_score
            if (stricter(rule, previous) and
                    (improvement < min_score_improvement or
                     math.isclose(improvement, min_score_improvement, rel_tol=1e-12, abs_tol=1e-12))):
                witnesses.append(previous)
        if witnesses:
            witness = max(witnesses, key=lambda item: (item.generation_score, -item.rule_id))
            decisions[rule.rule_id] = {
                "status": "stricter_score_improvement", "pruned_by_rule_id": witness.rule_id,
                "score_gap": witness.generation_score - rule.generation_score,
                "score_improvement": rule.generation_score - witness.generation_score,
            }
        else:
            retained.append(rule)
    # A nesting/equivalence witness may have been removed by this new stage.
    for decision in decisions.values():
        witness = decision["pruned_by_rule_id"]
        while witness is not None and decisions[witness]["pruned_by_rule_id"] is not None:
            witness = decisions[witness]["pruned_by_rule_id"]
        decision["retained_witness_rule_id"] = witness
    retained.sort(key=lambda rule: (-rule.generation_score, rule.rule_id))
    return RulePruningResult(tuple(retained), decisions, normalized)


def prune_rule_candidates(
    rules, *, stricter_score_improvement=None, max_reference_acceptance=None,
    similarity_filter=True, signal_similarity_threshold=0.95,
    background_similarity_threshold=0.95,
):
    """Select completed rules without changing the original candidate pool.

    Rate filtering precedes logical pruning; optional Jaccard pruning is last.
    Logical normalization/containment separates event and same-slot predicates,
    and respects minimum object counts. None disables logical pruning.
    Acceptance limits are per-rule weighted training fractions, not OR budgets.
    """
    if stricter_score_improvement is not None and (
            not math.isfinite(stricter_score_improvement) or stricter_score_improvement < 0):
        raise ValueError("stricter_score_improvement must be finite and nonnegative or None.")
    if max_reference_acceptance is not None and (
            not math.isfinite(max_reference_acceptance) or not 0 <= max_reference_acceptance <= 1):
        raise ValueError("max_reference_acceptance must be in [0, 1] or None.")
    if not isinstance(similarity_filter, bool):
        raise TypeError("similarity_filter must be a boolean.")
    for name, threshold in (
        ("signal_similarity_threshold", signal_similarity_threshold),
        ("background_similarity_threshold", background_similarity_threshold),
    ):
        if not 0 <= threshold <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    rules = list(rules)
    if len({rule.rule_id for rule in rules}) != len(rules):
        raise ValueError("Rule IDs must be unique.")
    logical = stricter_score_improvement is not None
    normalized = {
        rule.rule_id: normalize_rule(rule) if logical else rule.cuts
        for rule in rules
    }
    decisions = {
        rule.rule_id: {"status": "retained", "pruned_by_rule_id": None,
                       "score_gap": None, "retained_witness_rule_id": None}
        for rule in rules
    }
    eligible = []
    for rule in rules:
        if (max_reference_acceptance is not None
                and rule.statistics.false_positive_rate > max_reference_acceptance):
            decisions[rule.rule_id]["status"] = "reference_rate_limit"
        else:
            eligible.append(rule)
    if logical:
        if any(not math.isfinite(rule.generation_score) for rule in rules):
            raise ValueError("Rule scores must be finite.")
        merged = merge_equivalent_rules(eligible)
        decisions.update(merged.decisions)
        result = prune_stricter_rules(
            RulePruningResult(merged.retained, decisions, normalized),
            min_score_improvement=stricter_score_improvement,
        )
    else:
        result = RulePruningResult(tuple(eligible), decisions, normalized)
    if similarity_filter:
        retained = []
        for rule in result.retained:
            if rule.signal_selection is None or rule.background_selection is None:
                raise RuntimeError("Candidate rules do not contain training masks; fit the model before pruning")
            witness = next((previous for previous in retained
                            if _jaccard(rule.signal_selection, previous.signal_selection)
                            >= signal_similarity_threshold
                            and _jaccard(rule.background_selection, previous.background_selection)
                            >= background_similarity_threshold), None)
            if witness is None:
                retained.append(rule)
            else:
                result.decisions[rule.rule_id] = {
                    "status": "similarity", "pruned_by_rule_id": witness.rule_id,
                    "score_gap": witness.generation_score - rule.generation_score,
                }
        result.retained = tuple(retained)
        for decision in result.decisions.values():
            witness_id = decision["pruned_by_rule_id"]
            while witness_id is not None and result.decisions[witness_id]["pruned_by_rule_id"] is not None:
                witness_id = result.decisions[witness_id]["pruned_by_rule_id"]
            decision["retained_witness_rule_id"] = witness_id
    return result


def _jaccard(left, right):
    intersection = np.count_nonzero(left & right)
    union = np.count_nonzero(left | right)
    return intersection / union if union else 1.0
