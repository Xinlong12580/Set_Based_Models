"""Validation-based shortening of fixed rules, before final list selection."""

from copy import deepcopy
from dataclasses import replace
from itertools import product

import numpy as np

from ..core.data import SetInput
from ..core.predicates import CategoryCut, EventCut
from .generation import _object_priority_progress, _statistics, _weights
from .pruning import _signature, normalize_rule, prune_rule_candidates
from .serialization import PARAMETERS


def recalculate_rules(rules, signal, background, scorer):
    sw, bw = _weights(signal), _weights(background)
    recalculated = []
    for rule in rules:
        sm, bm = signal.slot_mask.copy(), background.slot_mask.copy()
        ss, bs = np.ones(signal.n_samples, bool), np.ones(background.n_samples, bool)
        steps = []
        for depth, step in enumerate(rule.steps, 1):
            cut = step.cut
            if isinstance(cut, EventCut):
                ss &= cut.evaluate(signal)
                bs &= cut.evaluate(background)
            else:
                if not signal.n_slots or not background.n_slots:
                    raise ValueError("Object tensors are required to recalculate object-level candidates")
                sm &= cut.evaluate(signal)
                bm &= cut.evaluate(background)
                ss &= np.count_nonzero(sm, axis=1) >= rule.minimum_count
                bs &= np.count_nonzero(bm, axis=1) >= rule.minimum_count
            statistics = _statistics(ss, bs, sw, bw, depth, rule.minimum_count)
            score = float(scorer(statistics))
            if not np.isfinite(score):
                raise ValueError("scoring_function must return finite scores")
            steps.append(replace(step, depth=depth, statistics=statistics, generation_score=score))
        recalculated.append(replace(rule, steps=tuple(steps), signal_selection=ss, background_selection=bs))
    return recalculated


def shortened_rules(rule):
    """Keep, drop, or replace a guarded measurement with its True guard."""
    choices = []
    for step in rule.steps:
        cut = step.cut
        is_event = isinstance(cut, EventCut)
        schema = rule.event_observables if is_event else rule.observables
        flag = schema[cut.observable_index].validity_column
        options = [None]
        if flag is not None:
            index = next(i for i, spec in enumerate(schema) if spec.name == flag)
            guard = CategoryCut(index, {1})
            options.append(replace(step, cut=EventCut(guard) if is_event else guard))
        options.append(step)
        choices.append(options)
    variants = []
    for selection in product(*choices):
        steps = tuple(replace(step, depth=depth) for depth, step in enumerate(
            (step for step in selection if step is not None), 1))
        if steps:
            variants.append(replace(rule, rule_id=len(variants), steps=steps,
                                    signal_selection=None, background_selection=None))
    return variants


def self_prune_candidates(model, signal, background, *, signal_slot_mask, background_slot_mask,
                          signal_weight, background_weight, signal_event_features,
                          background_event_features, scoring_function, stricter_score_improvement,
                          max_reference_acceptance, require_complete_priority, rule_ids, compute_jaccard):
    scorer = model.scoring_function if scoring_function is None else scoring_function
    signal = SetInput.from_arrays(signal, model.observables, slot_mask=signal_slot_mask,
                                 sample_weight=signal_weight, event_features=signal_event_features,
                                 event_observables=model.event_observables)
    background = SetInput.from_arrays(background, model.observables, slot_mask=background_slot_mask,
                                     sample_weight=background_weight, event_features=background_event_features,
                                     event_observables=model.event_observables)
    rules = model.candidate_rules_ if rule_ids is None else [model.get_rule(i) for i in rule_ids]
    configuration = {name: getattr(model, name) for name in PARAMETERS}
    configuration["max_candidate_rules"] = None
    refined = type(model)(model.observables, event_observables=model.event_observables,
                          scoring_function=scorer, **configuration)
    provenance = deepcopy(model.merge_provenance_)
    if provenance is None:
        provenance = {"sources": [{"name": None, "input_mode": model.input_mode_,
                                   "statistics_source": model.statistics_source_,
                                   "configuration": {name: getattr(model, name) for name in PARAMETERS}}],
                      "rules": [{"source_index": 0, "rule_id": rule.rule_id}
                                for rule in model.candidate_rules_]}
    sources, origins, parents, seen = provenance["sources"], [], [], {}
    evaluated = retained_count = 0
    for rule in rules:
        origin = provenance["rules"][rule.rule_id]
        priority = tuple(sources[origin["source_index"]]["configuration"]["object_cut_priority"])
        variants = recalculate_rules(shortened_rules(rule), signal, background, scorer)
        evaluated += len(variants)
        retained = prune_rule_candidates(
            variants, similarity_filter=False, stricter_score_improvement=stricter_score_improvement,
            max_reference_acceptance=max_reference_acceptance,
            required_object_cut_priority=priority if require_complete_priority else (),
        ).retained
        retained_count += len(retained)
        for candidate in retained:
            progress = _object_priority_progress(candidate.cuts, priority, model.observables)
            count = candidate.minimum_count if any(not isinstance(cut, EventCut) for cut in candidate.cuts) else None
            key = (count, _signature(normalize_rule(candidate)), priority, progress)
            if key not in seen:
                index = len(refined.candidate_rules_)
                seen[key] = index
                refined.candidate_rules_.append(replace(candidate, rule_id=index))
                origins.append(dict(origin))
                parents.append([rule.rule_id])
            else:
                index = seen[key]
                if rule.rule_id not in parents[index]:
                    parents[index].append(rule.rule_id)
                previous = refined.get_rule(index)
                if (candidate.depth, -candidate.generation_score) < (previous.depth, -previous.generation_score):
                    refined.candidate_rules_[index] = replace(candidate, rule_id=index)
                    origins[index] = dict(origin)
    refined.merge_provenance_ = deepcopy({"sources": sources, "rules": origins})
    refined.self_pruning_provenance_ = {
        "configuration": {"stricter_score_improvement": stricter_score_improvement,
                          "max_reference_acceptance": max_reference_acceptance,
                          "require_complete_priority": require_complete_priority},
        "source_rule_ids": [rule.rule_id for rule in rules],
        "evaluated_variant_count": evaluated, "locally_retained_variant_count": retained_count,
        "rules": [{"parent_rule_ids": ids} for ids in parents],
    }
    refined.input_mode_ = ("event_only" if not signal.n_slots else
                           "object_only" if not signal.event_observables else "combined")
    refined.statistics_source_ = "recalculated"
    if compute_jaccard:
        refined.compute_jaccard_matrices()
    return refined
