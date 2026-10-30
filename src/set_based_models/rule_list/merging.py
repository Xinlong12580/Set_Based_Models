"""Pool independent searches without comparing their training statistics."""

from dataclasses import replace

from .serialization import PARAMETERS, _plain_json
from .similarity import merged_jaccard_matrices


def merge_models(models, model_type, *, scoring_function, source_names=None):
    models = list(models)
    if not models:
        raise ValueError("Provide at least one model to merge")
    names = [None] * len(models) if source_names is None else list(source_names)
    if len(names) != len(models):
        raise ValueError("source_names must match the number of models")
    first = models[0]
    for model in models:
        if model.observables != first.observables or model.event_observables != first.event_observables:
            raise ValueError("Merged models must have identical ordered object and event schemas, "
                             "including categories and validity guards")

    combined = model_type(
        first.observables, event_observables=first.event_observables,
        max_rule_depth=max((rule.depth for model in models for rule in model.candidate_rules_), default=1),
        max_candidate_rules=None, scoring_function=scoring_function,
    )
    sources, origins = [], []
    for model, name in zip(models, names):
        offset = len(sources)
        provenance = model.merge_provenance_
        if provenance is None:
            sources.append({"name": name, "input_mode": model.input_mode_,
                            "statistics_source": model.statistics_source_,
                            "configuration": {key: getattr(model, key) for key in PARAMETERS}})
            source_origins = [{"source_index": 0, "rule_id": rule.rule_id}
                              for rule in model.candidate_rules_]
        else:
            # Flatten repeated merges so every rule still identifies its original search.
            sources.extend(provenance["sources"])
            source_origins = provenance["rules"]
        for rule, origin in zip(model.candidate_rules_, source_origins):
            rule_id = len(combined.candidate_rules_)
            combined.candidate_rules_.append(replace(
                rule, rule_id=rule_id, observables=combined.observables,
                event_observables=combined.event_observables,
                signal_selection=None, background_selection=None,
            ))
            origins.append({"source_index": offset + origin["source_index"], "rule_id": origin["rule_id"]})
    combined.merge_provenance_ = _plain_json({"sources": sources, "rules": origins})
    combined.input_mode_ = ("event_only" if not combined.observables else
                            "object_only" if not combined.event_observables else "combined")
    combined.statistics_source_ = "merged"
    combined.signal_jaccard_, combined.background_jaccard_, combined.jaccard_blocks_ = merged_jaccard_matrices(models)
    if combined.signal_jaccard_ is not None:
        combined.jaccard_source_ = "block_diagonal"
    return combined
