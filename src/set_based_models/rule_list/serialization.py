"""Versioned, declarative ERL artifacts: no training arrays or Python objects."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import warnings

import numpy as np

from ..core.data import ObservableSpec
from ..core.predicates import CategoryCut, EventCut, ThresholdCut
from .pruning import RulePruningResult
from .rules import ExistentialRule, RuleStep
from .statistics import RuleStatistics
from .similarity import jaccard_record, restore_jaccard


MODEL_FORMAT = "set_based_models_erl"
SCHEMA_VERSION = 1
PARAMETERS = (
    "max_rule_depth", "beam_width", "max_candidate_rules", "max_thresholds",
    "minimum_count", "min_signal_efficiency", "beam_signal_similarity_threshold",
    "beam_background_similarity_threshold", "beam_similarity_filter",
    "allow_validity_cuts", "object_cut_priority", "beam_novelty_weight",
)


def _json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__} as model JSON")


def _plain_json(value):
    # Also reject nonfinite numbers and detach the artifact from mutable inputs.
    return json.loads(json.dumps(value, default=_json_scalar, allow_nan=False))


def _observable_record(spec):
    categories = (None if spec.categories is None else
                  [{"code": code, "label": spec.categories[code]} for code in sorted(spec.categories)])
    return {"name": spec.name, "kind": spec.kind.value,
            "categories": categories, "validity_column": spec.validity_column}


def _observable(record):
    categories = record["categories"]
    if categories is not None:
        codes = {entry["code"]: entry["label"] for entry in categories}
        if len(codes) != len(categories):
            raise ValueError("Duplicate category codes in model schema")
        categories = codes
    return ObservableSpec(record["name"], record["kind"], categories,
                          validity_column=record["validity_column"])


def _cut_record(cut, objects, events):
    is_event = isinstance(cut, EventCut)
    inner = cut.cut if is_event else cut
    schema = events if is_event else objects
    inner.validate(schema)
    record = {"scope": "event" if is_event else "slot",
              "observable": schema[inner.observable_index].name}
    if isinstance(inner, ThresholdCut):
        record.update(type="threshold", comparison=inner.comparison.value, threshold=inner.threshold)
    elif isinstance(inner, CategoryCut):
        record.update(type="category", accepted_codes=sorted(inner.accepted_codes))
    else:
        raise TypeError(f"Unsupported cut type: {type(inner).__name__}")
    return record


def _cut(record, objects, events):
    scope = record["scope"]
    if scope not in ("slot", "event"):
        raise ValueError(f"Unknown cut scope {scope!r}")
    schema = events if scope == "event" else objects
    names = [spec.name for spec in schema]
    try:
        index = names.index(record["observable"])
    except ValueError as error:
        raise ValueError(f"Unknown {scope} observable {record['observable']!r}") from error
    if record["type"] == "threshold":
        cut = ThresholdCut(index, record["comparison"], record["threshold"])
    elif record["type"] == "category":
        cut = CategoryCut(index, record["accepted_codes"])
    else:
        raise ValueError(f"Unknown cut type {record['type']!r}")
    cut.validate(schema)
    return EventCut(cut) if scope == "event" else cut


def model_to_dict(model):
    objects, events = model.observables, model.event_observables
    rules = []
    for index, rule in enumerate(model.candidate_rules_):
        if rule.rule_id != index:
            raise ValueError("Candidate rule IDs must match their positions")
        if rule.observables != objects or rule.event_observables != events:
            raise ValueError("Every candidate must use the model's observable schemas")
        rules.append({
            "rule_id": rule.rule_id, "minimum_count": rule.minimum_count,
            "steps": [{"depth": step.depth,
                       "cut": _cut_record(step.cut, objects, events),
                       "statistics": asdict(step.statistics),
                       "generation_score": step.generation_score} for step in rule.steps],
        })
    result = model.pruning_result_
    pruning = None if result is None else {
        "configuration": model.pruning_configuration_,
        "selected_rule_ids": [rule.rule_id for rule in model.selected_rules_],
        "decisions": [{"rule_id": rule_id, "decision": decision}
                      for rule_id, decision in result.decisions.items()],
        "normalized_cuts": [{"rule_id": rule_id,
                             "cuts": [_cut_record(cut, objects, events) for cut in cuts]}
                            for rule_id, cuts in result.normalized_cuts.items()],
    }
    payload = {
        "format": MODEL_FORMAT, "schema_version": SCHEMA_VERSION,
        "observables": [_observable_record(spec) for spec in objects],
        "event_observables": [_observable_record(spec) for spec in events],
        "configuration": {name: getattr(model, name) for name in PARAMETERS},
        "input_mode": model.input_mode_, "statistics_source": model.statistics_source_,
        "rules": rules, "pruning": pruning,
    }
    if model.merge_provenance_ is not None:
        payload["merge_provenance"] = model.merge_provenance_
    if model.self_pruning_provenance_ is not None:
        payload["self_pruning_provenance"] = model.self_pruning_provenance_
    jaccard = jaccard_record(model)
    if jaccard is not None:
        payload["event_jaccard"] = jaccard
    return _plain_json(payload)


def model_from_dict(value, model_type, *, scoring_function):
    if not isinstance(value, dict) or value.get("format") != MODEL_FORMAT:
        raise ValueError(f"Expected a {MODEL_FORMAT!r} model artifact, not a runner report")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported ERL model schema version {value.get('schema_version')!r}")
    value = _plain_json(value)
    objects = tuple(_observable(record) for record in value["observables"])
    events = tuple(_observable(record) for record in value["event_observables"])
    configuration = dict(value["configuration"])
    configuration.setdefault("beam_novelty_weight", 0.0)
    model = model_type(objects, event_observables=events, scoring_function=scoring_function,
                       **{name: configuration[name] for name in PARAMETERS})
    for index, record in enumerate(value["rules"]):
        if record["rule_id"] != index:
            raise ValueError("Candidate rule IDs must match their positions")
        steps = tuple(RuleStep(
            step["depth"], _cut(step["cut"], objects, events),
            RuleStatistics(**step["statistics"]), step["generation_score"],
        ) for step in record["steps"])
        model.candidate_rules_.append(ExistentialRule(
            record["rule_id"], objects, steps, minimum_count=record["minimum_count"],
            event_observables=events,
        ))
    mode = value["input_mode"]
    if mode not in (None, "event_only", "object_only", "combined"):
        raise ValueError(f"Unknown input mode {mode!r}")
    model.input_mode_ = mode
    model.statistics_source_ = value.get("statistics_source", "training" if mode is not None else None)
    provenance = value.get("merge_provenance")
    if provenance is not None:
        if len(provenance["rules"]) != len(model.candidate_rules_):
            raise ValueError("Merge provenance must describe every candidate rule")
        for origin in provenance["rules"]:
            if not 0 <= origin["source_index"] < len(provenance["sources"]) or origin["rule_id"] < 0:
                raise ValueError("Unknown source model or rule in merge provenance")
        model.merge_provenance_ = provenance
    model.self_pruning_provenance_ = value.get("self_pruning_provenance")
    if value.get("event_jaccard") is not None:
        restore_jaccard(model, value["event_jaccard"])
    pruning = value["pruning"]
    if pruning is not None:
        selected = pruning["selected_rule_ids"]
        if len(set(selected)) != len(selected):
            raise ValueError("Selected rule IDs must be unique")
        model.selected_rules_ = [model.get_rule(rule_id) for rule_id in selected]
        decisions = {entry["rule_id"]: entry["decision"] for entry in pruning["decisions"]}
        normalized = {entry["rule_id"]: tuple(_cut(cut, objects, events) for cut in entry["cuts"])
                      for entry in pruning["normalized_cuts"]}
        expected_ids = set(range(len(model.candidate_rules_)))
        if (len(decisions) != len(pruning["decisions"]) or set(decisions) != expected_ids
                or len(normalized) != len(pruning["normalized_cuts"]) or set(normalized) != expected_ids):
            raise ValueError("Pruning records must describe each candidate rule exactly once")
        model.pruning_result_ = RulePruningResult(tuple(model.selected_rules_), decisions, normalized)
        model.pruning_configuration_ = pruning["configuration"]
    return model


def save_model(model, filename, *, overwrite):
    payload = json.dumps(model_to_dict(model), indent=2, allow_nan=False) + "\n"
    path = Path(filename).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if not overwrite:
            raise FileExistsError(f"Model file already exists: {path}")
        warnings.warn(f"Overwriting existing ERL model {path}", UserWarning, stacklevel=2)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def load_model(filename, model_type, *, scoring_function):
    with Path(filename).expanduser().open(encoding="utf-8") as handle:
        value = json.load(handle)
    return model_from_dict(value, model_type, scoring_function=scoring_function)
