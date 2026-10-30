"""Translate structured ERL rules into standalone tensor triggers."""

from copy import deepcopy
import json
import math
import operator
from pathlib import Path
import re
import warnings

import numpy as np

from .data import array, as_batch
from .trigger import Trigger


class RuleParseError(ValueError):
    """The requested rule or translation cannot be represented faithfully."""


class _RuleTuningError(RuleParseError):
    """The rule may be evaluated literally, but the requested tuning is unavailable."""


COMPARISONS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le}
PT_COLUMNS = {"electrons": "L1T_Electron_PT", "muons": "L1T_MuonTight_PT",
              "photons": "L1T_PhotonTight_PT", "jets": "L1T_JetPuppiAK4_PT"}


def plain(value):
    return json.loads(json.dumps(value, allow_nan=False))


def categories(spec):
    entries = spec.get("categories")
    if isinstance(entries, list):
        return {int(entry["code"]): entry["label"] for entry in entries}
    return {} if entries is None else {int(code): label for code, label in entries.items()}


def rule_record(rule, rule_id=None):
    if isinstance(rule, (str, Path)):
        path = Path(rule)
        if not path.is_file():
            raise RuleParseError("Supply an ERL model JSON path or a structured rule, not printed rule text.")
        rule = json.loads(path.read_text())
    if isinstance(rule, dict):
        if "rules" in rule:
            if rule.get("format") != "set_based_models_erl" or rule.get("schema_version") != 1:
                raise RuleParseError("Unsupported ERL model format/version.")
            if rule_id is None and len(rule["rules"]) != 1:
                raise RuleParseError("Specify rule_id to select exactly one rule from the model.")
            matches = [entry for entry in rule["rules"] if rule_id is None or entry["rule_id"] == rule_id]
            if len(matches) != 1:
                raise RuleParseError(f"Cannot select rule {rule_id}.")
            record = matches[0]
        else:
            record = rule.get("rule", rule)
        return deepcopy(record), rule.get("observables", []), rule.get("event_observables", [])
    if not hasattr(rule, "steps"):
        raise RuleParseError("Expected an ExistentialRule or a structured ERL JSON record.")

    def schema(specs):
        return [{"name": spec.name, "kind": spec.kind.value, "categories": spec.categories,
                 "validity_column": spec.validity_column} for spec in specs]

    objects, events = schema(rule.observables), schema(rule.event_observables)
    steps = []
    for step in rule.steps:
        is_event = type(step.cut).__name__ == "EventCut"
        cut = step.cut.cut if is_event else step.cut
        specs = events if is_event else objects
        record = {"scope": "event" if is_event else "slot", "observable": specs[cut.observable_index]["name"]}
        if type(cut).__name__ == "ThresholdCut":
            record.update(type="threshold", comparison=cut.comparison.value, threshold=float(cut.threshold))
        elif type(cut).__name__ == "CategoryCut":
            record.update(type="category", accepted_codes=sorted(cut.accepted_codes))
        else:
            raise RuleParseError(f"Unsupported cut type: {type(cut).__name__}.")
        steps.append({"cut": record})
    return {"rule_id": rule.rule_id, "minimum_count": rule.minimum_count, "steps": steps}, objects, events


def feature_schema(batch, scope, slot_source):
    if scope == "slot" and slot_source is None:
        return {}
    key = "event_feature_schema" if scope == "event" else slot_source + "_feature_schema"
    return {entry["name"]: entry for entry in batch.data.get(key, [])}


def check_schema(expected, actual):
    left, right = plain(expected), plain(actual)
    left_categories, right_categories = categories(left), categories(right)
    left.pop("categories", None)
    right.pop("categories", None)
    if left != right or set(left_categories.values()) != set(right_categories.values()):
        raise RuleParseError(f"Feature definition changed: {expected['name']}.")


def is_energy(scope, name):
    if scope == "slot":
        return name in {"pt", "pt_1", "pt_2", "min_pt", "sum_pt", "pair_pt"}
    return name == "met" or name.startswith(("sum_pt_", "leading_pt_", "subleading_pt_"))


def count_rank(cut, spec, event_schema):
    definition = spec.get("definition", {})
    kind = next((kind for kind, column in PT_COLUMNS.items() if definition.get("raw_column") == column), None)
    if (kind is None or definition.get("comparison") != ">" or not definition.get("finite_pt_required")
            or definition.get("object_truncation") is not False or definition.get("additional_cuts", "unknown") is not None):
        raise _RuleTuningError(f"{cut['observable']}: missing exact full-collection multiplicity definition.")
    threshold = float(cut["threshold"])
    rank = math.ceil(threshold) if cut["comparison"] == ">=" else math.floor(threshold) + 1
    if rank not in (1, 2):
        raise _RuleTuningError(f"{cut['observable']}: rank {rank} has no saved full-precision ranked PT; "
                             "capped float32 objects cannot guarantee this translation. Select other cuts with tune=[...].")
    floor = definition.get("pt_threshold_GeV")
    if floor is None or not np.isfinite(floor) or floor < 0:
        raise _RuleTuningError(f"{cut['observable']}: invalid embedded PT floor.")
    name = ("leading" if rank == 1 else "subleading") + "_pt_" + kind
    ranked = event_schema.get(name)
    expected = {"raw_column": PT_COLUMNS[kind], "rank": rank, "order": "descending",
                "pt_threshold_GeV": 0, "comparison": ">", "finite_pt_required": True,
                "object_truncation": False, "missing_fill": 0, "additional_cuts": None}
    if ranked is None or ranked.get("units") != "GeV" or any(
            key not in ranked.get("definition", {}) or ranked["definition"][key] != value for key, value in expected.items()):
        raise _RuleTuningError(f"{cut['observable']}: need the matching full-collection {name} definition.")
    return name, float(floor)


class ERLTrigger(Trigger):
    def __init__(self, definition):
        super().__init__(plain(definition))
        self.cut_actions = deepcopy(self.definition["cuts"])
        self.tunable = bool(self.default_thresholds)
        if not self.cut_actions or type(self.definition["minimum_count"]) is not int or self.definition["minimum_count"] < 1:
            raise RuleParseError("Need nonempty rule cuts and a positive integer minimum_count.")
        if self.definition.get("tunable", self.tunable) != self.tunable:
            raise RuleParseError("Tunable status disagrees with the parameter bindings.")
        used = set()
        for cut in self.cut_actions:
            if cut.get("action") not in ("fixed", "threshold", "count_pt") or cut.get("scope") not in ("event", "slot"):
                raise RuleParseError("Unsupported cut action/scope in parsed trigger.")
            if cut.get("type") not in ("threshold", "category") or (cut["type"] == "threshold" and cut.get("comparison") not in COMPARISONS):
                raise RuleParseError("Unsupported predicate in parsed trigger.")
            if cut["action"] != "fixed":
                parameter = cut.get("parameter")
                default = cut.get("default_pt") if cut["action"] == "count_pt" else cut.get("threshold")
                if parameter not in self.default_thresholds or self.default_thresholds[parameter] != default:
                    raise RuleParseError("Parameter binding does not preserve the learned default.")
                if cut["comparison"] not in (">", ">=") or not np.isfinite(default) or default < 0:
                    raise RuleParseError("Tunable predicates must have nonnegative lower momentum bounds.")
                used.add(parameter)
        if used != set(self.default_thresholds):
            raise RuleParseError("Parsed trigger contains unbound tuning parameters.")

    def _values(self, batch, scope, name):
        key = scope + ":" + name
        expected = self.definition["features"].get(key)
        if expected is None:
            raise RuleParseError(f"Missing bound feature definition: {key}.")
        schema = feature_schema(batch, scope, self.slot_source)
        if name not in schema:
            raise RuleParseError(f"Missing {scope} feature {name}.")
        check_schema(expected, schema[name])
        tensor = "event_features" if scope == "event" else self.slot_source
        columns = "event_feature_columns" if scope == "event" else self.slot_source + "_columns"
        if tensor not in batch.data or name not in batch.data[columns]:
            raise RuleParseError(f"Missing tensor column {scope}:{name}.")
        return array(batch.data[tensor])[..., batch.data[columns].index(name)].astype(np.float64, copy=False), schema[name]

    def _predicate(self, batch, cut, thresholds, presence):
        scope, name = cut["scope"], cut["observable"]
        values, spec = self._values(batch, scope, name)
        valid = np.ones(batch.size, dtype=bool) if scope == "event" else presence.copy()
        if values.shape != valid.shape:
            raise RuleParseError(f"Feature shape does not match the event/slot axis: {name}.")
        guard = cut.get("validity_column")
        if guard is not None:
            flags, _ = self._values(batch, scope, guard)
            if np.any(~np.isin(flags[valid], [0, 1])):
                raise RuleParseError(f"Invalid Boolean guard: {guard}.")
            valid &= flags == 1
        if np.any(~np.isfinite(values[valid])):
            raise RuleParseError(f"Nonfinite valid measurement: {name}.")
        if cut["kind"] != "continuous" and np.any(values[valid] != np.floor(values[valid])):
            raise RuleParseError(f"Noninteger discrete feature: {name}.")
        if cut["type"] == "category":
            if cut["kind"] == "categorical":
                actual = categories(spec)
                if set(actual.values()) != set(cut["categories"].values()):
                    raise RuleParseError(f"Category meanings changed: {name}.")
                accepted = [code for code, label in actual.items() if label in cut["accepted_labels"]]
                allowed = list(actual)
            else:
                accepted, allowed = cut["accepted_codes"], [0, 1]
            if np.any(~np.isin(values[valid], allowed)):
                raise RuleParseError(f"Unknown category code: {name}.")
            return valid & np.isin(values, accepted)
        if cut["action"] == "count_pt":
            if array(batch.data["event_features"]).dtype != np.float64:
                raise _RuleTuningError("Count translation requires the saved float64 full-collection ranked PT.")
            ranked, _ = self._values(batch, "event", cut["rank_feature"])
            if np.any(~np.isfinite(ranked)) or np.any(ranked < 0):
                raise _RuleTuningError(f"Invalid ranked PT: {cut['rank_feature']}.")
            original = COMPARISONS[cut["comparison"]](values, cut["threshold"])
            if np.any(valid & (original != (ranked > cut["default_pt"]))):
                raise _RuleTuningError(f"{name}: multiplicity and ranked PT disagree at the learned threshold.")
            return valid & (ranked > thresholds[cut["parameter"]])
        threshold = thresholds[cut["parameter"]] if cut["action"] == "threshold" else cut["threshold"]
        if scope == "slot" and np.ndim(threshold) == 1:
            threshold = np.asarray(threshold)[:, None]
        return valid & COMPARISONS[cut["comparison"]](values, threshold)

    def evaluate(self, data, thresholds=None, *, object_selection=None):
        batch = as_batch(data)
        overrides = {} if thresholds is None else thresholds
        unknown = set(overrides) - set(self.default_thresholds)
        if unknown:
            raise RuleParseError(f"Unknown tunable parameters: {sorted(unknown)}.")
        parameters = {**self.default_thresholds, **overrides}
        for name, value in parameters.items():
            value = np.asarray(value)
            if value.shape not in ((), (batch.size,)) or not np.all(np.isfinite(value)) or np.any(value < 0):
                raise RuleParseError(f"{name}: need a nonnegative finite scalar or one threshold per event.")
        selected = np.ones(batch.size, dtype=bool)
        if self.slot_source == "pair_obj":
            if "pair_obj" not in batch.data or "pair_slot_mask" not in batch.data:
                raise RuleParseError("The rule requires pair_obj and pair_slot_mask.")
            active = array(batch.data["pair_slot_mask"]).astype(bool).copy()
        else:
            active = batch.present.copy()
        presence = active.copy()
        has_slots = False
        for cut in self.definition["cuts"]:
            passed = self._predicate(batch, cut, parameters, presence if cut["scope"] == "slot" else None)
            if cut["scope"] == "event":
                selected &= passed
            else:
                active &= passed
                has_slots = True
        if has_slots:
            selected &= active.sum(axis=1) >= self.definition["minimum_count"]
        return selected

    __call__ = evaluate


def _parse_erl_rule(rule, data, *, rule_id=None, name=None, target_rate_khz=None, tune=None,
                    slot_source=None, require_tunable=True):
    record, objects, events = rule_record(rule, rule_id)
    batch = as_batch(data)
    schemas = {"slot": {spec["name"]: spec for spec in objects},
               "event": {spec["name"]: spec for spec in events}}
    cuts = [deepcopy(step["cut"]) for step in record["steps"]]
    if not cuts or type(record["minimum_count"]) is not int or record["minimum_count"] < 1:
        raise RuleParseError("Need nonempty rule cuts and a positive integer minimum_count.")
    for cut in cuts:
        if cut.get("scope") not in schemas or cut.get("observable") not in schemas[cut["scope"]]:
            raise RuleParseError(f"Unknown cut scope/observable: {cut}.")
    slot_names = {cut["observable"] for cut in cuts if cut["scope"] == "slot"}
    if slot_names:
        candidates = [source for source in ("obj", "pair_obj")
                      if slot_names <= set(batch.data.get(source + "_columns", []))]
        if slot_source is None:
            if len(candidates) != 1:
                raise RuleParseError("Cannot identify object versus pair slots; supply slot_source explicitly.")
            slot_source = candidates[0]
        if slot_source not in candidates:
            raise RuleParseError(f"Missing required features in {slot_source}.")
    elif slot_source is not None:
        raise RuleParseError("An event-only rule does not have a slot_source.")
    data_schemas = {scope: feature_schema(batch, scope, slot_source) for scope in schemas}
    selected_indices = None if tune is None else set(tune)
    if selected_indices is not None and not selected_indices <= set(range(len(cuts))):
        raise RuleParseError("tune must contain existing zero-based cut indices.")
    features, parameters = {}, {}

    def require(scope, feature):
        spec = data_schemas[scope].get(feature)
        if spec is None:
            raise RuleParseError(f"Missing {scope} feature definition: {feature}.")
        features[scope + ":" + feature] = plain(spec)
        return spec

    for index, cut in enumerate(cuts):
        scope, observable = cut["scope"], cut["observable"]
        model_spec = schemas[scope][observable]
        spec = require(scope, observable)
        cut.update(index=index, action="fixed", kind=model_spec["kind"],
                   validity_column=model_spec.get("validity_column"))
        if spec.get("kind") != cut["kind"]:
            raise RuleParseError(f"{observable}: model and data feature kinds differ.")
        guard = cut["validity_column"]
        if guard is not None:
            guard_spec = schemas[scope].get(guard, {})
            if (guard == observable or guard_spec.get("kind") != "boolean" or guard_spec.get("validity_column") is not None
                    or require(scope, guard).get("kind") != "boolean"):
                raise RuleParseError(f"Missing Boolean validity guard {guard}.")
        if cut["type"] == "category":
            if cut["kind"] not in ("boolean", "categorical") or not cut.get("accepted_codes"):
                raise RuleParseError(f"Invalid category predicate: {observable}.")
            cut["categories"] = categories(model_spec)
            if cut["kind"] == "categorical" and len(set(cut["categories"].values())) != len(cut["categories"]):
                raise RuleParseError(f"Ambiguous category labels in {observable}.")
            allowed = cut["categories"] if cut["kind"] == "categorical" else {0: False, 1: True}
            if not set(cut["accepted_codes"]) <= set(allowed):
                raise RuleParseError(f"Unknown category in {observable}.")
            cut["accepted_labels"] = [allowed[code] for code in cut["accepted_codes"]]
            if selected_indices is not None and index in selected_indices:
                raise _RuleTuningError(f"Cut {index}: categorical decisions cannot be momentum-tuned.")
            continue
        if (cut["type"] != "threshold" or cut.get("comparison") not in COMPARISONS
                or cut["kind"] not in ("continuous", "ordinal") or not np.isfinite(cut.get("threshold", np.nan))):
            raise RuleParseError(f"Unsupported threshold predicate: {cut}.")
        object_count = scope == "event" and re.fullmatch(r"n_(electrons|muons|photons|jets)(?:_pt[0-9p]+)?", observable)
        lower = cut["comparison"] in (">", ">=")
        automatic = lower and (object_count or (spec.get("units") == "GeV" and is_energy(scope, observable)))
        should_tune = automatic if selected_indices is None else index in selected_indices
        if not should_tune:
            continue
        parameter = f"cut_{index}_{observable}"
        if object_count and lower:
            ranked, default = count_rank(cut, spec, data_schemas["event"])
            require("event", ranked)
            cut.update(action="count_pt", rank_feature=ranked, default_pt=default)
            parameter += "_pt_min"
        elif lower and cut["kind"] == "continuous" and spec.get("units") == "GeV" and cut["threshold"] >= 0:
            default = float(cut["threshold"])
            cut["action"] = "threshold"
        else:
            raise _RuleTuningError(f"Cut {index} ({observable}): only lower GeV thresholds and exact object-count "
                                 "PT translations support this momentum calibration; embedded PF bins need raw candidates.")
        cut["parameter"] = parameter
        parameters[parameter] = default
    if require_tunable and not parameters:
        raise _RuleTuningError("The rule has no supported tunable momentum threshold.")
    definition = {"format": "collide2v_trigger", "schema_version": 1, "type": "erl",
                  "name": name or f"ERL_rule_{record['rule_id']}", "target_rate_khz": target_rate_khz,
                  "source_rule_id": record["rule_id"], "slot_source": slot_source,
                  "minimum_count": record["minimum_count"], "default_thresholds": parameters,
                  "tunable": bool(parameters), "cuts": cuts, "features": features}
    trigger = ERLTrigger(definition)
    trigger.evaluate(batch)
    return trigger


def parse_erl_rule(rule, data, *, rule_id=None, name=None, target_rate_khz=None, tune=None,
                   slot_source=None, allow_fixed=False):
    """Bind one rule; optionally retain the literal rule when tuning is unavailable."""
    batch = as_batch(data)
    options = {"rule_id": rule_id, "name": name, "target_rate_khz": target_rate_khz, "slot_source": slot_source}
    try:
        return _parse_erl_rule(rule, batch, tune=tune, **options)
    except _RuleTuningError as error:
        if not allow_fixed:
            raise
        trigger = _parse_erl_rule(rule, batch, tune=[], require_tunable=False, **options)
        trigger.definition["fixed_reason"] = str(error)
        warnings.warn(f"Keeping rule {trigger.definition['source_rule_id']} fixed: {error}", UserWarning, stacklevel=2)
        return trigger


def parse_erl_model(model, data, *, rule_ids=None, all_candidates=False,
                    name_prefix="ERL_rule", target_rate_khz=None, tune=None, skip_unsupported=False, allow_fixed=False):
    """Return triggers for retained rules, explicit IDs, or all candidates; strict by default.

    target_rate_khz can be a shared rate or a rule-ID-to-rate dictionary.
    tune optionally maps rule IDs to their adjustable zero-based cut indices.
    skip_unsupported warns and continues after a per-rule RuleParseError.
    allow_fixed retains faithfully evaluable rules without tunable parameters.
    """
    if isinstance(model, (str, Path)):
        model = json.loads(Path(model).read_text())
    if isinstance(model, dict):
        if model.get("format") != "set_based_models_erl" or model.get("schema_version") != 1:
            raise RuleParseError("Unsupported ERL model format/version.")
        records = model["rules"]
        sources = {record["rule_id"]: {"rule": record, "observables": model["observables"],
                                      "event_observables": model["event_observables"]} for record in records}
        pruning = model.get("pruning")
        selected = None if pruning is None else pruning["selected_rule_ids"]
    elif hasattr(model, "candidate_rules_"):
        records = model.candidate_rules_
        sources = {rule.rule_id: rule for rule in records}
        selected = None if model.pruning_result_ is None else [rule.rule_id for rule in model.selected_rules_]
    else:
        raise RuleParseError("Expected an ERL model object, model dictionary, or saved model JSON path.")
    if len(sources) != len(records):
        raise RuleParseError("The model contains duplicate candidate rule IDs.")
    if rule_ids is not None and all_candidates:
        raise RuleParseError("Choose rule_ids or all_candidates, not both.")
    ids = list(rule_ids) if rule_ids is not None else list(sources) if all_candidates else selected
    if ids is None:
        raise RuleParseError("The model has no final selection. Supply rule_ids or all_candidates=True.")
    if len(ids) != len(set(ids)) or set(ids) - set(sources):
        raise RuleParseError(f"Duplicate or missing rule IDs: {ids}.")
    if not ids:
        return []
    batch = as_batch(data)
    triggers = []
    for rule_id in ids:
        rate = target_rate_khz.get(rule_id) if isinstance(target_rate_khz, dict) else target_rate_khz
        indices = None if tune is None else tune.get(rule_id)
        try:
            trigger = parse_erl_rule(sources[rule_id], batch, name=f"{name_prefix}_{rule_id}",
                                     target_rate_khz=rate, tune=indices, allow_fixed=allow_fixed)
        except RuleParseError as error:
            if skip_unsupported:
                warnings.warn(f"Skipping rule {rule_id}: {error}", UserWarning, stacklevel=2)
                continue
            raise RuleParseError(f"Rule {rule_id}: {error}") from error
        triggers.append(trigger)
    return triggers
