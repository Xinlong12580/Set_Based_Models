"""Named, serializable selections with adjustable thresholds."""

from copy import deepcopy
import json
import operator
from pathlib import Path

import numpy as np

from .data import as_batch
from .physics import group, ht, matched_groups, multi_group


COMPARISONS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le}


def resolve(value, thresholds, settings):
    if isinstance(value, str) and value.startswith("$"):
        return thresholds[value[1:]]
    if isinstance(value, str) and value.startswith("@"):
        return settings[value[1:]] if settings["isolation_enabled"] else None
    if isinstance(value, list):
        return [resolve(item, thresholds, settings) for item in value]
    if isinstance(value, dict):
        return {key: resolve(item, thresholds, settings) for key, item in value.items()}
    return value


def evaluate_selection(selection, batch, thresholds, settings):
    operation = selection["op"]
    if operation in ("all", "any"):
        bits = [evaluate_selection(part, batch, thresholds, settings) for part in selection["conditions"]]
        return (np.logical_and if operation == "all" else np.logical_or).reduce(bits)
    options = resolve({key: value for key, value in selection.items() if key != "op"}, thresholds, settings)
    if operation == "event":
        return COMPARISONS[options["comparison"]](batch.event(options["field"]), options["threshold"])
    if operation == "ht":
        return ht(batch, options["pt_min"], options.get("eta_max", 2.4)) >= options["threshold"]
    functions = {"group": group, "multi_group": multi_group, "matched_groups": matched_groups}
    return functions[operation](batch, **options)


class Trigger:
    def __init__(self, definition):
        self.definition = deepcopy(definition)
        if self.definition.get("format") == "collide2v_erl_trigger":
            self.definition.update(format="collide2v_trigger", type="erl")
        if self.definition.get("format") != "collide2v_trigger" or self.definition.get("schema_version") != 1:
            raise ValueError("Unsupported trigger format/version.")
        self.name = self.definition["name"]
        self.default_thresholds = dict(self.definition["default_thresholds"])
        self.target_rate_khz = self.definition.get("target_rate_khz")
        self.slot_source = self.definition.get("slot_source")
        self.unavailable_reason = self.definition.get("unavailable_reason")
        self.tunable = bool(self.default_thresholds) and self.unavailable_reason is None

    @classmethod
    def from_dict(cls, definition):
        if definition.get("type") == "erl" or definition.get("format") == "collide2v_erl_trigger":
            from .erl import ERLTrigger
            return ERLTrigger(definition)
        return cls(definition)

    def to_dict(self):
        return deepcopy(self.definition)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n")

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    def evaluate(self, data, thresholds=None, *, object_selection=None):
        if self.unavailable_reason is not None:
            raise NotImplementedError(self.unavailable_reason)
        parameters = {**self.default_thresholds, **(thresholds or {})}
        settings = {"isolation_enabled": True, "electron_iso_max": 0.15, "photon_iso_max": 0.15,
                    **(object_selection or {})}
        return evaluate_selection(self.definition["selection"], as_batch(data), parameters, settings)

    __call__ = evaluate
