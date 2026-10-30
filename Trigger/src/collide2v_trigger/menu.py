"""A collection of triggers, independent of any concrete menu."""

import json
from pathlib import Path

import numpy as np

from .data import as_batch
from .trigger import Trigger


class TriggerMenu:
    def __init__(self, source=None, threshold_overrides=None, *, isolation=None,
                 electron_iso_max=None, photon_iso_max=None, triggers=()):
        self.isolation = True
        self.electron_iso_max = self.photon_iso_max = 0.15
        self.triggers, self.default_thresholds, self.thresholds, self.target_rates = {}, {}, {}, {}
        if source is not None:
            value = json.loads(Path(source).read_text()) if isinstance(source, (str, Path)) else source
            if value.get("format") != "collide2v_trigger_menu" or value.get("schema_version") != 2:
                from .legacy import upgrade_menu
                value = upgrade_menu(value)
            settings = value["object_selection"]
            self.isolation = settings["isolation_enabled"]
            self.electron_iso_max = settings["electron_iso_max"]
            self.photon_iso_max = settings["photon_iso_max"]
            for entry in value["triggers"]:
                definition = {key: item for key, item in entry.items() if key != "thresholds"}
                self.add_trigger(Trigger.from_dict(definition))
                self.thresholds[entry["name"]].update(entry.get("thresholds", {}))
        for trigger in triggers:
            self.add_trigger(trigger)
        if isolation is not None:
            self.isolation = isolation
        if electron_iso_max is not None:
            self.electron_iso_max = electron_iso_max
        if photon_iso_max is not None:
            self.photon_iso_max = photon_iso_max
        for name, values in (threshold_overrides or {}).items():
            self.thresholds[name].update(values)

    @property
    def all_triggers(self):
        return list(self.triggers)

    @property
    def implementable_triggers(self):
        return [name for name, trigger in self.triggers.items() if trigger.unavailable_reason is None]

    def to_dict(self):
        return {"format": "collide2v_trigger_menu", "schema_version": 2,
                "object_selection": self.object_selection(),
                "triggers": [{**trigger.to_dict(), "thresholds": self.thresholds_for(name),
                              "target_rate_khz": self.target_rate_khz(name)}
                             for name, trigger in self.triggers.items()]}

    @classmethod
    def from_dict(cls, value, **options):
        return cls(value, **options)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n")

    @classmethod
    def load(cls, path, **options):
        return cls(path, **options)

    def thresholds_for(self, name):
        return dict(self.thresholds[name])

    def target_rate_khz(self, name):
        return self.target_rates[name]

    def add_trigger(self, trigger):
        if isinstance(trigger, dict):
            trigger = Trigger.from_dict(trigger)
        if trigger.name in self.triggers:
            raise ValueError(f"Duplicate trigger name: {trigger.name}.")
        self.triggers[trigger.name] = trigger
        self.default_thresholds[trigger.name] = dict(trigger.default_thresholds)
        self.thresholds[trigger.name] = dict(trigger.default_thresholds)
        self.target_rates[trigger.name] = trigger.target_rate_khz

    def trigger_definitions(self):
        return [trigger.to_dict() for trigger in self.triggers.values()]

    def requires_pairs(self, triggers):
        return any(self.triggers[name].slot_source == "pair_obj" for name in triggers)

    def is_tunable(self, name):
        return self.triggers[name].tunable

    def object_selection(self):
        return {"isolation_enabled": self.isolation, "electron_iso_max": self.electron_iso_max,
                "photon_iso_max": self.photon_iso_max}

    def menu_summary(self):
        return [{"name": name, "target_rate_khz": self.target_rate_khz(name),
                 "thresholds": self.thresholds_for(name), "supported": trigger.unavailable_reason is None,
                 "tunable": trigger.tunable, "unavailable_reason": trigger.unavailable_reason}
                for name, trigger in self.triggers.items()]

    def evaluate(self, name, data, thresholds=None):
        parameters = {**self.thresholds[name], **(thresholds or {})}
        return self.triggers[name].evaluate(data, parameters, object_selection=self.object_selection())

    def trigger_bits(self, data, triggers=None):
        batch = as_batch(data)
        names = self.implementable_triggers if triggers is None else triggers
        bits = {name: self.evaluate(name, batch) for name in names}
        bits["passes_any"] = np.logical_or.reduce(list(bits.values())) if bits else np.zeros(batch.size, dtype=bool)
        return bits
