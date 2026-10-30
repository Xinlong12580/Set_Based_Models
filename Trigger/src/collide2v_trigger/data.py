"""Read converted PT dictionaries, file lists, and chunk manifests."""

from glob import glob
import json
from pathlib import Path

import numpy as np
import torch


def array(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


class Objects:
    def __init__(self, values, columns, schema, present):
        self.values = values
        self.columns = {name: index for index, name in enumerate(columns)}
        self.schema = {entry["name"]: entry for entry in schema}
        self.present = present

    def __len__(self):
        return self.values.shape[1]

    def field(self, name):
        return self.values[..., self.columns[name]]

    def valid(self, name):
        valid = self.present & np.isfinite(self.field(name))
        flag = self.schema.get(name, {}).get("validity_column")
        if flag is not None:
            valid &= self.field(flag) == 1
        return valid

    def eligible(self, eta_max=2.4, isolation=None):
        selected = self.valid("pt") & self.valid("eta") & (np.abs(self.field("eta")) < eta_max)
        if isolation is not None:
            selected &= self.valid("isolation_rho_corr") & (self.field("isolation_rho_corr") <= isolation)
        return selected


class TensorBatch:
    def __init__(self, data):
        if isinstance(data, (str, Path)):
            data = torch.load(data, map_location="cpu", weights_only=False, mmap=True)
        self.data = data
        self.values = array(data["obj"]).astype(np.float64, copy=False)
        self.size = len(self.values)
        columns = data["obj_columns"]
        self.columns = columns
        self.schema = data["obj_feature_schema"]
        pt = self.values[..., columns.index("pt")]
        self.present = np.isfinite(pt) & (pt > 0)
        if "obj_slot_mask" in data:
            self.present &= array(data["obj_slot_mask"]).astype(bool)
        spec = next(entry for entry in self.schema if entry["name"] == "type_id")
        self.type_codes = {label: int(code) for code, label in spec["categories"].items()}
        self._objects = {}

    def objects(self, kind):
        if kind not in self._objects:
            types = self.values[..., self.columns.index("type_id")]
            selected = self.present & (types == self.type_codes[kind])
            slots = np.any(selected, axis=0)
            self._objects[kind] = Objects(self.values[:, slots], self.columns, self.schema, selected[:, slots])
        return self._objects[kind]

    def event(self, name):
        names = self.data.get("event_feature_columns", [])
        if name in names:
            values = array(self.data["event_features"])[:, names.index(name)].astype(np.float64, copy=False)
            valid = np.isfinite(values)
            schema = {entry["name"]: entry for entry in self.data.get("event_feature_schema", [])}
            flag = schema.get(name, {}).get("validity_column")
            if flag is not None:
                valid &= array(self.data["event_features"])[:, names.index(flag)] == 1
        else:
            native_name = "puppi_met" if name == "met" else name
            index = self.data["native_event_info_columns"].index(native_name)
            values = array(self.data["native_event_info"])[:, index].astype(np.float64, copy=False)
            valid = array(self.data["native_event_info_validity"])[:, index].astype(bool) & np.isfinite(values)
        return np.where(valid, values, np.nan)


def as_batch(data):
    return data if isinstance(data, TensorBatch) else TensorBatch(data)


def input_files(inputs):
    if isinstance(inputs, (str, Path)):
        inputs = [inputs]
    paths = []
    for source in inputs:
        for path in sorted(glob(str(source))):
            path = Path(path)
            if path.suffix == ".json":
                manifest = json.loads(path.read_text())
                paths.extend(path.parent / entry["path"] for entry in manifest["files"])
            else:
                paths.append(path)
    if not paths:
        raise FileNotFoundError(f"No PT inputs matched: {inputs}")
    return list(dict.fromkeys(path.resolve() for path in paths))


def iter_batches(inputs, batch_size=10000, max_events=None, *, with_pairs=False):
    remaining = max_events
    for path in input_files(inputs):
        data = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        count = len(data["obj"]) if remaining is None else min(len(data["obj"]), remaining)
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            part = {key: value[start:stop] if isinstance(value, torch.Tensor) else value
                    for key, value in data.items() if with_pairs or key not in ("pair_obj", "pair_slot_mask")}
            yield TensorBatch(part)
        if remaining is not None:
            remaining -= count
            if remaining <= 0:
                break
