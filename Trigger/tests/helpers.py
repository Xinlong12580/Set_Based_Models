import numpy as np
import torch


KINDS = ("jets", "muons", "electrons", "photons")
COLUMNS = ["pt", "eta", "phi", "type_id", "mass", "charge", "dz", "isolation_rho_corr",
           "mass_valid", "charge_valid", "dz_valid", "isolation_rho_corr_valid"]


def dataset(events):
    widths = {kind: max(1, max((len(event.get(kind, [])) for event in events), default=0)) for kind in KINDS}
    values = np.zeros((len(events), sum(widths.values()), len(COLUMNS)), dtype=np.float32)
    start = 0
    for code, kind in enumerate(KINDS):
        for row, event in enumerate(events):
            for slot, particle in enumerate(event.get(kind, [])):
                fields = {"eta": 0, "phi": 0, "mass": 0, "charge": 1, "dz": 0, "isolation_rho_corr": 0,
                          "mass_valid": 1, "charge_valid": 1, "dz_valid": 1, "isolation_rho_corr_valid": 1,
                          **particle, "type_id": code}
                for name, value in fields.items():
                    values[row, start + slot, COLUMNS.index(name)] = value
        start += widths[kind]
    schema = [{"name": name, "kind": "continuous"} for name in COLUMNS]
    for spec in schema:
        if spec["name"] == "type_id":
            spec.update(kind="categorical", categories=dict(enumerate(KINDS)))
        if spec["name"] in ("mass", "charge", "dz", "isolation_rho_corr"):
            spec["validity_column"] = spec["name"] + "_valid"
    features = np.array([[event.get("met", 0), event.get("met_valid", 1)] for event in events], dtype=np.float64).reshape(-1, 2)
    return {"obj": torch.from_numpy(values), "obj_columns": list(COLUMNS), "obj_feature_schema": schema,
            "event_features": torch.from_numpy(features), "event_feature_columns": ["met", "met_valid"],
            "event_feature_schema": [{"name": "met", "validity_column": "met_valid"}, {"name": "met_valid"}]}
