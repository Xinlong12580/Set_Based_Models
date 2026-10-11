"""Raw Parquet -> event and object tensors, without regionization/event filtering."""

import json
import os
from pathlib import Path
import tempfile
from urllib.parse import unquote, urlsplit
import warnings

import awkward as ak
import numpy as np
import torch

from .features import LEPTON_D0_COLUMNS, MET_COLUMNS, PF_IMPACT_COLUMNS, PT_COLUMNS, build_event_tensor, default_features
from .objects import OBJECT_COLUMNS, OBJECT_SLOTS, RAW_OBJECT_COLUMNS, build_object_tensor, object_feature_schema
from .pairs import LEPTON_MASSES, PAIR_COLUMNS, PAIR_OBJECT_INDICES, build_pair_tensor, pair_feature_schema


# Producer values, not sums recomputed from PF candidates. Rho indices retain
# producer order and are not the old preprocessing's PF region IDs.
NATIVE_FIELDS = {
    "L1T_MET_MET": (("met",), "GeV"),
    "L1T_MET_Phi": (("met_phi",), "rad"),
    "L1T_PUPPIMET_MET": (("puppi_met",), "GeV"),
    "L1T_PUPPIMET_Phi": (("puppi_met_phi",), "rad"),
    "L1T_ScalarHT_HT": (("ht",), "GeV"),
    "L1T_Rho_Rho": (tuple(f"rho_{i}" for i in range(5)), "GeV per producer area"),
}


def build_native_info(records):
    """Preserve native values and masks; unavailable measurements are not cuts."""
    values, masks, names, schema = [], [], [], []
    for column, (columns, units) in NATIVE_FIELDS.items():
        width = len(columns)
        lengths = ak.to_numpy(ak.num(records[column], axis=1))
        if np.any((lengths != 0) & (lengths != width)):
            raise ValueError(f"{column} must contain {width} values or be empty per event.")
        padded = ak.pad_none(records[column], width, axis=1, clip=True)
        array = np.asarray(ak.to_numpy(ak.fill_none(padded, np.nan)), dtype=np.float64)
        valid = np.isfinite(array)
        if column.endswith("_MET"):
            valid &= array >= 0
        elif column.endswith("_Phi"):
            magnitude = ak.pad_none(records[column[:-3] + "MET"], 1, axis=1, clip=True)
            magnitude = np.asarray(ak.to_numpy(ak.fill_none(magnitude, np.nan)), dtype=np.float64)
            valid &= np.isfinite(magnitude) & (magnitude > 0)
        values.append(np.where(valid, array, 0.0))
        masks.append(valid)
        names.extend(columns)
        schema.extend({"name": name, "kind": "continuous", "units": units,
                       "raw_column": column, "index": index}
                      for index, name in enumerate(columns))
    return {
        "native_event_info": torch.from_numpy(np.concatenate(values, axis=1)),
        "native_event_info_validity": torch.from_numpy(np.concatenate(masks, axis=1)),
        "native_event_info_columns": names, "native_event_info_schema": schema,
    }


def convert_raw_parquet_to_pt(
    sources, output, *, label, max_events=None, thresholds=None,
    extra_features=(), include_native_event_info=True, include_met_features=True,
    include_pair_tensor=False, overwrite=True,
):
    """Convert raw file(s) in supplied order, preserving zero-object events.

    Inputs are individual local filenames or remote URLs, not regionized
    Parquet. Only columns required by the feature definitions are loaded.
    max_events is a total row cap; selected columns of each opened file are
    read before slicing. No process label is inferred from the filenames.
    Event MET/ratio/angular features and their validity flags are independent of the
    separate native-event-info tensor; disable them with include_met_features=False.
    Default angular features require aligned PT/Eta/Phi object collections;
    leading/subleading ranks use finite positive PT, not eta.
    All default jet and MET inputs use L1T_JetPuppiAK4 and L1T_PUPPIMET.
    The separate native-event-info tensor still preserves both MET collections.
    obj extends the old float32 PT/Eta/Phi/type_id layout, with 10 PUPPI jets,
    4 tight muons, 4 electrons and 4 tight photons (22 slots, no MET slot).
    Each collection is PT-ranked and zero-padded; event features are not capped.
    Eight extra object features have matching schema-linked validity flags.
    include_pair_tensor adds pair_obj/pair_slot_mask and their schema, without
    replacing obj or event_features. Pair generation uses the same object caps.
    Charged PF counts/sums use unweighted L1T_PFPart_PT and finite nonzero
    L1T_PFPart_Charge, without a vertex selection or additional candidate cap.
    Impact-parameter bins use producer D0/ErrorD0 in mm; invalid measurements
    are excluded, counted separately and reported with a warning.
    PT-ranked lepton D0 magnitudes (mm) have explicit validity flags, with
    no re-ranking when the leading/subleading lepton has invalid D0.
    Existing outputs are replaced with a warning; overwrite=False protects them.
    """
    files = [str(sources)] if isinstance(sources, (str, Path)) else [str(path) for path in sources]
    files = [path if "://" in path else str(Path(path).expanduser().resolve()) for path in files]
    if not files or len(set(files)) != len(files):
        raise ValueError("Provide a nonempty list of distinct input filenames.")
    if any(not path.endswith(".parquet") or any(char in path for char in "*?[") for path in files):
        raise ValueError("Inputs must be individual .parquet filenames; expand local globs in the shell.")
    if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < 2**63:
        raise ValueError("label must be a nonnegative int64-compatible integer.")
    if max_events is not None and (isinstance(max_events, bool) or not isinstance(max_events, int)
                                   or max_events < 1):
        raise ValueError("max_events must be a positive integer or None.")
    if not isinstance(include_pair_tensor, bool):
        raise TypeError("include_pair_tensor must be a Boolean.")
    output = Path(output).expanduser().absolute()
    source_list = output.with_suffix(".sources.json")
    if output == source_list:
        raise ValueError("Choose an output filename with a .pt suffix.")
    local_inputs = [Path(unquote(urlsplit(path).path)) if path.startswith("file://") else Path(path)
                    for path in files if "://" not in path or path.startswith("file://")]
    for target in (output, source_list):
        if target.is_symlink():
            raise ValueError(f"Output must not be a symbolic link: {target}")
        if any(target.resolve() == source.resolve() or
               (target.exists() and source.exists() and target.samefile(source))
               for source in local_inputs):
            raise ValueError(f"Output must not overwrite an input file: {target}")
        if target.exists():
            if not target.is_file():
                raise ValueError(f"Output must be a regular file: {target}")
            if not overwrite:
                raise FileExistsError(f"Refusing to overwrite {target}; choose a new output.")
            warnings.warn(f"Overwriting existing file: {target}", UserWarning, stacklevel=2)

    features = (*default_features(thresholds, include_met=include_met_features), *extra_features)
    if any(isinstance(feature.columns, (str, bytes)) for feature in features):
        raise TypeError("Feature columns must be a sequence of column names, not a string.")
    schema = [feature.schema() for feature in features]
    json.dumps(schema, allow_nan=False)  # Save only plain metadata, never callables.
    columns = list(dict.fromkeys(column for feature in features for column in feature.columns))
    columns.extend(RAW_OBJECT_COLUMNS)
    if include_native_event_info:
        columns.extend(NATIVE_FIELDS)
    columns = list(dict.fromkeys(columns))
    blocks, remaining = [], max_events
    for file_index, filename in enumerate(files):
        if remaining == 0:
            break
        records = ak.from_parquet(filename, columns=columns)
        missing = set(columns) - set(records.fields)
        if missing:
            raise ValueError(f"Missing raw columns in {filename}: {sorted(missing)}")
        if remaining is not None:
            records = records[:remaining]
        count = len(records)
        if not count:
            continue
        block = {
            "event_features": build_event_tensor(records, features),
            "obj": build_object_tensor(records),
            "raw_source_file": torch.full((count,), file_index, dtype=torch.int64),
            "raw_source_row": torch.arange(count, dtype=torch.int64),
        }
        if include_native_event_info:
            block.update(build_native_info(records))
        if include_pair_tensor:
            block["pair_obj"], block["pair_slot_mask"] = build_pair_tensor(records)
        blocks.append(block)
        if remaining is not None:
            remaining -= count
    if not blocks:
        raise ValueError("The input contains no events.")

    dataset = dict(blocks[0])
    if len(blocks) > 1:
        for key, value in dataset.items():
            if isinstance(value, torch.Tensor):
                dataset[key] = torch.cat([block[key] for block in blocks], dim=0)
    count = len(dataset["event_features"])
    dataset.update({
        "event_feature_columns": [feature.name for feature in features], "event_feature_schema": schema,
        "label": torch.full((count,), label, dtype=torch.int64),
        # PF remains a compatibility placeholder; obj contains real objects.
        # Event-only runners still build their own masked dummy object input.
        "pf": torch.zeros((count, 1, 1), dtype=torch.float32),
        "pf_columns": ["unused_dummy"], "obj_columns": list(OBJECT_COLUMNS),
        "obj_feature_schema": object_feature_schema(),
        "conversion_metadata": {"format": "raw_event_features", "schema_version": 7,
                                "event_filtering": False, "object_truncation": True,
                                "event_features_before_object_truncation": True,
                                "dummy_pf_and_obj": False, "dummy_pf": True,
                                "object_tensor": {
                                    "collections": {PT_COLUMNS[kind].removesuffix("_PT"): width
                                                    for kind, width in OBJECT_SLOTS.items()},
                                    "type_ids": {kind: index for index, kind in enumerate(OBJECT_SLOTS)},
                                    "sort_by_pt": True, "stable_sort": True,
                                    "selection": "finite PT > 0", "met_included": False,
                                    "padding": 0, "invalid_angle_fill": 0,
                                    "invalid_feature_fill": 0,
                                    "feature_validity_flags": True,
                                    "validity_flags_are_automatic_guards": False,
                                    "object_truncation": True,
                                    "event_features_before_object_truncation": True,
                                },
                                "native_event_info_included": include_native_event_info,
                                "met_features_included": include_met_features,
                                "object_pt_columns": dict(PT_COLUMNS),
                                "pf_impact_parameter_columns": list(PF_IMPACT_COLUMNS),
                                "pf_impact_parameter_units": "mm",
                                "lepton_d0_columns": dict(LEPTON_D0_COLUMNS),
                                "lepton_d0_units": "mm",
                                "met_columns": list(MET_COLUMNS) if include_met_features else []},
    })
    if include_pair_tensor:
        dataset.update({
            "pair_obj_columns": list(PAIR_COLUMNS), "pair_obj_feature_schema": pair_feature_schema(),
            "pair_object_indices": [list(pair) for pair in PAIR_OBJECT_INDICES],
        })
        dataset["conversion_metadata"].update({
            "schema_version": 8,
            "pair_tensor": {
                "slots": len(PAIR_OBJECT_INDICES), "columns": len(PAIR_COLUMNS), "dtype": "float32",
                "source": "raw values of the same PT-ranked, capped objects as obj",
                "ordering": "type_id ascending; PT descending within each type; producer-order ties",
                "self_pairs": False, "reversed_duplicates": False, "padding": 0,
                "presence_mask": "pair_slot_mask", "validity": "explicit schema-linked flags",
                "mass_hypotheses_GeV": {**LEPTON_MASSES, "photons": 0.0, "jets": "valid stored Mass"},
            },
        })
    names = dataset["event_feature_columns"]
    for observable in ("abs_dxy_mm", "abs_dxysig"):
        invalid = dataset["event_features"][:, names.index(f"n_charged_pf_pt_gt2_{observable}_invalid")].sum().item()
        if invalid:
            total = dataset["event_features"][:, names.index("n_charged_pf_pt_gt2")].sum().item()
            warnings.warn(f"{invalid:g} of {total:g} charged PF candidates with PT > 2 GeV have invalid "
                          f"{observable} measurements and are excluded from its bins and cumulative cuts. "
                          "Zero bin values do not imply absence of displaced activity. "
                          "Check D0/ErrorD0 availability and the producer's tracking model.",
                          UserWarning, stacklevel=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    # File paths stay outside .pt metadata so different samples can be mixed.
    # raw_source_file indexes this list; raw_source_row indexes the original file.
    # Finish serialization before replacing existing outputs, so a save failure
    # does not truncate an earlier dataset. Stage on the same filesystem.
    with tempfile.TemporaryDirectory(prefix=".raw_event_conversion_", dir=output.parent) as staging:
        staged_output = Path(staging) / "events.pt"
        staged_sources = Path(staging) / "sources.json"
        torch.save(dataset, staged_output)
        with staged_sources.open("w", encoding="utf-8") as handle:
            json.dump(files, handle, indent=2)
            handle.write("\n")
        for staged, target in ((staged_output, output), (staged_sources, source_list)):
            if overwrite:
                os.replace(staged, target)
            else:
                os.link(staged, target)  # Exclusive publication: never overwrite.
    return output
