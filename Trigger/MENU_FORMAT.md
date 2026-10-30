# Menu database format

All new menu databases use `format="collide2v_trigger_menu"`, `schema_version=2`.
Phase-2 presets, ERL-only menus, combined menus, and calibrated menus use the
same structure and `TriggerMenu.load(path)` entry point. Loading does not insert
additional triggers.

A complete one-trigger menu:

```json
{
  "format": "collide2v_trigger_menu",
  "schema_version": 2,
  "object_selection": {
    "isolation_enabled": true,
    "electron_iso_max": 0.15,
    "photon_iso_max": 0.15
  },
  "triggers": [
    {
      "format": "collide2v_trigger",
      "schema_version": 1,
      "type": "selection",
      "name": "MyJet",
      "default_thresholds": {"jet_pt": 230.0},
      "thresholds": {"jet_pt": 230.0},
      "target_rate_khz": 37.0,
      "scale_fields": ["jet_pt"],
      "scan_mode": "scale",
      "selection": {
        "op": "group",
        "kind": "jets",
        "thresholds": ["$jet_pt"],
        "eta_max": 2.4
      }
    }
  ]
}
```

`default_thresholds` preserves the original working point. `thresholds` contains
the current working point and changes after calibration. `target_rate_khz` may
be null until a target is assigned. Trigger names are arbitrary unique labels.
The trigger record without `thresholds` is also a standalone trigger definition,
readable by `Trigger.from_dict(...)` or `Trigger.load(...)`.

A fixed trigger has empty parameter dictionaries; literal cuts stay in its
selection or ERL definition. An unavailable entry has `unavailable_reason` and
is excluded from default evaluation. Selecting it explicitly raises an error.

## Declarative selections

`type="selection"` uses the `selection` expression. A string beginning with `$`
references a named threshold. `@electron_iso_max` and `@photon_iso_max` reference
shared isolation limits, or resolve to null when isolation is disabled. Numeric
literals are fixed cuts. Expressions can be combined recursively:

```json
{
  "op": "all",
  "conditions": [
    {"op": "group", "kind": "muons", "thresholds": ["$muon_pt"]},
    {"op": "event", "field": "met", "comparison": ">=", "threshold": "$met"}
  ]
}
```

| Operation | Selection |
| --- | --- |
| `all`, `any` | AND/OR of `conditions` |
| `event` | Named event field, `comparison` (`>`, `>=`, `<`, `<=`), and `threshold` |
| `ht` | Jet scalar PT sum with `pt_min`, `eta_max`, and summed `threshold` |
| `group` | A same-collection object combination satisfying a ranked `thresholds` list |
| `multi_group` | One combination from each entry in `groups`, optionally with common DZ compatibility |
| `matched_groups` | `first` and `second` object groups with leading-object separation/matching requirements |

Each group has a `kind` (`jets`, `muons`, `electrons`, or `photons`), thresholds,
and optional `eta_max`, `isolation`, and strict `pt_floor`. A `group` can also
specify `dz`/`dz_max`, `delta_eta_max`, `dr_max`, `mass_min`, `mass_window`,
`opposite_charge`, and `pair_mass_hypothesis`. A mass window accepts any eligible
pair within the group; opposite-charge checks apply to that same pair. A fixed
mass hypothesis is used for both members; otherwise the saved masses are used.

`multi_group` can apply `dz=true` and `dz_max` to all selected members together.
Objects cannot be reused between groups of the same collection.

`matched_groups` compares the leading-PT objects of its two groups using strict
`dr_min`/`dr_max` bounds. `second_delta_eta_pair=[i,j]` and
`second_delta_eta_max` can constrain two PT-ranked members of its second group.
Indices start at zero. These operations describe the Phase-2 preset without
Python functions selected by trigger name.

## Converted ERL rules

`type="erl"` uses the same common trigger fields, with `cuts`, bound `features`,
`slot_source`, `minimum_count`, and `source_rule_id` in place of a selection
expression. The parser provides these fields; they preserve the rule's exact
feature schemas, guards, categories, and object-count semantics. See
[ERL_PARSER.md](ERL_PARSER.md). Both trigger types can occupy the same menu list.

## Calibration

`scale_fields` lists adjustable coordinates; omitted fields remain unchanged.
When absent, all named thresholds participate. `scan_mode="scale"` scales cuts
proportionally. `scan_mode="offset"` adds the coordinate to each selected cut.
Without an explicit mode, a selected zero default chooses offset; otherwise
scale is used. The Phase-2 preset explicitly uses scale, including seeds with
one zero-valued leg. A scan can override its reference and selected fields.

`calibrated_thresholds.json` is a complete menu database. Calibration preserves
all definitions, unselected triggers, isolation settings, original defaults, and
updated target rates. It adds `calibration_complete`, `fixed_triggers`, and
`collision_rate_khz` annotations. `calibration_complete` refers to adjustable
triggers; fixed entries are evaluated at their original cuts. Detailed fit
statuses remain in `calibration.json`.
