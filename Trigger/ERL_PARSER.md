# One ERL rule to a tunable trigger

`parse_erl_rule` returns a callable `ERLTrigger`. The parser and the exported
trigger require only this package, NumPy, and PyTorch. Neither imports
Set_Based_Models or executes training code.

The input is a native `ExistentialRule`, a saved ERL model JSON plus a rule ID,
or a dictionary containing `rule`, `observables`, and `event_observables`.
Printed rule text is not an input format. The reference PT supplies feature
definitions, units, category meanings, and object/pair representation.

## Parse and export

From the `Trigger` directory after installation:

```bash
collide2v-trigger parse-erl \
  ../training_outputs/ggHgammagamma_scan_gini.validation.model.json \
  --rule-id 343 --data ../data/SIGNAL/test_ggHgammagamma.pt \
  --name LearnedPhotonActivity --target-rate-khz 20 \
  --output results/learned_photon_activity.json
```

This prints the action assigned to each cut, then saves the executable
declarative trigger. The output keeps the rule, relevant feature definitions,
guards, count requirement, parameter bindings, and source rule ID. It omits
training statistics and Jaccard matrices.

The Python interface accepts the same model path or an existing rule object:

```python
from collide2v_trigger import parse_erl_rule, ERLTrigger

trigger = parse_erl_rule(
    "model.json", "reference.pt", rule_id=343,
    name="LearnedPhotonActivity", target_rate_khz=20,
)
print(trigger.default_thresholds)
print(trigger.cut_actions)

flags = trigger("events.pt")
flags = trigger("events.pt", {"cut_0_sum_pt_photons": 30.0})
trigger.save("trigger.json")
restored = ERLTrigger.load("trigger.json")
```

Each call returns one Boolean per event. Parameter values can be scalars or
arrays with one value per event. Arrays support the calibrator's batch bisection.
Unspecified parameters retain their learned defaults.

## Convert a model to a list of triggers

`parse_erl_model` wraps the single-rule parser. It accepts a saved model path,
model dictionary, or native `ExistentialRuleList`, and returns `list[ERLTrigger]`:

```python
from collide2v_trigger import parse_erl_model, TriggerMenu, calibrate_menu

triggers = parse_erl_model("model.json", "reference.pt", target_rate_khz=20)
menu = TriggerMenu(triggers=triggers)
calibrate_menu("minbias.pt", menu, [trigger.name for trigger in triggers],
                output_dir="results/learned_menu")
```

By default it converts only the final retained rules, preserving their saved
order. An unpruned model requires an explicit selection. An empty retained
selection returns an empty list. Other selections are:

```python
triggers = parse_erl_model(model, data, rule_ids=[343, 548])
triggers = parse_erl_model(model, data, all_candidates=True)
```

Names default to `ERL_rule_<id>`; `name_prefix="Learned"` produces `Learned_<id>`.
One shared target rate can be supplied, or `target_rate_khz={343: 20, 548: 10}`
for per-rule rates. `tune={343: [0, 1]}` chooses adjustable cut indices for that
rule; other rules use automatic selection. Missing target rates remain unset
and must be supplied when calibrating.

Conversion is strict by default: if any requested rule fails, the wrapper raises
`RuleParseError` identifying its ID and returns no partial list. Select another
subset, specify that rule's adjustable cuts, or explicitly allow skipping:

```python
triggers = parse_erl_model(model, data, skip_unsupported=True)
```

Skip mode emits a `UserWarning` for each failed rule, including its ID and reason,
and preserves the order of the remaining triggers. If all selected rules fail,
it returns an empty list. Model-level errors, invalid rule selections, and
unexpected errors still stop conversion. The model is read once, and the
reference tensor batch is shared across conversions.

To keep faithfully evaluable rules that cannot be momentum-tuned, use:

```python
triggers = parse_erl_model(model, data, allow_fixed=True)
```

This first attempts normal tuning. If tuning is unavailable, it retries the
entire original rule with every cut fixed and emits a warning with its ID and
reason. Fixed triggers have `trigger.tunable == False`, empty
`default_thresholds`, and a saved `fixed_reason`. Their original cuts, guards,
and count requirements remain intact. Missing original features, unsupported
predicates, and invalid tuning indices still raise errors.

`allow_fixed=True` also works with `parse_erl_rule`; the single-rule CLI exposes
`--allow-fixed`. It takes precedence over `skip_unsupported=True` for rules that
can be retained exactly; skip mode then applies only to remaining parse errors.
Both options default to false.

Each returned trigger can be saved separately with `trigger.save(path)` or
registered directly with `TriggerMenu(triggers=triggers)`.

Save the entire menu, including every imported rule, in one JSON file:

```python
menu = TriggerMenu(triggers=triggers)
menu.save("menu.json")
restored = TriggerMenu.load("menu.json")
```

The file includes imported definitions, effective thresholds, target rates,
and isolation settings. Reloading requires the Trigger package and no original
ERL model or individual trigger files.
Fixed status and the empty parameter dictionary survive saving/loading.
`TriggerMenu()` is empty: this example saves only the converted ERL rules.
The Phase-2 preset and calibrated menus use the same version-2 menu format;
see [MENU_FORMAT.md](MENU_FORMAT.md). Individual ERL definitions use the common
`collide2v_trigger` format with `type="erl"`; older exports remain readable.

## Selection and tuning policy

Every cut is read structurally. Numeric comparisons retain `>`, `>=`, `<`, or
`<=`. Categories are mapped by their saved labels rather than assuming that
codes occupy the same columns or use the same numbering.

The default tuning policy exposes lower bounds on:

- Object/member `pt`, `pt_1`, and `pt_2`.
- Pair `min_pt`, `sum_pt`, and `pair_pt`.
- Event `met`, leading/subleading PT, and scalar PT sums in GeV.
- Full-collection object multiplicities, through the exact rank translation
  described below.

Categories, guards, upper bounds, mass windows, isolation, displacement,
geometry, composition, and ratios stay fixed by default. Existing PF counts
and sums can remain fixed cuts. A lower bound on a saved PF scalar PT sum can
be exposed directly; the PF selection/bin definition remains unchanged.

Use zero-based cut indices to choose the adjustable cuts explicitly:

```python
trigger = parse_erl_rule(rule, "reference.pt", tune=[0, 3])
```

Or add `--tune 0 3` to the CLI. Unselected cuts remain executable with their
original definitions. Explicit selection also permits other continuous lower
GeV bounds, such as a mass lower bound; fixed upper bounds remain unchanged.
An explicitly selected unsupported cut raises `RuleParseError`.
Repeated occurrences of the same feature have separate parameters, identified
by cut index. The parser does not retarget every occurrence of that feature.

## Multiplicity translation

For an integer rank `k` and the producer's strict PT comparison:

```text
N(objects with PT > T) >= k  <=>  kth-highest PT > T
```

For example, `n_photons_pt20 >= 2` becomes a condition on
`subleading_pt_photons`, with a tunable PT floor defaulting to 20 GeV.
The count requirement stays at two. Noninteger count cut thresholds are reduced
to the corresponding integer rank using the original comparison.

The current exact translator supports ranks one and two. It requires matching
full-collection leading/subleading metadata and float64 event features. At parse
time and on every subsequent batch, it checks that the original count predicate
agrees with the rank predicate at the learned floor. Inconsistency raises an
error instead of changing the rule's default decisions.

Higher-rank full-precision PT values are not saved. The capped float32 object
tensor does not guarantee an exact substitute, so automatic translation of a
higher-rank count raises an error. To tune another cut while keeping that
multiplicity fixed, select the other cut explicitly with `tune=[...]`.

Upper multiplicity conditions stay fixed. Retargeting both sides of an
exact-count/window condition to the same moving PT floor could be nonmonotonic;
this parser does not introduce that scan.

## Menu evaluation and calibration

```python
from collide2v_trigger import TriggerMenu, calibrate_menu

menu = TriggerMenu(triggers=[trigger])
# Alternatively: menu.add_trigger(trigger)
flags = menu.evaluate(trigger.name, "events.pt")
result = calibrate_menu(
    "minbias.chunks.json", menu, [trigger.name],
    target_rate_khz=20, output_dir="results/learned_calibration",
)
```

The CLI loads the exported definition with `--erl-trigger`:

```bash
collide2v-trigger calibrate ../data/SIGNAL/test_minbias.chunks.json \
  --erl-trigger results/learned_photon_activity.json \
  --triggers LearnedPhotonActivity \
  --output-dir results/learned_calibration
```

If the exported trigger has no target rate, supply `--target-rate-khz`.
With only `--erl-trigger`, the menu contains only those explicitly supplied rules.
To add rules to the reference menu, also pass `--menu phase2`.
Menu isolation settings add no requirements to an imported rule.

Positive defaults use the existing proportional GeV scan. If any selected
parameter has a zero default, the plan uses a common additive GeV offset:
`threshold_i = learned_threshold_i + coordinate`. Coordinate zero then reproduces
the learned rule. The default nonnegative offset range only tightens it; choose
a subset of parameters if a different one-dimensional prescription is needed.
The scan metadata records `mode="scale"` or `mode="offset"`.

Fixed triggers are evaluated at their original cuts and excluded from threshold
scans. Calibration reports mark them `status="fixed"` and record their measured
rates without claiming a target-rate match. They need no target rate. Profiles
and score caches retain them, including when all requested triggers are fixed.
`calibration_complete` refers to the adjustable triggers; `fixed_triggers` lists
the triggers that were intentionally left unchanged. `menu.is_tunable(name)` and
the `tunable` field in `menu_summary()` expose the distinction.

Threshold profiles and saved scores embed the trigger definition. Reloading
either restores it without the ERL model or a separate trigger JSON:

```bash
collide2v-trigger evaluate ../data/SIGNAL/test_ggHgammagamma.pt \
  --menu results/learned_calibration/calibrated_thresholds.json \
  --triggers LearnedPhotonActivity --output results/learned_ggh_rates.json
```

Pair rules use `pair_obj` and `pair_slot_mask`; the runners load these on demand.
Object cuts are ANDed on the same object, pair cuts on the same pair, and only
then is the saved minimum count applied. Pair counts mean matching pair slots,
including overlapping pairs, exactly as in ERL. Event cuts stay at event scope.
The model's explicit validity-guard policy is retained.

## Errors and data contract

`RuleParseError` reports unsupported predicates, no tunable momentum condition,
missing or ambiguous object/pair representation, missing features/guards,
changed feature definitions, inconsistent ranked/count values, invalid category
meanings, and unsupported requested translations. Parsing fails before the CLI
writes the trigger file. File and JSON-reading errors propagate normally.

Changing a PF pT/D0/significance bin internally needs the underlying PF candidates,
which the current PT format does not save. Such a request is rejected; a parser
cannot reconstruct those candidates from aggregates. Boolean decisions, ratio
cuts, and upper bounds are also rejected as explicitly selected momentum knobs.

Use reference data made with the training conversion settings. ERL model schemas
record names, kinds, category labels, and guards, but not complete producer
definitions. The parser binds those definitions from the supplied reference PT
and checks them on later inputs; it cannot infer an unknown training conversion.
The current runner convention is finite positive object PT for presence, plus
an explicit object mask when supplied, or the saved pair-presence mask.
