# COLLIDE-2V trigger menus

A standalone NumPy/PyTorch package for evaluating trigger efficiencies and rates,
and calibrating transverse thresholds on converted `.pt` events.

`TriggerMenu` is a generic container. `TriggerMenu()` is empty; no reference menu
is inserted automatically. The Phase-2 proxy menu is an optional JSON database.
Phase-2, ERL, mixed, and calibrated menus all use the same JSON format and loader.
The package does not import Set_Based_Models or its converter at runtime.

## Install

From this directory, in an environment with PyTorch:

```bash
python -m pip install -e .
collide2v-trigger menu phase2
```

Every command also works as `python -m collide2v_trigger ...`.

## Load or build a menu

```python
from collide2v_trigger import PHASE2_MENU, TriggerMenu

phase2 = TriggerMenu.load(PHASE2_MENU)
learned = TriggerMenu.load("results/ggHgammagamma_menu.json")
calibrated = TriggerMenu.load("results/calibration/calibrated_thresholds.json")

custom = TriggerMenu(triggers=[trigger])
custom.add_trigger(another_trigger)
custom.save("results/custom_menu.json")
```

The bundled database is
[`src/collide2v_trigger/menus/phase2_proxy.json`](src/collide2v_trigger/menus/phase2_proxy.json).
It contains the selections, thresholds, scan fields, fixed cuts, reference rates,
and unavailable-object explanations. It lists 33 supported seeds and nine
unavailable StaEG/tau seeds. Copy/edit this JSON to make a different proxy menu;
trigger names do not select Python functions.

Each menu file contains an ordered `triggers` list with complete definitions,
original default thresholds, effective thresholds, and target rates, plus shared
object-selection settings. Saving after tuning changes the effective thresholds;
loading restores the settings automatically. `menu.triggers` is the registry of
trigger objects, and `menu.thresholds_for(name)` returns its current thresholds.
See [MENU_FORMAT.md](MENU_FORMAT.md) for the JSON fields and selection operations.

Older version-1 menu snapshots, parsed ERL triggers, threshold profiles, and
score caches remain readable. Saving a loaded menu writes the new version-2
format. Old menus preserve their previously implicit Phase-2 entries.

## Inputs

Use dictionaries saved by `raw_event_conversion`: `obj`, `obj_columns`,
`obj_feature_schema`, and event features with their names and schemas. Native
PUPPI MET is used if the derived `met` feature was omitted. Inputs may be PT
paths, quoted wildcard patterns, or `.chunks.json` manifests. Files listed more
than once are counted once. Manifest paths resolve relative to the manifest.

Columns and category codes are resolved by name/schema. Positive finite PT
identifies real objects; `obj_slot_mask` is honored when present. Measurement
validity flags are applied. Pair tensors load only for ERL rules requiring them.
Files are memory-mapped and evaluated in batches.

Relative-DZ selections need signed `dz` and `dz_valid`; reconvert older schema-10
files before evaluating the full Phase-2 preset. Object schema 11 and pair schema
12 contain these fields. Selections that do not use DZ can use older inputs.

## Evaluate efficiency and rate

```bash
collide2v-trigger evaluate ../data/SIGNAL/test_minbias.chunks.json \
  --menu phase2 --output results/phase2_rates.json

collide2v-trigger evaluate ../data/SIGNAL/test_ggHgammagamma.pt \
  --menu results/ggHgammagamma_menu.json --output results/learned_efficiency.json
```

Supply `--menu` with a JSON filename or `phase2`. `--threshold-config` remains an
alias for `--menu`. `--triggers NAME ...` restricts evaluation; otherwise all
supported entries in that menu are evaluated. `--bits bits.npz` saves Boolean
per-trigger and combined-OR arrays.

```python
from collide2v_trigger import evaluate_rates

result = evaluate_rates("signal.pt", learned)
efficiency = result["triggers"]["passes_any"]["efficiency"]
```

Reports include passing counts, efficiencies, per-trigger rates, and the measured
OR (`passes_any`). Rates are unweighted fractions times `--collision-rate-khz`,
default 40,000 kHz = 40 MHz. Signal acceptance is the signal-efficiency quantity;
the rate on a signal sample is a 40-MHz-equivalent normalization.

## Calibrate thresholds

```bash
collide2v-trigger calibrate ../data/SIGNAL/train_minbias.chunks.json \
  --menu phase2 --output-dir results/calibration

collide2v-trigger evaluate ../data/SIGNAL/test_minbias.chunks.json \
  --menu results/calibration/calibrated_thresholds.json \
  --output results/calibrated_test_rates.json
```

Each trigger uses its saved target rate unless `--target-rate-khz` overrides it.
Custom/ERL triggers need an assigned target before tuning. Use independent
minbias files for fitting and subsequent validation; the automatic evaluation
at the end checks the fitting sample.

```python
from collide2v_trigger import calibrate_menu

report = calibrate_menu("minbias.chunks.json", phase2,
                        output_dir="results/calibration")
restored = TriggerMenu.load("results/calibration/calibrated_thresholds.json")
```

Batch bisection finds each event's highest accepted threshold coordinate. By
default, adjustable thresholds scale proportionally, with the largest as the GeV
coordinate. The database's `scale_fields` excludes fixed quantities such as an
HT jet floor. `--reference-field` and `--scale-fields` choose another plan. For
example, 22/12-GeV photon cuts become 44/24 GeV at coordinate 44 GeV. ERL plans
with zero defaults use an additive offset. This changes selection thresholds,
not the saved particle momenta.

The default range is 0–1,000 GeV. `--score-tolerance` controls per-event precision;
`--tolerance` controls rate fitting. Both default to 0.01 GeV. Alternatively,
`--method monotonic --spacing 0.25` scans a grid. The fit selects the side with
estimated rate at most the target. Fixed triggers retain their original cuts,
need no target, and receive status `fixed`.

Outputs:

- `critical_thresholds.npz`: per-trigger event scores, scan plans, and the complete input menu.
- `calibrated_thresholds.json`: the complete calibrated menu, in the same format as the input database.
- `calibration.json`: targets, fit statuses, scan points, applied cuts, and directly reevaluated rates.

Unselected triggers remain in the calibrated menu with their existing settings.
Target overrides are saved. The original menu object remains unchanged.

Reuse scores to change targets without repeating event scans:

```bash
collide2v-trigger calibrate --scores results/calibration/critical_thresholds.npz \
  --target-rate-khz 10 --output-dir results/calibration_10khz
```

Saved input paths, event cap, scan bounds, and menu settings are reused; CLI menu
options are not reapplied. Final direct evaluation still reads the PT inputs,
which must not have changed.

Unreachable targets and unidentified zero-acceptance fits retain their original
thresholds. `--export-unreachable-boundary` exports the scan boundary instead,
while retaining the failure status. `calibrated` denotes an empirical rate match;
`discrete` denotes a lower achieved rate due to event steps, ties, or precision.
Per-event scores are upper estimates within the score tolerance; report
`applied_measurement` and `evaluation` give direct rates at the exported cuts.
Upper-bound passes indicate that a wider scan may be needed. Individual rate
targets do not constrain the combined menu OR.

## Convert ERL rules

```python
from collide2v_trigger import parse_erl_model

triggers = parse_erl_model("model.json", "reference.pt", allow_fixed=True)
menu = TriggerMenu(triggers=triggers)
menu.save("results/learned_menu.json")
```

This menu contains only the converted rules. `allow_fixed=True` retains faithfully
evaluable rules without tuning parameters, with a warning. Missing features and
unsupported semantics still fail; `skip_unsupported=True` can explicitly skip
those remaining failures. Both parser options default to false.

[ERL_PARSER.md](ERL_PARSER.md) covers individual-rule conversion, supported cuts,
exact multiplicity translation, and calibration. The step-06 quick runner writes
an ERL-only menu by default; its `BASE_MENU` option explicitly adds converted
rules to a saved menu, including the Phase-2 preset.

The separate quick runners [07–09](../quick_runners/README.md)
calibrate Phase-2, match the top ERL rule to its achieved background rates, and
compare the saved working points on independent minbias and signal samples.
Their Phase-2 calibration defaults to isolation disabled.

## Physics conventions of the Phase-2 preset

Muon/electron/photon proxies use the saved tight-object collections; LooseID and
LooseNN are not extra cuts. Electron/photon isolation defaults to
`isolation_rho_corr <= 0.15`; `--no-isolation` removes it. Absolute PV-DZ cuts are
disabled. Relative multi-lepton DZ requires `max(dz) - min(dz) < 10 mm`.
Eta, mass, charge, and angular requirements stay fixed during default calibration.

HT is recomputed from retained PUPPI jets with a strict PT floor and
`abs(eta) < 2.4`; native ScalarHT and total object PT are not substituted. The
quad-jet floor is 25 GeV, and other HT floors are 30 GeV by default.

The converted representation caps collections at 10 jets and four objects of
each lepton/photon type. Discarded objects cannot be recovered. Object geometry
columns lack original measurement-validity flags after zero filling. Dataset
version and source selection belong to input preparation. These are COLLIDE-2V
proxy rates, rather than an absolute CMS rate prediction.

## Code and checks

`menu.py` holds the container; `trigger.py` evaluates declarative selections;
`physics.py` handles object combinations; `erl.py` converts/evaluates ERL rules;
`data.py`, `evaluation.py`, and `calibration.py` provide the core workflows.
Concrete Phase-2 definitions live only in the JSON database.

```bash
python -m unittest discover -s tests -v
```
