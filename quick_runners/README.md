# Signal quick runners

Run these Bash scripts from any directory, after activating an environment with
the repository's conversion/ERL dependencies. For EOS, install `fsspec-xrootd`
and working XRootD bindings, with any required credentials. `python` must be
Python 3.10+; set `PYTHON_BIN=/path/to/python` to select another interpreter.
No installation or environment activation is performed by the scripts.

## Run all steps

From the repository root:

```bash
./quick_runners/pro_SIG.sh
```

Or run the steps individually:

```bash
./quick_runners/01_select_SIG_sources.sh
./quick_runners/02_convert_minbias_SIG.sh
./quick_runners/03_convert_SIG.sh
./quick_runners/04_make_SIG_mixture.sh
./quick_runners/05_train_SIG_ERL.sh
```

The first step lists minbias and `WJetsToLNu_13TeV-madgraphMLM-pythia8` under
`root://eosproject.cern.ch//eos/project/f/foundational-model-dataset/samples/production_final/`.
It excludes hidden/non-Parquet entries and selects all lexicographically ordered
files with footer `dataset_version=collide2v_v1.0` by default (`MAX_SOURCE_FILES=0`).
Set a positive limit at the top of step 1 to reduce the number of footer reads.
Only footer metadata is read at this step; full selected columns are read by
conversion. URLs are pinned to text manifests so subsequent steps use the same
ordered inputs without another directory listing. This is not random sampling
of EOS files or events. Metadata read failures stop the workflow, rather than
silently replacing an unreadable file with another one.

## Defaults and outputs

Each executable contains its own configuration at the top; there is **no central
config file**. Environment variables can override those defaults. If editing
paths or conversion settings directly, keep them consistent across the relevant
steps. `SIG_PROCESS` defaults to `WJetsToLNu_13TeV-madgraphMLM-pythia8`.
Outputs below are relative to the repository:

| Output | Contents |
| --- | --- |
| `data/<SIG_PROCESS>/sources/minbias.txt`, `SIG.txt` | Ordered raw EOS URLs |
| `data/<SIG_PROCESS>/minbias.pt`, `SIG.pt` | Up to 100,000 events each, with event/object/pair tensors |
| `data/<SIG_PROCESS>/minbias.sources.json`, `SIG.sources.json` | Conversion source lists; row provenance is inside each PT |
| `data/<SIG_PROCESS>/mixture/reference.pt` | 5,000 minbias events |
| `data/<SIG_PROCESS>/mixture/anomalous.pt` | Shuffled 4,500 minbias + 500 signal events (90% background) |
| `training_outputs/ERL_<SIG_PROCESS>_pairs.model.json` | Reloadable core ERL model |
| `training_outputs/ERL_<SIG_PROCESS>_pairs.report.json` | Structured training report |
| `training_outputs/ERL_<SIG_PROCESS>_pairs.training.log` | Training stdout and stderr, including progress |

Sampling is without replacement, with seed 42 and disjoint reference/mixture
minbias rows. Independent physics events still depend on the source production.
Conversion preserves zero-object events and does not regionize or filter events.
The converter caps the total per process, not each file. Short files can yield
fewer events: increase `MAX_SOURCE_FILES` or lower the requested mixture counts.
The mixer requires at least `REFERENCE_COUNT + BACKGROUND_COUNT` minbias events
and `SIGNAL_COUNT` signal events. Set `MAX_SOURCE_FILES=0` to select all version-matching
files (this can require many footer reads).

**Rerunning replaces these outputs.** Conversion and mixing warn before replacing
files; the training log also warns and the trainer uses `--overwrite-output`. Other
datasets are not reused or overwritten by default. The source-selection step
refreshes the manifests; rerun from step 2 to reuse existing manifests instead.

## ERL setup

Step 5 calls `casual_tests/train_ERL.py`, which saves the core model after final
pruning and writes a separate training report. Its search settings remain:

- `pair_obj` only; event features and single-particle slots are not ERL inputs.
- Compulsory ordered priority: `type_id_1`, `type_id_2`, `pair_mass:>`,
  `pair_mass:<=`; complete priority required for final retention.
- Learned particle categories and mass bounds; no prescribed lepton types or
  mass window. This is a **configured pair-window scan**, not unconstrained
  feature discovery. Four priority cuts consume the current four-cut depth.
- Score `Gini gain - 20 * reference acceptance`.
- Logical improvement threshold `-0.01`; beam/final similarity enabled at 0.9.
- Nominal **per-rule** ceiling 40 kHz at 40 MHz (reference acceptance <= 0.001).
- Existing runner search budgets: depth 4, beam width 40, candidate cap 300,
  up to 32 thresholds. The wrapper does not modify the model or runner.

The ERL settings are preserved from the previous workflow. In particular, they
do **not** target the characteristic single-lepton + MET signature of W→ℓν:
MET is an event feature, and step 5 currently disables event inputs. Change the
ERL command in step 5 for an event-feature or combined-input study; changing
`SIG_PROCESS` alone changes the dataset, not the search strategy.

Acceptances/rates describe fitting samples, not independent validation. Zero
observed reference passes do not establish zero rate; the per-rule ceiling does
not constrain the combined OR rate. Full pair conversion/search can take
substantial memory and time.

## Customization

Environment overrides must be consistent across steps. For example:

```bash
MAX_SOURCE_FILES=12 MAX_EVENTS=100000 PYTHON_BIN=python3 ./quick_runners/pro_SIG.sh

# A different signal gets separate data/report/log paths automatically.
SIG_PROCESS=ggHgammagamma ./quick_runners/pro_SIG.sh

REFERENCE_PENALTY=10 ./quick_runners/05_train_SIG_ERL.sh

# Keep both type cuts compulsory, but let the remaining cuts choose features.
# The final completeness check then requires only the two type cuts.
./quick_runners/05_train_SIG_ERL.sh \
  --object-cut-priority type_id_1 type_id_2
```

Steps 2–5 forward extra arguments to their Python command. Changes to conversion
features must be applied to **both** process conversions so their schemas match.
`pro_SIG.sh` calls the five scripts, without sourcing a config or changing their
settings. It takes no CLI arguments; exported settings are inherited by all steps.
Set `SIG_WORK_DIR` for independent input datasets and `SIG_TRAINING_PREFIX` for
independent model/report/log sets. `SIG_MODEL`, `SIG_REPORT`, and `SIG_LOG` can
override individual files. Relative paths are best avoided for these overrides.
Existing datasets and historical reports/logs are not renamed by this change.

## Convert step 05 output to a trigger menu

```bash
./quick_runners/06_convert_ERL_to_triggers.sh
```

This separate step follows the current step-05 scan output. It reads
`training_outputs/ggHgammagamma_scan.validation.model.json`, uses
`data/SIGNAL_train_ggHgammagamma_mixture/reference_validation.pt` for feature
definitions, and converts the final retained rules. It saves one complete menu
to `Trigger/results/ggHgammagamma_menu.json`, containing only the imported ERL
triggers. It uses the same version-2 menu format as the Phase-2 JSON preset and
calibrated menus. It performs conversion only; training and calibration
are separate steps. The local `Trigger/src` is used without requiring installation.
`pro_SIG.sh` continues to run steps 01–05; run step 06 afterward.

By default, rules without supported tuning parameters are retained exactly as
fixed triggers (`ALLOW_FIXED=1`). A rule that cannot be faithfully evaluated
stops the script with its ID, and no new menu is written. Explicitly select a
subset with space-separated rule IDs:

```bash
RULE_IDS="269 261" ./quick_runners/06_convert_ERL_to_triggers.sh
```

To skip unsupported rules with a warning for each ID and reason:

```bash
SKIP_UNSUPPORTED=1 ./quick_runners/06_convert_ERL_to_triggers.sh
```

Skipping remains disabled by default (`SKIP_UNSUPPORTED=0`). Fixed fallback is
attempted first when `ALLOW_FIXED=1`; this option skips remaining parse failures.

To explicitly retain untunable rules as fixed triggers (the runner default):

```bash
ALLOW_FIXED=1 ./quick_runners/06_convert_ERL_to_triggers.sh
```

Tuning-only failures produce warnings and are kept with their original cuts and
empty parameter dictionaries. Missing features and genuinely unsupported rule
semantics still fail. If both options are enabled, fixed fallback is attempted
first and only remaining parse failures are skipped. Calibration leaves fixed
triggers unchanged and reports their measured rates with status `fixed`.
Set `ALLOW_FIXED=0` for tunable-only conversion; unsupported tuning then raises
an error unless skip mode is enabled.

To explicitly include a base menu, set `BASE_MENU` to its JSON path:

```bash
BASE_MENU=Trigger/src/collide2v_trigger/menus/phase2_proxy.json \
  MENU_OUTPUT=Trigger/results/ggHgammagamma_with_phase2.json \
  ./quick_runners/06_convert_ERL_to_triggers.sh
```

`TriggerMenu()` itself is empty. Existing version-1 combined snapshots remain
loadable; saving them writes the common version-2 format.

IDs refer to the particular saved model and may change after rerunning step 05.
`TARGET_RATE_KHZ=20` sets a shared target for later calibration; by default the
learned triggers' targets remain unset. Configuration at the top also supports
`PYTHON_BIN`, `SIG`, `SIG_WORK_DIR`, `ERL_MODEL`, `REFERENCE_PT`, `MENU_OUTPUT`, and `BASE_MENU`.
The default `SIG=ggHgammagamma` matches the currently active step-05 block.
Relative overrides are resolved from the repository root.

Reload the single output with:

```python
from collide2v_trigger import TriggerMenu

menu = TriggerMenu.load("Trigger/results/ggHgammagamma_menu.json")
```

## Calibrate and compare with Phase-2: steps 07–09

Run from an environment with PyTorch, using the same `PYTHON_BIN` for each step:

```bash
./quick_runners/07_calibrate_phase2.sh
./quick_runners/08_calibrate_erl_to_phase2.sh
./quick_runners/09_compare_erl_phase2.sh
```

These runners are separate from `pro_SIG.sh`. Steps 05 and 06 must already have
produced the ERL model and converted menu. Step 07 is independent of ERL training.
Each script resolves default paths from the repository and works from any directory.

| Runner | Operation | Default outputs |
| --- | --- | --- |
| `07_calibrate_phase2.sh` | Calibrate all supported Phase-2 seeds on training minbias | `Trigger/results/phase2_calibration/{calibrated_thresholds.json,calibration.json,critical_thresholds.npz}` |
| `08_calibrate_erl_to_phase2.sh` | Select the top retained ERL rule and fit one working point per usable Phase-2 rate | `Trigger/results/ggHgammagamma_phase2_matched/{matched_menu.json,matching.json,critical_thresholds.npz}` |
| `09_compare_erl_phase2.sh` | Evaluate the frozen menus on test minbias and test signals | `Trigger/results/ggHgammagamma_phase2_comparison/{comparison.json,comparison.csv}` |

Step 07 defaults to **isolation disabled** (`NO_ISOLATION=1`). It reads
`data/SIGNAL/train_minbias.chunks.json` and the packaged Phase-2 database, using
the database's per-trigger target rates. `NO_ISOLATION=0` keeps the database's
isolation setting. It forwards extra CLI arguments, such as `--upper 2000`.
Rerunning replaces the calibration files; rerun step 08 before comparing with a
changed Phase-2 menu.

Step 08 selects the highest validation score among the model's retained rules
(ties use the smaller rule ID), then finds that rule in step 06's menu. A fixed
top rule raises an error rather than silently choosing another rule. It uses the
same minbias inputs, event count, and collision-rate normalization as step 07;
it does not read test samples. The original ERL predicates remain unchanged.

Each ERL working point targets the Phase-2 trigger's **directly measured achieved
calibration rate**, rather than its nominal target or a numerical GeV threshold.
For example, a discrete 36.8-kHz Phase-2 fit with a nominal target of 37 kHz is
matched to 36.8 kHz. The ERL event scores are computed once and reused for every
target. Each working point gets a unique trigger name and effective thresholds
in one loadable menu file. Source menus are not modified.

`matching.json` records every attempted pairing, source rule/score, menu
snapshots, target and measured rates, fit status, and `rate_matched`. Phase-2
unreachable/unidentified fits and zero achieved rates are flagged and excluded
from fitting. An unreachable ERL fit is also flagged and is not exported as a
matched trigger. Discrete ERL fits remain available for comparison with their
actual rate differences shown; `rate_matched` is true only for equal observed
calibration rates.

Step 09 loads both menus without fitting. It checks that they match the snapshots
from step 08 and that test minbias files differ from calibration inputs. Defaults
are `data/SIGNAL/test_minbias.chunks.json` and all four signal test manifests:
WJetsToLNu, ggHgammagamma, ZZ_leptonic, and ggHbb. Signal outputs are efficiencies,
passing counts, and ERL/Phase-2 efficiency ratios; physical signal production
rates are not inferred. Independent test-minbias rates and counts are included.
Undefined efficiency ratios are null in JSON and blank in CSV. Excluded pairings
remain in the JSON report.

Configuration is at the top of each script:

- Shared: `PYTHON_BIN`, `BATCH_SIZE`, `PHASE2_CALIBRATION`.
- Step 07: `SIG_WORK_DIR`, `MINBIAS_TRAIN`, `PHASE2_DATABASE`, `NO_ISOLATION`, `COLLISION_RATE_KHZ`, `MAX_EVENTS`.
- Step 08: `SIG`, `ERL_MODEL`, `ERL_MENU`, `ERL_MATCH_DIR`, `LOWER`, `UPPER`, `SCORE_TOLERANCE`, `TOLERANCE`.
- Step 09: `SIG`, `SIG_WORK_DIR`, `PHASE2_MENU`, `ERL_MATCH_DIR`, `COMPARISON_DIR`, `MINBIAS_TEST`, `SIGNALS`, `MAX_EVENTS`.

`SIG=ggHgammagamma` selects the current step-05/06 outputs. `SIGNALS` is a
space-separated list, for example `SIGNALS="ggHgammagamma ZZ_leptonic"`.
`SIG_WORK_DIR` defaults to `data/SIGNAL`. Step 08 inherits step 07's event count;
`MAX_EVENTS` in step 09 caps comparison samples only. Use separate output
directories for smaller exploratory runs. Relative overrides resolve from the
repository root.

```bash
export PYTHON_BIN=/home/xinlong/miniconda3/envs/hep/bin/python
SIGNALS="ggHgammagamma" ./quick_runners/09_compare_erl_phase2.sh
```
