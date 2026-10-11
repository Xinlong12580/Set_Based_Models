# Set-based models

NumPy implementations of an existential decision tree and beam-search rule
discovery for binary signal/background classification. Rules can combine cuts
on individual objects with cuts on whole events.

## Complete workflow: raw Parquet to an ERL test

This example converts minbias and ggHgammagamma samples, creates a contaminated
anomalous sample, discovers rules with the latest ERL runner, and plots them.
It uses only this repository: neither `aidascoutrepo` nor stage1/stage2
regionization is required.

Run the commands in **Bash from the repository root**, using Python 3.10 or
newer. Supply raw Parquet files with the required `L1T_*` collections, including
PUPPI jets/MET and the PF/object properties described in
[the conversion guide](raw_event_conversion/README.md). Datasets are not
included or downloaded automatically.

### 1. Install dependencies

Create a virtual environment, or skip the first two commands if you already
have an appropriate environment activated:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r raw_event_conversion/requirements.txt
python -m pip install matplotlib pytest
export PYTHONDONTWRITEBYTECODE=1
```

The requirements cover NumPy, PyTorch, Awkward, PyArrow and fsspec. Matplotlib
is needed for plots; pytest is needed for the pytest-based core tests below.

For EOS input, also install `fsspec-xrootd` and the XRootD Python bindings/client
appropriate to your environment, and obtain any required access credentials.
Pass full `root://.../file.parquet` URLs to `--input`, not a sample-directory
URL; the converter does not list remote directories.

### 2. Convert both raw samples

**Replace the two input-directory paths below before running.** The example
outputs use a dedicated directory. Conversion retains up to 10,000 events per
sample; the mixture step needs at least 3,000 minbias events and 100 signal
events. The same feature settings must be used for both samples.

```bash
set -euo pipefail
MINBIAS_PARQUET_DIR="/absolute/path/to/minbias"
SIGNAL_PARQUET_DIR="/absolute/path/to/ggHgammagamma"
mkdir -p data/example_ggHgammagamma reports logs

python -m raw_event_conversion \
  --input "${MINBIAS_PARQUET_DIR}"/*.parquet \
  --output data/example_ggHgammagamma/minbias_10000.pt \
  --label 1 --max-events 10000

python -m raw_event_conversion \
  --input "${SIGNAL_PARQUET_DIR}"/*.parquet \
  --output data/example_ggHgammagamma/ggHgammagamma_10000.pt \
  --label 2 --max-events 10000
```

Each PT file contains real object tensors, event features, schemas, process
labels and provenance. Default shapes are `obj [N,22,20]` and
`event_features [N,193]`. Event summaries use the full stored collections
before object-slot caps. Labels 1 and 2 are diagnostic process identifiers;
they do not define the classes optimized by ERL.

### 3. Build reference and anomalous datasets

```bash
python scripts/make_reference_anomalous.py \
  --background data/example_ggHgammagamma/minbias_10000.pt \
  --signal data/example_ggHgammagamma/ggHgammagamma_10000.pt \
  --output-dir data/example_ggHgammagamma/mixture \
  --reference-count 2000 --background-count 1000 --signal-count 100 \
  --seed 42
```

- `reference.pt`: 2,000 minbias events.
- `anomalous.pt`: 1,000 minbias events plus 100 ggHgammagamma events, shuffled.
- The reference and mixture use disjoint minbias rows. The anomalous input is
  about 91% background; ERL compares the whole mixture against the reference,
  without using process labels in its score.

To study another process, replace the signal raw files and use a distinct
output directory; the conversion, mixer and runner are not specific to Higgs.

### 4. Run the latest ERL discovery example

Start with event-only inputs. This exercises penalized Gini scoring, built-in
logical pruning and both similarity stages:

```bash
python casual_tests/run_ERL_Gini_nested.py \
  --signal data/example_ggHgammagamma/mixture/anomalous.pt \
  --background data/example_ggHgammagamma/mixture/reference.pt \
  --event-only \
  --reference-penalty 1 --stricter-score-improvement 0.001 \
  --beam-similarity-filter --final-similarity-filter \
  --signal-similarity-threshold 0.9 --background-similarity-threshold 0.9 \
  --report reports/ERL_example_ggHgammagamma_event_only.txt \
  --overwrite-report | tee logs/ERL_example_ggHgammagamma_event_only.log
```

These are demonstration settings, not a universally optimal physics tune.
The report is **JSON despite its `.txt` extension**. It records candidates,
scores, acceptances, configuration and final-pruning decisions.

- For **combined** discovery, remove `--event-only`; optionally add
  `--object-cut-priority type_id pt`.
- For **object-only** discovery, replace `--event-only` with `--object-only`;
  optionally add `--basic-object-features --object-cut-priority type_id pt`.
- For **particle-pair** discovery, convert both raw samples with
  `--with-pair-tensor`, then run the flexible ERL script with
  `--pair-tensor --object-only --object-cut-priority type_id_1 type_id_2`.
  Omit `--object-only` to also use event features. See the
  [pair-tensor guide](raw_event_conversion/PAIRS.md) for features and a command.
- Add `--progress` to the flexible ERL runner for a live beam-search progress
  display: depth, processed parents, candidate cuts, beam filtering and elapsed
  time. It is off by default and writes to stderr, so `> run.log` or
  `| tee run.log` still captures only stdout. Redirect `2>&1` to capture progress
  as well; non-terminal output uses occasional plain lines rather than a live
  bar. Progress covers candidate generation, not final pruning or report writing.
- Use different report/log names for each configuration. Object priority
  cannot be used in event-only mode.
- To request a nominal per-rule rate limit, add
  `--max-trigger-rate-khz 40 --input-rate-mhz 40`. This means reference
  acceptance at most 0.1%, or only two observed events in this reference.
  The limit does not constrain the combined OR rate or include uncertainty.

### 5. Plot the discovered rules

```bash
python scripts/plot_acceptance_report.py \
  reports/ERL_example_ggHgammagamma_event_only.txt \
  --output reports/ERL_example_ggHgammagamma_event_only_acceptance.png \
  --top 10 --overwrite
```

The plot shows retained, mixture-enriched rules: reference acceptance on the
horizontal axis, mixture acceptance vertically, and rule definitions alongside.
If no rules satisfy that display selection, use `--rules all --direction all`
to inspect the candidate pool rather than assuming discovery succeeded.

For mixtures converted with `--with-pair-tensor`, inspect the diphoton mass
distribution directly (no ERL fit is required):

```bash
python scripts/plot_diphoton_mass.py data/ggHgammagamma_mixture/anomalous.pt \
  --output reports/ggHgammagamma_diphoton_mass.png --mass-range 0 250 --bins 100
```

The plot overlays the mixture and its stored process labels and excludes
padded/invalid pairs. By default all photon pairs contribute; add
`--pairs leading` for one entry per event from the two highest-PT stored
photons. Optional `--leading-pt-min`, `--subleading-pt-min`, and `--sum-pt-min`
apply strict PT cuts in GeV. `--log-y` enables a logarithmic count axis.

Rerunning this workflow replaces its example outputs: conversion and mixing
warn before overwrite; the report and plotting commands explicitly allow it.
Use `--no-overwrite` for conversion/mixing, or omit the corresponding overwrite
flags for reports/plots, to protect existing outputs. Do not commit generated
datasets, logs, plots or the virtual environment.

### 6. Run regression checks

The latest flexible-runner tests use small, temporary synthetic PT files and
do not require EOS or the samples converted above:

```bash
PYTHONPATH=src:. python -m unittest discover \
  -s tests -p test_flexible_erl_runner.py -v
```

For core regression coverage, including native input modes, validity guards,
logical pruning and object priority:

```bash
PYTHONPATH=src:. python -m pytest -q \
  tests/test_data.py tests/test_objectives.py tests/test_tree.py \
  tests/test_rule_objectives.py tests/test_rule_union.py tests/test_rule_list.py \
  tests/test_event_features.py tests/test_erl_input_modes.py \
  tests/test_rule_list_pruning.py tests/test_validity_guards.py \
  tests/test_object_cut_priority.py
```

The discovery run is a proof-of-concept, **not independent validation**: its
acceptances are measured on the training inputs. Before proposing a trigger,
evaluate rules and their OR on held-out events and assess rate uncertainty,
reconstruction stability and hardware constraints.

See the [technical software reference](reports/ERL_implementation.md)
([PDF](reports/ERL_implementation.pdf)) or the
[introductory report](reports/ERL_introductory_report.pdf) for more detail.

For report-analysis integrations, start with the [AI analysis guide](docs/AI_GUIDE.md)
and [report-format reference](docs/REPORT_FORMAT.md). The
[JSON Schema](schemas/quantitative_report.schema.json) documents the versioned
structure, and [small annotated examples](examples/reports/README.md) illustrate
EDT/ERL interpretation without real datasets. These files do not enable a
GitHub connection or change model behavior.

## Inputs

Supply signal and background separately. For each class:

| Argument | Shape | Meaning |
| --- | --- | --- |
| Slot tensor | `(n_samples, n_slots, n_observables)` | Object features, such as pt, eta, type |
| Slot mask | `(n_samples, n_slots)` | Boolean mask: `False` means padding |
| Event features | `(n_samples, n_event_observables)` | Optional precomputed HT, MET, multiplicities, etc. |
| Sample weights | `(n_samples,)` | Optional finite, non-negative event weights |

The event and slot arrays must have matching sample rows. Signal and background
must share the slot/feature shape and schemas. Every observable has an
`ObservableSpec`: continuous, ordinal, categorical, or boolean. Categorical
values are numeric codes, and ordinal/boolean values must be exact integers.
Valid measurements must be finite; values excluded by schema-declared validity
guards are ignored during measurement validation. Provide a slot mask to
exclude zero padding.

## Event and slot cuts

```python
from set_based_models import (
    ExistentialDecisionTree, ExistentialRuleList, ObservableSpec,
    TPRFPRRuleScore, gini_gain,
)

observables = (
    ObservableSpec("pt", "continuous"),
    ObservableSpec("btag", "continuous"),
)
event_observables = (
    ObservableSpec("HT", "continuous"),
    ObservableSpec("MET", "continuous"),
    ObservableSpec("n_electrons", "ordinal"),
)

tree = ExistentialDecisionTree(
    observables, event_observables=event_observables,
    max_depth=4, gain_function=gini_gain,
)
rules = ExistentialRuleList(
    observables, event_observables=event_observables,
    max_rule_depth=4, scoring_function=TPRFPRRuleScore(),
)

# signal/background: (N, slots, 2); event matrices: (N, 3).
for model in (tree, rules):
    model.fit(
        signal, background,
        signal_event_features=signal_events,
        background_event_features=background_events,
        signal_slot_mask=signal_mask,
        background_slot_mask=background_mask,
    )

predictions = tree.predict(X, event_features=events, slot_mask=mask)
best_rule = rules.get_rule(0)
print(best_rule.to_text())
rule_predictions = best_rule.predict(X, event_features=events, slot_mask=mask)
union = rules.build_union([0])  # Or choose several candidate indices.
union_predictions = union.predict(X, event_features=events, slot_mask=mask)
```

By default both searches consider cuts from either scope at every step.
ERL can optionally restrict the order of object features (see below). A rule may read:

```text
event: HT > 600 AND event: MET > 200 AND (EXISTS slot: pt > 100 AND btag > 0.8)
```

Slot conditions must hold on the same objects. `minimum_count` controls how many
objects must satisfy all slot cuts. Event cuts directly select events and do
not alter their surviving objects. An event-only rule does not impose an object
count and can accept events whose slot masks are entirely false. All cuts count
toward tree/rule depth. Scorers and metrics use the same event-level yields for
both scopes.

Event schemas are optional: omit them and the event arrays for the original
slot-only API. Once a schema is configured, supply its full event matrix for
fitting and event prediction, including when an individual learned rule uses
only some (or none) of those features.

## ERL input modes

ERL accepts object tensors, event matrices, or both. Configure schemas for the
inputs you use; at least one schema is required.

```python
# Object-only: existing calls are unchanged.
object_rules = ExistentialRuleList(observables, scoring_function=TPRFPRRuleScore())
object_rules.fit(signal, background,
                 signal_slot_mask=signal_mask, background_slot_mask=background_mask)

# Event-only: no object tensor, object schema, or dummy slots are needed.
event_rules = ExistentialRuleList(
    event_observables=event_observables, scoring_function=TPRFPRRuleScore(),
)
event_rules.fit(signal_event_features=signal_events,
                background_event_features=background_events)
# After fitting, assuming at least one candidate was found:
selection = event_rules.get_rule(0).predict(event_features=events)
union_selection = event_rules.build_union([0]).predict(event_features=events)

# Combined: supply both schemas and both tensor types, as in the example above.
```

Object arguments remain 3D; pass 2D event matrices using the named
`signal_event_features` / `background_event_features` arguments. Both classes
must use the same input mode. For event-only fitting, omit object masks as well
as object tensors. `input_mode_` records `object_only`, `event_only`, or
`combined` after fitting. Event-only rules ignore `minimum_count`; rules with
object cuts still require objects at prediction time. Search, scoring, and
pruning settings are unchanged.

The event-only ERL runners (ordinary, penalized, and nested Gini) use this
native mode; their command-line arguments are unchanged. The event-only EDT
runner still uses its existing fully masked dummy-slot implementation.

## ERL search memory

Mask packing is automatic for all input modes; no runner switch or tensor
regeneration is needed. Temporary expansions store event and slot masks as
packed bits, shared with the exact duplicate-detection key. Beam similarity
uses bitwise Jaccard comparisons without unpacking every candidate.

Only the current parent is unpacked for cut generation and evaluation.
Non-retained expansions are released after beam selection; completed-layer
results keep cut histories and packed event selections, not slot masks.
Public rules still expose ordinary Boolean event selections and predictions.
This changes storage, not scores, search coverage, ranking or pruning policies.

Packing reduces mask payload approximately eightfold, but input arrays,
validity arrays and other allocations still require RAM. Candidate generation
still precedes beam selection: beam width is not a bound on expansion count.

## ERL object-cut priority

An optional ordered list restricts the sequence of object cuts:

```python
rules = ExistentialRuleList(
    observables, object_cut_priority=["type_id", "pt"],
    scoring_function=TPRFPRRuleScore(),
)
```

The next object cut must use `type_id`, then `pt`; afterwards all enabled
object features are available, including those already used. This does not
fix the category, threshold, or comparison direction. Event cuts remain
unrestricted and do not advance object priority. All cuts still consume depth.

Repeated names are allowed and consume separate positions. For example,
`--object-cut-priority type_id pt pt` requires two successive momentum cuts
after the type cut. This permits a two-sided interval but does not force
opposite comparison directions; both cuts may refine the same bound.
Beam selection and final pruning still apply to these rules.

Numerical entries can optionally specify `>`, `>=`, `<`, or `<=`. For example:

```bash
--object-cut-priority type_id 'pt:>' 'pt:<='
```

This requires a lower momentum bound followed by an upper bound, with both
thresholds optimized from the data. Quote these entries to prevent shell
redirection. Python also accepts `(feature, direction)` pairs, for example
`object_cut_priority=["type_id", ("pt", ">"), ("pt", "<=")]`.
Directions apply only to numerical features; categorical entries remain plain
names. A window still takes two search steps, and its first bound must survive
beam selection. Plain names retain unrestricted comparison directions.

Incomplete rules, such as `EXISTS type_id in {photon}`, remain eligible for
ranking and final selection by default; completing the list is optional. Normal
acceptance, beam-width, score, candidate-cap, and pruning limits still apply.
If a required feature has no usable cut, that prefix cannot add another object
cut; it can still be returned or extended with event cuts. There is no fallback
to a lower-priority object feature. No priority (the default, `()` or `None`)
preserves the unrestricted search.

Names must belong to the object schema and be enabled for discovery; repeated
entries are supported. Guard-only validity flags cannot be prioritized without opting into
standalone validity cuts. Priorities require actual object inputs at fitting.
Beam deduplication/similarity include priority progress. Final logical and
similarity filters remain independent of the priority order.

Add `--require-complete-priority` to retain only rules consuming every priority
entry, including repeats and specified directions. Incomplete prefixes remain
available during beam search and in the report's candidate pool, but are
excluded before final rate, equivalence, logical, and similarity pruning.
For retained rules containing both bounds, use:

```bash
--object-cut-priority type_id 'pt:>' 'pt:<=' \
  --require-complete-priority
```

The core API is `model.prune_candidates(require_complete_priority=True, ...)`.
This is off by default; an empty priority list imposes no restriction. If no
complete candidate survives search, final selection is empty rather than
falling back to incomplete rules. Completion uses original cut histories,
not display-simplified predicates.

The flexible runner exposes this as:

```bash
python casual_tests/run_ERL_Gini_nested.py \
  --signal data/ggHgammagamma_mixture/anomalous.pt \
  --background data/ggHgammagamma_mixture/reference.pt \
  --object-only --basic-object-features --object-cut-priority type_id pt \
  --report reports/ERL_ggHgammagamma_priority_objects.txt
```

Without `--object-only`, event cuts can interleave with the ordered object cuts.
Reports record the priority order and remaining priority features for each rule.

## Rule evaluation helpers

For manual rules/trees, wrap an existing predicate to identify its event scope:
`EventCut(ThresholdCut(0, ">", 600))` or `EventCut(CategoryCut(0, {1}))`.
An event tree node has `aggregation=None`. Indices refer to the schema for that
scope; the two schemas can share feature names. Tree `cut_chain()` records
both scopes, while `slot_cut_chain()`/`slot_cut_function()` extract only the
positive slot conditions and omit event routing.

Rule `slot_mask()` includes the event conditions but does not apply
`minimum_count`; use `predict()` for acceptance. `matches_slot()` accepts one
event feature vector. `matches_slots()` accepts a shared vector, per-event
features for a 3D slot tensor, or per-slot event features. These helpers evaluate
conditions without count aggregation. A union aggregates each rule separately
before taking the logical OR.

The final logical-pruning threshold accepts negative values. For example,
`--stricter-score-improvement -0.008` allows a stricter rule's score to drop by
less than 0.008 relative to applicable retained looser rules. Positive values
require improvement; zero requires a strictly better score. Rate and similarity
filters still apply independently, and ranking remains score-based.

## Automatic validity guards

An observable can declare a Boolean validity column in the same scope:

```python
observables = (
    ObservableSpec("n_neutrals", "ordinal", validity_column="n_neutrals_valid"),
    ObservableSpec("n_neutrals_valid", "boolean"),
)
```

Every cut on `n_neutrals` then requires `n_neutrals_valid == True`, on the same
object. The same declaration works in `event_observables`. Guards are enforced
by training, threshold generation, rule prediction, slot helpers and unions;
they do not add rule steps. Invalid values are ignored while measured zeros
remain valid. Model schemas retain the association, so prediction does not
require callers to reconstruct masks. Logical pruning accounts for the guards.

The flexible ERL runner reads existing object `validity_column` and event
`definition.validity_feature` metadata automatically. Native event-only ERL
runners do so as well; the legacy event-only EDT loader is unchanged. Inputs
without validity associations retain their previous behavior. In ERL discovery,
schema-linked flags are **guard-only by default**: neither standalone True nor
False cuts are generated. Ordinary Boolean observables remain selectable.
Use `allow_validity_cuts=True` in the core model, or `--allow-validity-cuts` in
the flexible/native event-only ERL runners, to opt into explicit flag cuts.
Automatic measurement guards stay enabled even with this opt-in. Tensor shapes
and manually constructed rules are unchanged. These masks do not remove
dataset-dependent missingness or validate unflagged measurements.

## Running

The library requires NumPy; set `PYTHONPATH=src` to import it from this checkout.
The synthetic example runs without a dataset:

```bash
python casual_tests/run_mixed_observables.py
```

With pytest installed, run the tests with:

```bash
PYTHONPATH=src python -m pytest -q
```

The other scripts in `casual_tests` additionally use PyTorch and external HEP
datasets. The older `set_based_dt` imports remain available as compatibility
aliases.
