# Event-only EDT and Gini ERL

These runners search only the `event_features` tensor. ERL uses native event-only
input: no object tensor, object schema, or dummy slots are supplied to the model.
The EDT runner retains its zero-valued dummy tensor `[N,1,1]` and false slot mask
`[N,1]`; its masked continuous placeholder generates no object predicates.
The actual `obj` and `pf` tensors are ignored and can be absent. The penalized
and nested event-only ERL runners also use native input.

Run from the repository root in the `hep` environment, substituting your actual
event-feature mixture and reference paths:

```bash
python casual_tests/run_EDT_event_only.py \
  --signal data/ZZ_event_features/anomalous.pt \
  --background data/ZZ_event_features/reference.pt \
  --report reports/EDT_ZZ_event_only.txt

python casual_tests/stale/run_ERL_Gini_event_only.py \
  --signal data/ZZ_event_features/anomalous.pt \
  --background data/ZZ_event_features/reference.pt \
  --report reports/ERL_ZZ_event_only.txt
```

The example input directory is a placeholder, not a directory created by these
scripts. First convert both samples using `data_conversion.stage1_to_pt` with
matching settings, then use `scripts/make_reference_anomalous.py` if you need a
statistically disjoint reference and a background-contaminated anomalous region.
That mixer carries all top-level event tensors through selection and shuffling.
Passing a pure signal `.pt` as `--signal` instead tests pure signal/reference
separation, not the heavily contaminated anomaly-region setting.

## Inputs

Required keys are `event_features [N,F]`, `event_feature_columns`, and
`label [N]`. Input feature columns, order, types, and saved definitions must match
between the two files. Empty inputs and invalid measurements are rejected.
Labels are used only for diagnostic counts; every signal-file row is treated
as the mixture class irrespective of its stored process label.

The stage1 multiplicity converter's default `event_features [N,20]` contains the requested
multiplicities above zero and positive PT thresholds. When `event_feature_schema`
is present, its observable kinds are respected, including future finite
continuous extensions. Legacy files without a schema are interpreted as
nonnegative ordinal multiplicities.

The separate `native_event_info` tensor is **not** automatically used. Its
masked/undefined measurements need a validity-aware modeling policy before
being included. These runners do not treat missing native values as physical
zeros or silently append them to the event features.

## Settings and reports

Settings match the existing quantitative event-aware runners:

- EDT: weighted Gini gain, maximum depth 6, minimum leaf population 20,
  maximum 64 thresholds per observable, default classification threshold 0.5.
- ERL: weighted Gini gain, depth 3, beam width 40, maximum 200 candidate rules,
  maximum 32 thresholds, minimum mixture acceptance 0.01, beam and final
  similarity thresholds inherited from `_quantitative_reporting.py` (currently
  0.1 for both classes).

Both print readable logs and save JSON in a `.txt` report. Reports retain the
existing plotting schema, explicitly mark `feature_mode: event_only`, and
contain no active object observables. Acceptances are evaluated on the training
inputs, not an independent validation sample. Similarity pruning and ranking
are unchanged; reference-enriched Gini rules are identified separately.
Native ERL reports additionally record `native_event_only: true`,
`dummy_object_input: null`, and internal object-array shape `[N,0,0]`. These
empty bookkeeping arrays contain no objects and are not passed to ERL fitting.

`--report` chooses the output name. Without it, the default is
`reports/<model>_<signal-parent>_<signal-stem>_event_only.txt` under the working
directory. Existing reports are protected; use `--overwrite-report` to replace
one. Input tensors cannot be used as report targets. `--print-top 0` prints all
ranked results. ERL also accepts `--print-reference-top` and `--print-pruned-top`.

Plot a new report with the existing plotting script:

```bash
python scripts/plot_acceptance_report.py reports/ERL_ZZ_event_only.txt \
  --output reports/ERL_ZZ_event_only.png
```

Synthetic verification only (no production datasets):

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover \
  -s tests -p test_event_only_runners.py -v
```
