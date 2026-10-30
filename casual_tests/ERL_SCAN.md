# Scan training settings, then select on validation

Three small drivers reuse the existing tensor loader, penalized-Gini score,
progress display, and quantitative report format:

| File | Purpose |
| --- | --- |
| `train_ERL.py` | Train one scalar configuration and save its unpruned model and report. |
| `validate_ERL.py` | Load and merge candidates, optionally self-prune each rule on validation, and apply final pruning. |
| `scan_ERL.py` | Run a training grid, validate a saved scan, or run both stages. |

The original trainers and quick runners remain available. No data conversion or
mixture construction changes are required. Run these commands from the repository
root using the same Python environment as the existing trainer.

For a single configuration followed by validation, use
[`train_ERL_validation.py`](ERL_VALIDATION.md). It calls the same trainer and
validator. The former `train_ERL_scan.py` entry point is now `train_ERL.py`.

## One command for both stages

```bash
python casual_tests/scan_ERL.py \
  --signal data/SIGNAL_train_ggHgammagamma_mixture/anomalous_train.pt \
  --background data/SIGNAL_train_ggHgammagamma_mixture/reference_train.pt \
  --event-only \
  --beam-width 40 \
  --beam-novelty-weight 0 0.03 \
  --beam-similarity-filter \
  --beam-signal-similarity-threshold 0.95 0.99 \
  --beam-background-similarity-threshold 0.99 \
  --reference-penalty 0 \
  --validation-reference-penalty 0 \
  --self-prune \
  --final-similarity-filter \
  --final-signal-similarity-threshold 0.9 \
  --final-background-similarity-threshold 0.9 \
  --stricter-score-improvement 0 \
  --max-trigger-rate-khz 40 --input-rate-mhz 40 \
  --output-prefix training_outputs/ggHgammagamma_scan \
  --overwrite-output --progress
```

This trains **four models**: two novelty weights times two mixture beam
thresholds. Every list is an independent axis of a Cartesian grid, including
the separate mixture/reference beam thresholds. Lists are not zipped.

The wrapper defaults validation paths to sibling `anomalous_validation.pt` and
`reference_validation.pt`. More generally, it removes a trailing `_train` from
each training stem and appends `_validation.pt`. Override with
`--validation-signal FILE` and `--validation-background FILE`.

## Run the stages separately

Train one configuration directly:

```bash
python casual_tests/train_ERL.py \
  --signal data/SIGNAL_train_ggHgammagamma_mixture/anomalous_train.pt \
  --background data/SIGNAL_train_ggHgammagamma_mixture/reference_train.pt \
  --event-only --reference-penalty 0 \
  --beam-novelty-weight 0.03 \
  --beam-similarity-filter \
  --beam-signal-similarity-threshold 0.99 \
  --beam-background-similarity-threshold 0.99 \
  --output-prefix training_outputs/ggHgammagamma_single --progress
```

This writes `.training.model.json` and `.training.report.json` at that prefix.
The trainer accepts one value per numeric option and does not create a scan
manifest. To train a grid without running validation, use the wrapper:

```bash
python casual_tests/scan_ERL.py --train-only \
  --signal data/SIGNAL_train_ggHgammagamma_mixture/anomalous_train.pt \
  --background data/SIGNAL_train_ggHgammagamma_mixture/reference_train.pt \
  --event-only --reference-penalty 0 \
  --beam-novelty-weight 0 0.03 \
  --beam-similarity-filter \
  --beam-signal-similarity-threshold 0.95 0.99 \
  --beam-background-similarity-threshold 0.99 \
  --output-prefix training_outputs/ggHgammagamma_scan --progress
```

Merge and select on validation, without training again:

```bash
python casual_tests/scan_ERL.py --validate-only \
  --signal data/SIGNAL_train_ggHgammagamma_mixture/anomalous_train.pt \
  --background data/SIGNAL_train_ggHgammagamma_mixture/reference_train.pt \
  --event-only \
  --output-prefix training_outputs/ggHgammagamma_scan \
  --validation-reference-penalty 0 \
  --self-prune --self-prune-score-improvement 0.001 \
  --stricter-score-improvement 0 \
  --final-similarity-filter \
  --final-signal-similarity-threshold 0.9 \
  --final-background-similarity-threshold 0.9 \
  --max-trigger-rate-khz 400 --input-rate-mhz 40 \
  --overwrite-output
```

No model filenames or manifest argument are needed. Use the same
`--output-prefix` as the training scan; when omitted, the usual prefix derived
from the signal parent directory and feature mode is used.

- If `<prefix>.scan.json` exists, exactly the models recorded there are loaded.
  Other files left under the prefix are ignored.
- Without a manifest, the wrapper discovers sorted
  `<prefix>.run[0-9]*.training.model.json` files. All matching training models are
  used; validation models and other prefixes are excluded.
- Training is skipped entirely, including loading the training tensors. Search
  limits and beam filters are unused. Score/delta arguments can supply the
  validation defaults; set `--validation-score` and `--validation-score-delta`
  explicitly to choose the new objective. Other grid parameters need not be
  supplied again. Set self-pruning and final-pruning options for the new selection.
- With `*_train.pt` input names, validation siblings are inferred as in the
  combined workflow. Other input filenames, including `*_validation.pt`, are
  used directly in validate-only mode. `--validation-signal` and
  `--validation-background` always override that inference.
- Validation artifacts use the same prefix. Existing outputs still require
  `--overwrite-output`; training models and the scan manifest are not rewritten.
  `--train-only` and `--validate-only` are mutually exclusive.

The step 5 shell script forwards extra arguments, so it supports both modes:

```bash
bash quick_runners/05_train_SIG_ERL.sh --train-only
bash quick_runners/05_train_SIG_ERL.sh --validate-only
```

For explicit manifest selection or a different validation output prefix, use
the standalone validator:

```bash
python casual_tests/validate_ERL.py \
  --manifest training_outputs/ggHgammagamma_scan.scan.json \
  --signal data/SIGNAL_train_ggHgammagamma_mixture/anomalous_validation.pt \
  --background data/SIGNAL_train_ggHgammagamma_mixture/reference_validation.pt \
  --event-only --reference-penalty 0 \
  --final-similarity-filter \
  --final-signal-similarity-threshold 0.9 \
  --final-background-similarity-threshold 0.9 \
  --stricter-score-improvement 0 \
  --max-trigger-rate-khz 40 --input-rate-mhz 40
```

Instead of a manifest, pass explicit paths or quoted wildcard patterns:

```bash
python casual_tests/validate_ERL.py \
  --models 'training_outputs/ggHgammagamma_scan.run*.training.model.json' \
  --signal data/SIGNAL_train_ggHgammagamma_mixture/anomalous_validation.pt \
  --background data/SIGNAL_train_ggHgammagamma_mixture/reference_validation.pt \
  --event-only --reference-penalty 0 \
  --final-similarity-filter --final-signal-similarity-threshold 0.9 \
  --final-background-similarity-threshold 0.9 \
  --output-prefix training_outputs/ggHgammagamma_combined
```

`--signal-similarity-threshold` and `--background-similarity-threshold` are
aliases for the **final** thresholds in validation and the wrapper. They do not
change beam thresholds. Both stages use Jaccard similarity; these options change
thresholds, not the similarity metric.

## Training settings and defaults

Scoring is selectable with `--score gini|excess|ratio` (alias `--beam-score`).
`--score-delta` / `--delta` controls the regularized acceptance scores. Both
accept lists and are Cartesian scan axes; the scalar trainer accepts one value.
Validation uses `--validation-score` and `--validation-score-delta`, with
`--pruning-score` / `--pruning-score-delta` aliases. Each defaults to the uniform
training value; mixed score/delta grids fall back to Gini/0.001 respectively.
The standalone validator accepts `--score` and `--score-delta` directly.
See [external scoring](../docs/ERL_SCORING.md) for formulas and examples.

The trainer accepts one value for each numeric option. Only the wrapper accepts
lists and expands them into combinations:

| Option | Default |
| --- | ---: |
| `--max-rule-depth` | 4 |
| `--beam-width` | 40 |
| `--max-candidate-rules` | 300 |
| `--max-thresholds` | 32 |
| `--minimum-count` | 1 |
| `--min-signal-efficiency` | 0.01 |
| `--reference-penalty` | 1 |
| `--score` | gini |
| `--score-delta` | 0.001 |
| `--beam-novelty-weight` / `--novelty-beta` | 0 |
| `--beam-signal-similarity-threshold` | 0.1 |
| `--beam-background-similarity-threshold` | 0.1 |

Common flags apply to every run: `--beam-similarity-filter`,
`--object-cut-priority FEATURE[:DIRECTION] ...`, `--allow-validity-cuts`, and
`--progress`. Beam similarity filtering is off unless enabled. Novelty remains
available when hard similarity filtering is off. Higher similarity thresholds
prune less. Signed novelty weights are supported.

Each model is saved **without final list pruning**. Search-time filtering and
the per-run candidate cap still apply. The wrapper loads training data once and
passes the same input tuple to each scalar training call. Models are fitted and
saved sequentially rather than kept in memory together. The training tuple is
released before the wrapper loads validation data.

Other runners can call `train_ERL.train(args, data=None)` directly with a
scalar argument namespace. It returns `(model_path, report_path)` after one fit.
Omitting `data` loads the inputs for that call; passing the tuple returned by
`load_inputs(signal_path, background_path, args)` reuses loaded inputs. Grid
expansion, run naming, manifests, and validation belong to the wrapper.

## Validation selection

Validation has its own scalar reference penalty, final similarity thresholds,
logical improvement threshold, priority-completion flag, and optional rate limit.
The wrapper calls its penalty `--validation-reference-penalty`; the standalone
validator calls it `--reference-penalty`. Both also accept
`--pruning-reference-penalty`. Its default is 1, independently of the training
penalty list. Training accepts `--beam-reference-penalty` as an alias for
`--reference-penalty`.

Final similarity filtering is off by default; both final thresholds default to
0.1. The logical improvement threshold defaults to 0, accepts negative values,
and applies after priority completion and the optional per-rule rate limit.
`--require-complete-priority` checks each candidate's own source priority.

All candidates are merged using the core `load_many()` API. Existing scores
from different searches are replaced by common validation scores before pruning.
Reports preserve per-source search settings, original rule IDs, and, when using
the manifest, training penalty choices. No new beam search runs on validation.

### Optional self-pruning before final selection

Add `--self-prune` to the scan command or standalone validator. The validation
stage then calls the core `self_prune_candidates()` directly instead of first
recalculating the unchanged pool:

1. Merge the unpruned training candidates.
2. Evaluate all nonempty shortened versions of each rule on validation, including
   guard-only versions, and prune within each rule family.
3. Pool and deduplicate the survivors, compute their event Jaccard matrices, then
   apply the existing final logical and similarity pruning.

Self-pruning uses the common validation score. Its improvement threshold defaults
to `--stricter-score-improvement`. Set `--self-prune-score-improvement VALUE` to
choose an independent local threshold; negative values are accepted. Both stages
use the same reference-rate limit and source-priority completion requirement.
These are scalar validation settings, not additional scan axes.

Self-pruning is off unless requested. The active scan in
`quick_runners/05_train_SIG_ERL.sh` enables it. `--train-only` never self-prunes,
and training artifacts stay unchanged. The single-configuration wrapper also
accepts the same two options through the shared validator.

There is no need to retrain an existing scan to use it: add `--self-prune` to the
standalone `validate_ERL.py --manifest ...` command above. The validation model
contains the shortened candidate pool plus its final selection. Its report adds
`self_pruning_provenance`, local expansion counts, and per-rule
`self_pruning_parent_rule_ids` referencing the merged pool before shortening.
See the [core self-pruning guide](../docs/ERL_MODEL_FORMAT.md#shorten-each-rule-before-final-pruning)
for the subset and validity-guard semantics.

Use the same input flags for both stages: `--event-only`, `--object-only`,
`--pair-tensor`, and `--basic-object-features` work as in the old trainer. Without
a modality flag, both object and event features are enabled. Schemas must match;
single-object and pair models are separate pools. Training outputs stay unchanged
when validation runs. The final report can be loaded by the existing plotting script.

## Outputs

For prefix `training_outputs/ggHgammagamma_scan`:

- `.run000.training.model.json`, `.run001.training.model.json`, etc.: core models.
- `.run000.training.report.json`, etc.: unpruned training reports.
- `.scan.json`: model/report paths and search settings for every run.
- `.validation.model.json`: pooled candidates with refreshed statistics and final selection.
- `.validation.report.json`: score-ranked candidates, origins, and pruning decisions.

Manifest paths are absolute, so it can be used from another working directory.
Validation defaults to the manifest's output prefix; `--output-prefix` overrides
it. With `--models` and no prefix, validation uses `training_outputs/ERL_combined`.
Existing outputs require `--overwrite-output` (alias `--overwrite-report`).

Validation is used for selection. Keep an independent test sample for the final
performance measurement. The optional rate limit applies to each rule, not to the
combined OR of retained rules.
