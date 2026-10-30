# Train one configuration, select rules on validation

`train_ERL_validation.py` is a thin wrapper around `train_ERL.train()` and
`validate_ERL.validate()`. It runs one scalar configuration. For a parameter
grid, use [scan_ERL.py](ERL_SCAN.md).

```bash
python casual_tests/train_ERL_validation.py \
  --signal data/SIGNAL_train_ggHgammagamma_mixture/anomalous_train.pt \
  --background data/SIGNAL_train_ggHgammagamma_mixture/reference_train.pt \
  --event-only \
  --beam-reference-penalty 0 --pruning-reference-penalty 10 \
  --beam-novelty-weight 0.03 \
  --beam-similarity-filter \
  --beam-signal-similarity-threshold 0.99 \
  --beam-background-similarity-threshold 0.99 \
  --final-similarity-filter \
  --final-signal-similarity-threshold 0.9 \
  --final-background-similarity-threshold 0.9 \
  --stricter-score-improvement 0.001 \
  --max-trigger-rate-khz 40 --input-rate-mhz 40 \
  --output-prefix training_outputs/ggHgammagamma \
  --overwrite-output --progress
```

These values illustrate independent controls, not an optimized physics tune.
All numeric search and pruning options take one value. The ordered
`--object-cut-priority` remains a sequence of feature names/directions.

## Scores and thresholds

Use `--score gini|excess|ratio` for training and `--score-delta` / `--delta` for
the regularized acceptance scores. Validation inherits both unless overridden
by `--pruning-score` and `--pruning-score-delta` (also named `--validation-...`).
Gini remains the default. Reference penalties below apply only to Gini; delta
defaults to 0.001 and applies only to excess/ratio. See the
[scoring guide](../docs/ERL_SCORING.md) for exact formulas and examples.

- **Beam score:** Gini gain minus `--beam-reference-penalty` times reference
  acceptance. Default 1; `--reference-penalty` remains an alias for this
  training-only setting.
- **Pruning score:** the same formula with an independent
  `--pruning-reference-penalty`, default 1. Alias:
  `--validation-reference-penalty`. Validation recalculates every candidate
  prefix using this score before pruning; training scores are never reused
  for final ranking. Set both penalties explicitly to keep them equal.
- `--beam-novelty-weight` / `--novelty-beta` affects training exploration only.
  It does not enter the intrinsic validation score.
- `--beam-signal-similarity-threshold` and
  `--beam-background-similarity-threshold` control search-time Jaccard filtering.
  The corresponding `--final-...` options control validation pruning.
- Legacy `--signal-similarity-threshold` and
  `--background-similarity-threshold` are aliases for final thresholds.
  In this single-run wrapper, final values also supply the beam defaults when
  explicit beam thresholds are omitted. Explicit beam values always win.
- `--stricter-score-improvement`, `--require-complete-priority`, and the optional
  trigger-rate limit apply only during validation selection. Similarity filters
  are independently enabled with `--beam-similarity-filter` and
  `--final-similarity-filter`.

## Sequence and outputs

1. The trainer fits once and saves all unpruned candidates.
2. Its training inputs are released.
3. The validator loads the saved model and validation tensors, recalculates
   statistics and scores, then applies final pruning.
4. The validator saves a separate model/report. The training files stay unchanged.

With `--self-prune`, step 3 evaluates and locally prunes shortened versions of
each rule before pooling them for final pruning. `--self-prune-score-improvement`
sets a separate local threshold; otherwise it inherits
`--stricter-score-improvement`. Both stages share the rate and priority-completion
requirements. See [the scan workflow](ERL_SCAN.md#optional-self-pruning-before-final-selection).

Validation paths default to sibling `anomalous_validation.pt` and
`reference_validation.pt`: remove a trailing `_train` from each training stem
and append `_validation.pt`. Override with `--validation-signal FILE` and
`--validation-background FILE`. Use the [split script](../scripts/split_train_validation.py)
to prepare disjoint inputs.

For prefix `training_outputs/ggHgammagamma`:

- `.training.model.json`: unpruned core model; `pruning` is null.
- `.training.report.json`: candidates marked `not_pruned`.
- `.validation.model.json`: recalculated candidates and final selected rules.
- `.validation.report.json`: validation scores, acceptances, origins, and pruning audit.

`--model` and `--training-report` / `--report` override training paths only.
Validation outputs use the prefix. `--overwrite-output` permits replacement.
For native event-only operation, use `--event-only`; single-object, pair, combined,
basic-feature, validity-guard, priority, and progress options are inherited from
the trainer. Search budgets are adjustable, with the previous defaults 4/40/300/32
for depth, beam width, candidate cap, and threshold cap.

Training reports have `summary.final_pruning_applied=false`; false retention flags
mean not yet selected. Plot them with `--rules all`. Validation reports support
ordinary retained-rule plots. Validation is used for selection; keep an untouched
test sample for final performance. Rate limits apply per rule, not to the final OR.
