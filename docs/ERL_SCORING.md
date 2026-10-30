# External ERL scores

The current trainer, validator, single-run wrapper and scan wrapper support
three external scoring plugins. The ERL core and its algorithm are unchanged.
Let **a** be weighted acceptance of the entire anomalous mixture and **b**
weighted reference acceptance; both are fractions in [0, 1].

| `--score` | Formula |
|---|---|
| `gini` | Gini gain − reference penalty × b |
| `excess` | max(0, a − b) / sqrt(b + delta) |
| `ratio` | a / sqrt(b + delta) |

`gini` remains the default, including the existing reference penalty of 1.
Set `--reference-penalty 0` for pure Gini. The two acceptance plugins live in
[`casual_tests/erl_scores.py`](../casual_tests/erl_scores.py); the existing
Gini plugin remains in `_erl_common.py`. The score factory selects among them.

For `excess` and `ratio`, `--score-delta` (alias `--delta`) defaults to **0.001**.
Delta must be positive and finite so every cut, including zero reference passes,
has a finite score. It is an acceptance fraction, not an event count or percent.
For unweighted inputs, delta corresponds to adding delta × N_reference to the
denominator's reference count. The plugin does not automatically rescale delta
when the sample size changes. Larger delta reduces sensitivity to tiny reference
acceptances and weakens the preference for extremely clean small selections.

The reference penalty applies only to `gini`; delta applies only to the other
two scores. Neither acceptance score adds an extra reference penalty. The
`excess` score is zero for equal-acceptance or reference-enriched rules, whereas
`ratio` can be positive for both. These scores use no stored process labels and
are not calibrated discovery significances.

## Train and validate one configuration

```bash
python casual_tests/train_ERL_validation.py \
  --signal data/SIGNAL_train_ggHgammagamma_mixture/anomalous_train.pt \
  --background data/SIGNAL_train_ggHgammagamma_mixture/reference_train.pt \
  --event-only --score excess --score-delta 0.001 \
  --self-prune \
  --output-prefix training_outputs/ggHgammagamma_excess
```

The wrapper inherits the training score and delta for validation unless you
override them with `--pruning-score` and `--pruning-score-delta` (aliases:
`--validation-score` and `--validation-score-delta`). For example, add
`--pruning-score ratio --pruning-score-delta 0.01` to the command above.
The existing training and validation Gini penalties remain independently
configured and retain their defaults.

`train_ERL.py` accepts the same `--score` and `--score-delta` flags and performs
training only. `--beam-score` and `--beam-score-delta` are aliases.

## Scan scores or delta values

```bash
python casual_tests/scan_ERL.py \
  --signal data/SIGNAL_train_ggHgammagamma_mixture/anomalous_train.pt \
  --background data/SIGNAL_train_ggHgammagamma_mixture/reference_train.pt \
  --event-only \
  --score excess ratio --score-delta 0.001 0.01 \
  --validation-score excess --validation-score-delta 0.01 \
  --self-prune \
  --output-prefix training_outputs/ggHgammagamma_score_scan
```

This trains four models, then recalculates their pooled candidates using one
common validation score. Score and delta join the existing Cartesian search
grid; other scan axes still apply. Parameters irrelevant to a score are ignored
by that score, but still form grid axes if multiple values are supplied.

If validation score/delta are omitted, each inherits its uniform training value.
A mixed score grid defaults validation to `gini`; a mixed delta grid defaults
validation delta to 0.001. Set the validation options explicitly for mixed scans.
`--validate-only` supports the same choices without fitting any model.

## Rescore an existing scan

```bash
python casual_tests/validate_ERL.py \
  --manifest training_outputs/ggHgammagamma_scan.scan.json \
  --signal data/SIGNAL_train_ggHgammagamma_mixture/anomalous_validation.pt \
  --background data/SIGNAL_train_ggHgammagamma_mixture/reference_validation.pt \
  --event-only --score excess --delta 0.001 --self-prune \
  --output-prefix training_outputs/ggHgammagamma_excess_validation
```

The standalone validator defaults to Gini with delta 0.001, and accepts
`--pruning-score` / `--pruning-score-delta` as aliases. Original training models
are unchanged. Rescoring fixed candidates does not recover cuts lost during
the original beam search. Local and final score-improvement thresholds remain
absolute values in the selected score's units; they are not rescaled.

## Reports and Python use

Reports record `score_name`, `score_formula`, `score_parameters` and `score_delta`.
Each rule and prefix stores the actual selected score. `gini_gain` and
`prefix_gini_gain` remain diagnostics, with `gini_gain_role=diagnostic_only`
for acceptance scores. Their reference penalty fields are `null` when unused.
The manifest preserves each run's score choice and delta. Core models keep
their existing format: scores/statistics are saved; scoring callables are not.

```python
from casual_tests.erl_scores import ExcessRuleScore, AcceptanceRatioRuleScore

score = ExcessRuleScore(delta=0.001)
model = ExistentialRuleList(..., scoring_function=score)
# Or provide the plugin to load/recalculate/self_prune_candidates.
```

The acceptance plugin classes themselves need only the Python standard library.
The factory imports the existing Gini implementation only when it is requested.
