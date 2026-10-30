# Reference-penalized event-only Gini ERL

`run_ERL_Gini_penalized_event_only.py` is a separate runner using native
event-only ERL input, without a dummy object tensor. Inputs are the same event-level
`.pt` files as `run_ERL_Gini_event_only.py`. Process labels are diagnostic only;
real objects and the separate `native_event_info` tensor are not used.

```bash
python casual_tests/stale/run_ERL_Gini_penalized_event_only.py \
  --signal data/ggHgammagamma_mixture/anomalous.pt \
  --background data/ggHgammagamma_mixture/reference.pt \
  --report reports/ERL_ggHgammagamma_penalized_event_only.txt \
  --reference-penalty 1.0 \
  > ggHgammagamma_penalized.log
```

Add `--overwrite-report` to replace an existing report. Without `--report`, a
distinct `reports/ERL_Gini_penalized_<signal-parent>_<signal-stem>_event_only.txt`
is used under the working directory. Input files cannot be overwritten.

## Adjustable score

The larger-is-better score is:

```text
score = raw weighted Gini gain - lambda * reference acceptance
```

`--reference-penalty` sets lambda, default **1.0**. Acceptance is a fraction,
not a percentage: at the default, 1% reference acceptance subtracts 0.01 from
the Gini gain. Increasing lambda punishes reference acceptance more strongly.
Zero recovers ordinary Gini scoring with the same search settings. Lambda must
be finite and nonnegative. The default is a starting point, not a calibrated
trigger-rate limit or a known-signal optimization.

The scorer is passed directly to ERL and used at every beam-search depth and
for final ranking. Negative scores remain eligible; there is no hard reference
acceptance cutoff on intermediate prefixes. Beam width and similarity pruning
can still remove prefixes. Depth 3, beam width 40, 200 candidate rules, 32
thresholds, and minimum mixture acceptance 0.01 match the original runner.

Similarity thresholds default to the **current** values in
`_quantitative_reporting.py`; changing the penalty alone does not change them.
To preserve more overlapping alternatives, optionally use:

```bash
  --signal-similarity-threshold 0.95 --background-similarity-threshold 0.95
```

Both flags apply to beam filtering and final pruning. Higher thresholds prune
less aggressively. `--no-beam-similarity-filter` disables beam similarity
filtering, but not final report pruning; all surviving candidate rules are
always saved in the report.

## Reports and interpretation

Logs and JSON `.txt` reports separate `score`, genuine `gini_gain`, and
`reference_penalty_term`, including their prefix histories. Rules and pruned
rules are ranked by **penalized score**, not raw Gini. The existing plotter
accepts these reports and respects their order; its Gini annotations still show
the raw gain, not the penalized score.

```bash
python scripts/plot_acceptance_report.py \
  reports/ERL_ggHgammagamma_penalized_event_only.txt \
  --output reports/ERL_ggHgammagamma_penalized_event_only.png
```

All acceptances are training-sample measurements. Zero observed reference
events does not establish zero background rate. Increasing the penalty cannot
favor double photons over single photons when both already accept zero
reference events, nor recover photons absent from the producer collection.
Use independent validation and inspect complementary signatures, not only the
top-ranked rule.
