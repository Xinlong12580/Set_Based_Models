# Event-only ERL with unified logical-subset pruning

This runner keeps its existing filename for compatibility and calls the ERL
model's built-in final pruning directly. Other runners retain their previous
default behavior.
It uses the same event-feature inputs, native event-only ERL (no objects), training-class
semantics, and adjustable `Gini - lambda * reference acceptance` score as the
penalized runner. Native event info and stored process labels are not model inputs.

```bash
python casual_tests/stale/run_ERL_Gini_nested_event_only.py \
  --signal data/ggHgammagamma_mixture/anomalous.pt \
  --background data/ggHgammagamma_mixture/reference.pt \
  --report reports/ERL_ggHgammagamma_nested_event_only.txt \
  --reference-penalty 1.0 --stricter-score-improvement 0.001 \
  > ggHgammagamma_nested.log
```

Add `--overwrite-report` for an existing report. Without `--report`, the default
name is `reports/ERL_Gini_nested_<signal-parent>_<signal-stem>_event_only.txt`.

## Policy

There is one score-improvement control: **`--stricter-score-improvement`**, which
defaults to **0**. A stricter rule T is discarded when a retained logically
looser rule L exists and:

```text
score(T) - score(L) <= stricter_score_improvement
```

This covers both tighter same-column thresholds (`pt > 30` versus `pt > 20`)
and additional AND conditions (`pt > 20 AND eta < 2` versus `pt > 20`). A finite
nonnegative threshold is required. Zero removes tied/worse stricter rules;
larger thresholds require more improvement and prune more aggressively. The
threshold is an absolute score difference, not a statistical significance.
Equality is pruned, with 1e-12 roundoff tolerance. Higher-scoring stricter rules
never suppress looser counterparts. Containment uses the actual predicates,
not coincident training selections.

`--nested-score-gap` has been removed; old commands using it fail argument
parsing. Replace the former two score controls with the single improvement
threshold. This is not an exact reproduction of the former two-stage policy:
only retained looser rules can now suppress descendants in one unified pass.

Provably equivalent conjunctions are merged independently of score improvement, preferring
the shortest original representation. Normalization removes repeated/weaker
same-column bounds and intersects category sets. Integer bounds are canonicalized
according to ordinal semantics; continuous strict/inclusive endpoints remain distinct.
Containment is conservative: it uses **same-column predicates**, not coincident
training masks, and does not infer relationships between different observables
(such as photon counts at different PT thresholds).

## Search versus final pruning

### Built-in model API

After fitting an `ExistentialRuleList`, call:

```python
selected = model.prune_candidates(
    stricter_score_improvement=0.01,
    max_reference_acceptance=0.001,  # optional: 40 kHz / 40 MHz
    similarity_filter=False,
)
audit = model.pruning_result_.decisions
normalized = model.pruning_result_.normalized_cuts
```

`selected_rules_` stores the selected rules, while `candidate_rules_` remains
unchanged with its original rule IDs. `pruning_result_` contains `retained`,
`decisions`, and `normalized_cuts`; refitting clears it and the selected list.
No scoring function is assumed: the filter compares the stored generation scores.

The model defaults to `stricter_score_improvement=None` (logical pruning off)
and `similarity_filter=True`, preserving existing Jaccard-only callers. Unlike
the model default, this event-only runner enables logical pruning with an
improvement of 0 and disables final similarity pruning. Both filters can be
enabled together in the model; Jaccard runs last and can remove looser rules.

Logical normalization, equivalence merging, and containment support event-only,
object-only, and mixed rules. Event and object columns are compared separately;
object cuts constrain the same matching slots and minimum object counts are
respected. This particular runner still uses only event inputs; use
`train_ERL.py` for object-only or combined discovery.
Rate and optional similarity filters still apply to all rule types. The acceptance
limit is a fraction in [0, 1], not a physical rate; the runner converts kHz/MHz
to this fraction (a budget at or above the entire input rate imposes no restriction).

Logical-subset pruning runs only **after** candidate generation and does not interrupt
extension of intermediate prefixes. No final Jaccard pruning is applied, because
it could discard a simpler rule when a complex rule scores higher.

Beam similarity filtering is **off by default** for the same reason. Depth 3,
beam width 100, at most 1000 pooled candidates, 32 thresholds, and minimum mixture acceptance 0.01
still limit the search: this policy cannot recover candidates never generated.
Opt in with `--beam-similarity-filter`; its two threshold flags inherit the current
shared values and affect only beam filtering, not the final logical-subset policy.
Only retained beam rules enter the pool, so depth 3 and width 100 currently
produce at most 300 candidates. Reports record the actual model settings.

## Optional trigger-rate budget

Add `--max-trigger-rate-khz 40 --input-rate-mhz 40` to enforce a per-rule
reference acceptance of at most `40 / (40 * 1000) = 0.001` (0.1%). The input
rate defaults to 40 MHz; the budget is disabled unless explicitly supplied.
Both values must be finite; the input rate must be positive and the budget
nonnegative. A zero budget keeps only rules with zero observed reference acceptance.

```bash
python casual_tests/stale/run_ERL_Gini_nested_event_only.py \
  --signal data/ggHgammagamma_mixture/anomalous.pt \
  --background data/ggHgammagamma_mixture/reference.pt \
  --report reports/ERL_ggHgammagamma_rate40_event_only.txt \
  --reference-penalty 10 --stricter-score-improvement 0.01 \
  --max-trigger-rate-khz 40 --input-rate-mhz 40 \
  --overwrite-report | tee gammagamma_rate40.log
```

Rate filtering runs **before equivalence merging and logical-subset pruning**,
but after candidate generation.
An over-budget simpler rule cannot suppress an eligible compound rule. Search
and scores are unchanged; all generated candidates remain in the report. Rules
exactly at the budget are eligible. Each record adds `estimated_reference_rate_khz`
and `passes_reference_rate_limit`; over-budget rules have pruning status
`reference_rate_limit`. The configuration records the rates, acceptance bound,
and filter order; the summary adds `rate_rules_removed`.

These are training-sample point estimates, valid as physical rates only if the
reference represents the stream at the specified input rate and target conditions.
No confidence bound is applied: independently validate rare acceptances and
their uncertainty. The limit is **per rule**, not the rate of an OR of several
rules or the incremental rate added to an existing trigger menu.

## Processing order and score comparisons

Processing order is **rate filtering -> normalization/equivalence merging ->
unified stricter-rule filtering**. Looser rules are visited before their stricter
descendants, even if the descendant's score is slightly higher. Only retained
predecessors can suppress a rule; final ranks are still score-descending. Neither
the search nor candidate scores change. An over-budget looser rule cannot act
as a witness. This is a score policy, **not** Pareto acceptance dominance: it can
discard a useful lower-rate operating point whose score improvement is insufficient.

For example, require an improvement strictly above 0.0001:

```bash
python casual_tests/stale/run_ERL_Gini_nested_event_only.py \
  --signal data/ggHgammagamma_mixture/anomalous.pt \
  --background data/ggHgammagamma_mixture/reference.pt \
  --report reports/ERL_ggHgammagamma_rate40_stricter_event_only.txt \
  --reference-penalty 1 \
  --max-trigger-rate-khz 40 --input-rate-mhz 40 \
  --stricter-score-improvement 0.0001 \
  --overwrite-report | tee gammagamma_rate40_stricter.log
```

The report records `stricter_score_improvement`, `stricter_rules_removed`, and
pruning status `stricter_score_improvement`, with witness and score improvement.
All candidates remain available for alternate operating-point comparisons.

## Reporting

All generated candidates are saved, score-ranked, with genuine `gini_gain`,
penalty, score, raw and normalized cuts, and prefix histories. Per-rule `pruning`
records explain retention/removal, the witness rule, score gap, and final retained
witness if equivalent rules form a chain. `pruned_rank` ranks retained rules.

`retained_after_final_pruning` is the final retention flag. For backward
compatibility, `retained_after_nested_pruning` aliases final retention after
rate filtering, equivalent-rule merging and unified stricter-rule pruning. The legacy
`retained_after_similarity_pruning` field mirrors it solely so the existing
plotter works unchanged. Its `--rules pruned` selects the final retained rules,
not Jaccard-pruned ones. Plot annotations show raw Gini; ranks follow penalized score.

```bash
python scripts/plot_acceptance_report.py \
  reports/ERL_ggHgammagamma_nested_event_only.txt \
  --output reports/ERL_ggHgammagamma_nested_event_only.png
```

The configuration no longer includes `nested_score_gap`, and the summary no
longer includes `nested_rules_removed`. All subset-removal decisions use
`stricter_score_improvement`; equivalent and rate removals remain separate.
Historical reports are not modified. The legacy complexity-only Python helper
is retained for compatibility but is not used by this runner. Production logical
pruning lives in `src/set_based_models/rule_list/pruning.py`; the model package
does not import any test-side helpers. Reports identify the implementation as
`ExistentialRuleList.prune_candidates`.

All acceptances and score differences describe training samples, not independent validation.
The policy can remove a tighter, lower-scoring operating point with useful
background rejection; keep the complete report for that comparison.
