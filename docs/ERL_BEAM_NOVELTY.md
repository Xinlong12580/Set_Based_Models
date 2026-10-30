# Optional novelty-aware beam selection

Core ERL accepts `beam_novelty_weight=beta`, default `0.0`. Both trainers expose
`--beam-novelty-weight` (alias `--novelty-beta`). For example, add:

```bash
--beam-novelty-weight 0.01
```

to a command using `casual_tests/train_ERL.py` or
`casual_tests/train_ERL_validation.py`. This is an illustrative value, not an
optimized physics setting. Existing commands and quick-runner settings are not
automatically changed.

## Selection rule

For candidate `R` and already selected candidates `B` in the current layer:

```text
novelty(R | B) = 1 - max Jaccard(signal_events(R), signal_events(Q)) for Q in B
selection_score(R | B) = base_score(R) + beta * novelty(R | B)
```

An empty `B` gives novelty 1 for every candidate, so the first selected candidate
has the highest base score. Each subsequent selection refreshes overlap against
the growing beam. Ties preserve the existing score-sorted candidate order.

- Positive beta rewards distinctive anomaly/mixture-event selections.
- Zero uses the original selection path, including existing similarity behavior.
- Negative finite beta is allowed and instead favors overlapping selections.
- Beta uses the same units as the base score; no score normalization is applied.
- Novelty uses unweighted signal-event Jaccard, even when the base score uses
  sample weights. Two empty masks have Jaccard 1, as in existing similarity code.
- Background and individual slot/pair identities do not enter the bonus. Two
  rules accepting the same events receive no mutual novelty, even if they use
  different object types or matching slots.

The hard beam similarity filter is independent and may be enabled at the same
time. It retains its existing signal/background, slot-mask, and priority-progress
checks. Novelty does not bypass minimum signal efficiency, finite-score checks,
or cut priorities.

## Candidate cap and final pruning

If the pooled candidates exceed `max_candidate_rules`, nonzero beta applies the
same novelty-aware selection at that cap, without an additional hard similarity
filter. Survivors are then sorted by base score for ID assignment. With no
truncation, no extra pool selection runs.

**Intrinsic scores are never replaced by novelty-adjusted scores.** Prefix
`generation_score`, statistics, report rankings, validation recalculation, and
logical/final pruning continue to use the supplied base scoring function.
Selection utility is context-dependent, not a permanent property of a rule.
The train/validation workflow still saves unpruned training candidates and
performs final pruning only after validation recalculation.

The parameter is saved in core model configuration and quantitative reports.
Older version-1 artifacts without it load with beta zero. Changing beta after
fitting does not recover discarded candidates; retrain to change exploration.

## Cost and interpretation

Masks remain packed. Selection caches overlaps and uses lazy upper bounds to
avoid revisiting candidates that cannot currently win. Novelty can nevertheless
require substantially more comparisons than score-only selection, especially
with large beams, expansion pools, or beta values. It is not a memory limit.

Distinctiveness is not statistical significance or a physics label. A large
bonus can preserve noise or reference-enriched candidates. Low Jaccard can also
describe a small subset of a broad rule, not necessarily additional coverage.
Choose beta using validation, inspect support/rate uncertainties, and reserve
an independent test sample for final performance assessment.
