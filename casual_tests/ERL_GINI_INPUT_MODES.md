# Flexible event/object Gini ERL runner

`run_ERL_Gini_nested.py` loads object and event tensors by default and supports
native event-only or object-only operation. It can replace the single-object
tensor with a separate pair tensor and directly uses the core ERL pruning API.

## Commands

Run from the repository root in the `hep` environment. Both input files must
have compatible schemas for the enabled features.

```bash
# Both tensors (default).
python casual_tests/run_ERL_Gini_nested.py \
  --signal data/ggHgammagamma_mixture/anomalous.pt \
  --background data/ggHgammagamma_mixture/reference.pt \
  --report reports/ERL_ggHgammagamma_combined.txt \
  --reference-penalty 10 \
  --stricter-score-improvement 0.01 \
  --max-trigger-rate-khz 40 --input-rate-mhz 40
```

For event-only input, add `--no-object-features` (alias `--event-only`) and
choose a distinct report, such as `reports/ERL_ggHgammagamma_event_only.txt`.
For object-only input, add `--no-event-features` (alias `--object-only`) and
choose `reports/ERL_ggHgammagamma_object_only.txt`. Disabling both is an error.
Add **`--basic-object-features`** to use only `pt`, `eta`, `phi`, and `type_id`
from the object tensor. This works in combined or object-only mode and leaves
event inputs unchanged. It cannot be combined with `--event-only`. The default
still uses every object column. For example:

```bash
python casual_tests/run_ERL_Gini_nested.py \
  --signal data/ggHgammagamma_mixture/anomalous.pt \
  --background data/ggHgammagamma_mixture/reference.pt \
  --report reports/ERL_ggHgammagamma_basic_objects.txt \
  --object-only --basic-object-features \
  --reference-penalty 0 --stricter-score-improvement 0.01 \
  --max-trigger-rate-khz 40 --input-rate-mhz 40
```

Columns are selected by name and reordered to `pt, eta, phi, type_id`; their
saved schema/category mapping is retained. All four columns must exist. Extra
object values do not enter validation or search in basic mode, though input
schema metadata must still be well formed. Input PT files are not modified.

### Compulsory order for object cuts

Add `--object-cut-priority type_id pt` to either command above to require
the first object cut to be on `type_id` and the next on `pt`. Once every list
position has been consumed, all enabled object features become available again,
including refinement of previously used features. The category, comparison
direction, and threshold are still optimized by the scorer.

This is a restriction on the **next object cut**, not a completion requirement:
short rules such as `EXISTS type_id in {photon}` remain eligible candidates
and can rank first. Event cuts remain unrestricted, can appear before/between
object cuts in combined mode, and do not advance the object priority. All
explicit cuts still consume depth. An empty/default priority leaves search
unchanged. `--event-only` cannot be combined with an object priority.

Repeated names require another cut at that position: for example,
`--object-cut-priority type_id pt pt` requires two successive momentum cuts
after the type cut. Their directions remain optimized, so this permits but
does not guarantee a two-sided interval.

To require opposite directions, use:

```bash
--object-cut-priority type_id 'pt:>' 'pt:<='
```

Quote entries containing comparison operators to avoid shell redirection.
Numerical columns support `>`, `>=`, `<`, and `<=`; their thresholds remain
optimized. Python accepts the same strings or `(feature, direction)` pairs.
Reports retain the directed entries and track each position separately.
Incomplete prefixes remain eligible, so a returned rule need not complete the
interval. The two bounds still consume two search steps; no joint interval
optimization or exemption from beam/similarity/logical/rate pruning is added.

`--require-complete-priority` makes completing every entry compulsory for
**final retention only**. It filters incomplete candidates before rate,
equivalence, logical, and final similarity pruning, so an incomplete rule
cannot suppress a completed one. Search still keeps and extends incomplete
prefixes, and the report keeps them with status `incomplete_object_priority`.
Completion uses original ordered cuts, counting repeated entries and respecting
directions; event cuts and automatic validity guards do not consume entries.
The default is off. An empty list is already complete. If no complete candidate
survives search, the retained list is empty. The core equivalent is
`model.prune_candidates(require_complete_priority=True, ...)`.

Names must be present among the enabled object columns. For
example, `charge` cannot be prioritized with `--basic-object-features`.
Guard-only validity flags cannot be prioritized unless standalone flag cuts
are explicitly enabled. If the next required feature has no usable cut, no
other object feature is substituted; the prefix can still be returned or
extended with event cuts. Normal acceptance, beam, score, cap, and pruning
filters still apply; priority does not guarantee retention of every type.

The core option is `ExistentialRuleList(..., object_cut_priority=["type_id", "pt"])`.
Priority progress participates in beam-state deduplication and similarity:
equal masks at different stages can require different next object cuts.
Apart from optional completion eligibility, final pruning policies remain
independent of the expansion order. Reports store
`configuration.object_cut_priority` and `object_cut_priority_remaining` for
each rule, alongside the existing scoped prefix history.

The report also stores `configuration.require_complete_priority`, per-rule
`object_cut_priority_complete`, and `summary.incomplete_priority_rules_removed`.

The example penalty and score-improvement values are explicit choices, not
defaults or universally optimized values.

Without `--report`, the output name is
`reports/ERL_Gini_nested_flexible_<sample-parent>_<signal-stem>_<mode>.txt`.
Modes are `combined`, `event_only`, and `object_only`. Existing reports are
protected unless `--overwrite-report` is supplied. Reports cannot overwrite
input tensors or their hard-link aliases, and report symlinks are rejected.
Basic-object mode adds `_basic_objects` before the mode suffix in the default
report filename. Reports/logs identify the selected object features.

### Pair inputs

Use `--pair-tensor` (alias `--use-pair-tensor`) after converting both classes with
`--with-pair-tensor`. It replaces `obj` with `pair_obj`; it does not concatenate
single-object and pair tensors. The runner reads `pair_slot_mask`,
`pair_obj_columns`, and `pair_obj_feature_schema`. Event inputs remain unchanged.

Pair priorities use member names, for example:

```bash
--pair-tensor --object-only \
  --object-cut-priority type_id_1 type_id_2 'min_pt:>' \
  --require-complete-priority
```

All object conditions then refer to the same pair of distinct particles.
Different matching pairs can share a member. Basic mode selects `pt`, `eta`,
`phi`, and `type_id` for each member, together with required coordinate guards;
derived pair features are excluded. See the
[pair-tensor guide](../raw_event_conversion/PAIRS.md) for feature definitions.
The default report name identifies pair mode when enabled.

## Loading and feature meanings

- `label [N]` is required for diagnostic process counts, but never determines
  the training class. Every signal-input row is the mixture class.
- Enabled objects require `obj [N,S,P]`, `obj_columns`, and, for expanded
  columns, `obj_feature_schema`. Column kinds and category labels come from
  the saved schema, so the current `[N,22,20]` raw-converter tensor is supported
  without hardcoded slot caps or a fixed feature count. The known legacy
  `[pt, eta, phi, type_id]` layout also works without a schema, using its old
  jet/muon/electron/photon/MET type mapping.
- The named `pt` column identifies real slots with PT > 0; padded slots are
  excluded. Present object values are validated rather than silently
  replacing nonfinite values with zero. Signed ordinal charge is supported.
- Enabled events require `event_features [N,F]`, `event_feature_columns`, and
  optionally `event_feature_schema`. The saved schema supplies kinds and
  definitions, including continuous, Boolean, and categorical columns. Without
  a schema, the legacy nonnegative ordinal multiplicity convention is used.
- Disabled tensors and their metadata may be absent, malformed, or different
  between the two files; they are not validated or supplied to ERL.
- Enabled tensors must align with label rows. Signal/reference object shapes,
  feature order, kinds, and saved definitions must match. Sample sizes may differ.
- `pf` and `native_event_info` are not inputs. No dummy object is constructed
  for event-only fitting. **Schema-declared validity columns automatically
  guard their associated measurements**, in training and prediction.

### Automatic validity guards

No new CLI flag or tensor conversion is required for the existing raw-converter
files. Object schemas declare `validity_column`; event schemas declare
`definition.validity_feature`. Both associations are read automatically.
For example, a cut on `n_neutrals <= 1` means
`n_neutrals_valid == True AND n_neutrals <= 1` on the same object.
An angle/ratio event cut similarly requires its event validity flag.

Invalid placeholders are excluded from threshold generation and never satisfy
the associated cut, regardless of their numeric fill value. Measured zeros
with true validity remain valid. Multiplicities, PT sums and ranked PT retain
their meaningful zero semantics unless explicitly guarded in the schema.

Guards add **no search depth**. Reports include `validity_guard` per cut,
guarded text and feature-to-flag maps. Logical pruning includes the implicit
conditions in containment/equivalence proofs. Schema-linked validity flags are
**guard-only by default**: neither standalone True nor standalone False cuts
are generated on them. The columns remain in the inputs for automatic guarding.
Ordinary Boolean features and meaningful multiplicity/absence cuts remain
available. `--allow-validity-cuts` explicitly restores standalone validity cuts
for detector/missingness studies, without disabling the automatic guards.
The core option is `ExistentialRuleList(..., allow_validity_cuts=True)`.
Reports save the policy and logs identify it. Schemas without associations retain their
old behavior: no flags are guessed from zeros or column-name suffixes.

Current single-object eta/phi have no dedicated validity flags; their conversion-
time missing-value fills cannot be recovered by these guards. Pair-member
coordinates have dedicated guards. Basic mode preserves these guards when
selecting pair inputs.

Within a rule, object cuts must hold on the **same object**. A muon type cut
AND an electron type cut does not mean one of each. Use event multiplicities
for independent mixed-type object requirements, or use separate member type
conditions in pair mode.

## Score, search, and pruning

Current flexible-runner settings are:

| Setting | Default |
| --- | --- |
| Score | Full-sample weighted Gini gain minus reference penalty times reference acceptance |
| Reference penalty | 1.0; `--reference-penalty 0` gives pure Gini |
| Rule depth / beam width / candidate cap | 4 / 40 / 300 |
| Numeric threshold cap | 32 per feature/prefix |
| Minimum mixture acceptance | 0.01 |
| Matching object count | 1 |
| Object-cut priority | None; enable with `--object-cut-priority type_id pt` |
| Complete priority required | Off; opt in with `--require-complete-priority` |
| Object representation | Single objects; opt into `--pair-tensor` |
| Progress display | Off; opt in with `--progress` |
| Standalone validity flag cuts | Off; opt in with `--allow-validity-cuts` |
| Beam similarity | Off; opt in with `--beam-similarity-filter` |
| Final similarity | Off; opt in with `--final-similarity-filter` |
| Mixture/reference Jaccard thresholds | Current shared defaults, both 0.1 |
| Required stricter-rule score improvement | 0; adjustable with `--stricter-score-improvement` |
| Per-rule rate budget | None; optional `--max-trigger-rate-khz` |
| Assumed input rate | 40 MHz; adjustable with `--input-rate-mhz` |

The reference penalty acts during beam search. Negative finite scores remain
eligible. Search size/depth/caps remain constructor settings in the script,
not CLI flags. Neither penalty nor pruning uses stored process labels.

The runner directly calls `ExistentialRuleList.prune_candidates`. The order is:

1. Optional priority-completion eligibility.
2. Optional per-rule reference-acceptance eligibility.
3. Scope-aware equivalence merging and stricter-score-improvement pruning for
   event-only, object-only, and mixed rules.
4. Optional final Jaccard pruning for all rule types.

Event and object columns are compared separately, even when their indices
coincide. All object conditions must hold on the **same matching slots**.
Minimum counts are respected: two matching objects imply one matching object,
but a tighter single-object threshold does not imply two objects. Event-only
rules ignore minimum counts, as they do during prediction. The report records
`logical_pruning_applicable: true` for all supported rule types.

The stricter score difference must exceed the supplied threshold;
equality is removed with `1e-12` tolerance. Tighter same-column thresholds and
extra AND conditions are covered in either scope. For example, `EXISTS pt > 50`
is pruned against `EXISTS pt > 40` unless its score improves enough; adding
`AND type_id in {photon}` is also a logical restriction. Relations across different
feature columns are not inferred. `--nested-score-gap` is not supported.

Negative values allow bounded score loss: `--stricter-score-improvement -0.008`
keeps a stricter rule at this stage if its score drops by less than 0.008
relative to every applicable retained looser rule. Equivalent-rule merging,
rate filtering, similarity filtering, and final score ranking are unchanged.

Similarity thresholds use `--signal-similarity-threshold` and
`--background-similarity-threshold`; they apply to either enabled similarity
stage. Higher values prune less aggressively. Final similarity can remove
either looser alternatives or stricter refinements preserved by logical pruning,
so it is opt-in. Beam similarity also checks surviving slot selections and
priority progress. Final similarity checks event selections only.

The rate fraction is `budget_kHz / (1000 * input_MHz)`: 40 kHz at 40 MHz gives
0.001 reference acceptance. It is a post-search, **per-rule training estimate**,
not a confidence bound or a budget for the OR of retained paths. Validate
on independent representative reference events; zero observed reference
acceptance does not establish a zero rate.

### Progress and memory

`--progress` writes search activity to stderr: depth, processed parents and cuts,
beam filtering, evaluated candidates and elapsed time. Redirect `2>&1` to
capture it with stdout. It does not cover final pruning or report writing and
does not promise an end-to-end ETA.

ERL automatically packs internal Boolean selections into bits, unpacks only
the parent currently being extended, and drops slot selections no longer
needed after beam filtering. Public rules expose ordinary Boolean event masks.
Mask payload is roughly eight times smaller; total memory savings depend on
input arrays and other state. There is no disk spilling or hard memory cap.

## Reports and plots

The `.txt` report contains JSON and preserves the existing plotting format.
Object/event scopes are explicit in the prefix history and cut records.
It stores raw/simplified rules, penalized scores, raw Gini, penalty terms,
counts/acceptances, rate estimates, ranks, and pruning reasons/witness IDs.
All capped candidate rules are saved, including discarded ones.

`feature_mode` and enabled-feature flags identify active inputs. All rule types
have scope-aware `normalized_cuts`, alongside untouched original steps, and a
per-rule `minimum_count`. Legacy nested/similarity retention
fields alias final retention for plotting; inspect `pruning.status` to learn
which filter actually removed a rule.

```bash
python scripts/plot_acceptance_report.py \
  reports/ERL_ggHgammagamma_combined.txt \
  --output reports/ERL_ggHgammagamma_combined.png
```

All reported acceptances use the training inputs. Reconvert both classes and
remake mixtures to include new features; old PT files are not auto-updated.

Synthetic verification (no production inputs):

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover \
  -s tests -p test_flexible_erl_runner.py -v
```
