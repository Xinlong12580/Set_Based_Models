# Quantitative report format

## Scope and loading

Quantitative EDT and ERL runners save JSON, commonly using a `.txt` extension:

```python
import json

with open("reports/my_run.txt", encoding="utf-8") as handle:
    report = json.load(handle)
```

The current envelope identifies itself as:

```json
{
  "format": "set_based_models_quantitative_report",
  "schema_version": 1
}
```

The [JSON Schema](../schemas/quantitative_report.schema.json) describes current
version-1 structures, including older quantitative runners and the flexible
ERL runner. A shared version does **not** imply identical optional fields or
algorithms. Unknown additional properties are allowed for additive extensions.
An empty rule list is valid, including when completion or rate filtering leaves
no retained rules. A report with an unsupported version needs an explicit adapter.

## Top-level fields

| Field | Meaning |
| --- | --- |
| `format`, `schema_version` | Report-family identifier and structure version, not a Git/software version |
| `created_utc` | Report creation timestamp in UTC |
| `model` | Writer/model variant; inspect configuration as well |
| `class_semantics` | Input-source class definitions and purity warning |
| `inputs.signal`, `inputs.background` | Mixture/reference descriptions: paths, counts, shapes, stored-label counts |
| `object_observables`, `event_observables` | Ordered runtime feature names, kinds, categories, and optional guards |
| `configuration` | Settings actually recorded for this run |
| `summary` | Candidate/pruning counts or tree/default-selection statistics |
| `results.rules` | ERL candidates in full-pool score order |
| `results.leaves` | EDT leaves in descending training mixture fraction |

Exactly one result family is expected. Optional fields include `feature_mode`,
`object_tensor_key`, saved feature schemas, validity policy, native/dummy input
notes, and an evaluation note. Feature schemas can contain units, sources and
formulas beyond the runtime observable definitions. A path can be local and
inaccessible to a remote integration; it is not a downloadable dataset URL.

Future extensions may add a `provenance` block with code revision, worktree
state, input identifiers/checksums and generation settings. Current writers
do not populate these automatically. Likewise, `example_metadata` in the
bundled fixtures labels their synthetic status; it is not a runner output field.

## Shared selection metrics

Each rule or leaf has a `metrics` object. Runners currently use unit weights;
these fields are raw-count statistics.

| Field | Definition |
| --- | --- |
| `signal_accepted`, `signal_total` | Accepted/all anomalous-mixture input events |
| `background_accepted`, `background_total` | Accepted/all reference input events |
| `signal_acceptance` | `signal_accepted / signal_total` |
| `background_acceptance` | `background_accepted / background_total` |
| `acceptance_difference` | Mixture acceptance minus reference acceptance |
| `acceptance_ratio` | Mixture/reference acceptance; JSON `null` if reference acceptance is zero |
| `mixture_label_purity` | Accepted mixture events / accepted events from both inputs |
| `direction` | `mixture-enriched`, `reference-enriched`, or `equal-acceptance`, comparing acceptances |
| `selected_signal_stored_label_counts` | Diagnostic process-label counts in the accepted mixture subset |
| `selected_background_stored_label_counts` | Diagnostic process-label counts in the accepted reference subset |

Empty selections have `mixture_label_purity = 0`. Labels are stringified numeric
keys in JSON; their physical meaning requires dataset information.
`mixture_label_purity` is **training-class purity, not genuine-process purity**.
Custom weighted studies must distinguish these count metrics from weighted
scores/acceptances; the unit-weight runner makes them coincide.

## ERL rule fields and ranking

| Field | Meaning |
| --- | --- |
| `rule_id` | Stable ID within this candidate pool; not comparable across runs |
| `rank` | One-based rank in the full capped pool; printed as `global_rank` |
| `pruned_rank` | One-based score rank among final retained rules; `null` when removed |
| `depth` | Number of original explicit cuts; automatic guards do not add depth |
| `effective_depth_after_simplification` | Number of simplified explicit predicates, not proof of priority completion |
| `raw_rule`, `simplified_rule` | Human-readable original/simplified conditions |
| `steps` | Original ordered prefix history, with counts and scores at each step |
| `normalized_cuts` | Optional scope-aware simplified predicates |
| `minimum_count` | Optional matching-slot requirement; event-only rules ignore it |
| `score` | Optional explicit optimization score; present in the flexible penalized runner |
| `gini_gain` | Whole-rule gain versus its complement on original input totals |
| `reference_penalty_term` | Optional penalty coefficient times reference acceptance |
| `direction` | Same enrichment direction as `metrics.direction` |
| `retained_after_final_pruning` | Authoritative final-retention flag when present |
| `pruning` | Optional reason, witness and score comparison audit |

Older pure-Gini reports lack `score`; their optimization score is `gini_gain`.
The flexible runner uses `score = gini_gain - reference_penalty_term`.
Negative scores are legitimate. Its `retained_after_nested_pruning` and
`retained_after_similarity_pruning` are compatibility aliases for **final**
retention, not independent stage-by-stage survival records. Older reports may
only provide `retained_after_similarity_pruning`; interpret this in conjunction
with their recorded policy. Do not infer a logical-pruning decision from that
legacy field alone.

The console's section number is the position within a displayed subsection;
it is not necessarily `rank` or `pruned_rank`. Rank by score is not rank by
genuine-signal purity, rate, or physics interpretability. The plotting script
filters the selected subset and preserves report order.

### Cut records and prefix history

Cut records include `scope` (`slot` or `event`), `observable`,
`observable_kind`, `cut_type`, `raw_text`, and `display_text`:

- Threshold cuts add `comparison` (`>`, `>=`, `<`, `<=`) and `threshold`.
- Category/Boolean cuts add integer `accepted_codes`; resolve names through
  the appropriate observable's `categories` mapping.
- Guarded cuts add `validity_guard`, giving its scope, observable and required
  true value. Guards constrain the same slot or event as the measurement.

For integer-valued counts, raw `n_photons > 1.5` displays as `n_photons >= 2`.
Use the raw comparison and threshold for faithful numeric reconstruction.
Within one ERL rule, every slot predicate applies to the same object or pair.
Do not evaluate different slot conditions on independently chosen objects.

Each ERL step records its depth, scope, cut, display, prefix accepted counts,
acceptances and `prefix_gini_gain`. Penalized runners additionally record
`prefix_score` and `prefix_reference_penalty_term`. These are whole-prefix
statistics, **not incremental gains** for the last added cut.

### Priority, pruning and rate fields

`configuration.object_cut_priority` contains ordered strings, including
repetitions and optional directions, for example `type_id`, `pt:>`, `pt:<=`.
Per-rule `object_cut_priority_remaining` and `object_cut_priority_complete`
describe progress. Completion uses original ordered cuts. Simplification may
merge cuts that consumed separate priority positions.

The flexible runner's final order is optional priority completion, optional
reference-rate eligibility, equivalence merging, stricter-score filtering,
then optional final similarity. Search-time similarity is separate. Older
variants may use different policies; read their configuration.

| `pruning.status` | Meaning |
| --- | --- |
| `retained` | Kept after configured final filters |
| `incomplete_object_priority` | Requested cut sequence not completed; no pruning witness |
| `reference_rate_limit` | Reference acceptance exceeds the per-rule ceiling; no witness |
| `equivalent_rule` | Logically equivalent representative kept instead |
| `stricter_score_improvement` | A strict subset fails the required improvement against a looser rule |
| `similarity` | Accepted-event overlap exceeds both configured class thresholds |

`pruned_by_rule_id` identifies the direct witness where applicable;
`retained_witness_rule_id`, when present, follows witness chains to the final
survivor. `score_gap` is witness score minus removed-rule score. Logical
`score_improvement` is the opposite sign. Null means no comparison was recorded.
The threshold test is stricter-minus-looser score **greater than** the configured
signed threshold, with `1e-12` roundoff tolerance; equality is removed.

`estimated_reference_rate_khz = background_acceptance * input_rate_mhz * 1000`.
`passes_reference_rate_limit` is true when no ceiling is enabled or the rule
fits it. A requested ceiling in kHz becomes an acceptance fraction by dividing
by `1000 * input_rate_mhz`. These are training estimates per rule, not confidence
bounds or an OR-menu rate.

The report saves the capped candidate pool, including final-pruned rules.
It does not save every rejected expansion or candidate lost to the beam/cap.

## EDT leaves and paths

EDT `rank` orders leaves by `leaf_signal_fraction`, the training mixture
fraction in that leaf. `leaf_id` is a tree node ID, not an ERL rule ID.
`predicted_class_at_default_threshold` compares the leaf fraction against
`configuration.decision_threshold`. It need not agree with enrichment direction
when input sample sizes differ. `summary.default_prediction` describes the
union of default predicted-positive leaves, not just the highest-ranked leaf.

`path` contains ordered tree decisions: `source_node_id`, `branch`
(`positive` or `negative`), `scope`, `cut`, `selected_condition`, and
`source_split_gain`. The raw cut is the node's positive test; a negative branch
**negates that test**. Use `selected_condition` to explain the actual path.
A guarded negative branch is `NOT (guard AND measurement)`, not simply the
opposite numerical bound, and can accept invalid measurements.
Object-level tree branches additionally depend on their inherited surviving-slot
state; do not reinterpret their cut records as independent existential tests.

EDT split gains are local node gains, not ERL whole-rule scores. Leaves within
one tree are mutually exclusive and partition its input events, so leaf counts
can be added for an OR of leaves. ERL candidates may overlap; their marginal
counts cannot determine an OR acceptance.

The current event-only EDT study runner does not attach automatic validity
links when loading event features. Saved schema definitions alone do not prove
that a guard was active; check runtime observables and validity-policy fields.

## Checks and evidence limits

- Accepted counts must not exceed totals; recompute fractions from counts.
- Selected label counts must sum to the corresponding accepted count.
- Compare rule totals against input counts; check IDs, witness references and ranks.
- Prefix accepted counts cannot increase along an ERL AND chain.
- Check score/penalty and rate arithmetic when the needed settings are recorded.
- For EDT, leaf counts sum to input totals and default-selection counts sum over
  the predicted-positive leaves. Split gain is not determined by a leaf alone.
- Validate schema links, feature units, and same-slot/pair meanings before making
  claims about physics. Report source/version limitations explicitly.
- Absence of a rule, overlap information, validation statistics or provenance
  is not evidence that the missing quantity has a particular value.

The synthetic fixtures have arithmetic checks in
`tests/test_report_documents.py`. No runtime dependency or stricter runner
input validation is introduced by this schema.
