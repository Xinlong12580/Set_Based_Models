# Core ERL model artifacts and inference

`ExistentialRuleList` can export and restore a versioned, human-readable JSON
model. This is separate from the runner's quantitative report: the artifact
contains executable rules, whereas the report supplies study-specific analysis.
Only NumPy and the core package are needed to load and predict. There is no
dependency on PyTorch, a runner, the data converter, or the optional EDT extension.

For a readable summary and top rules, use the external
[model printer](ERL_PRINTING.md):

```bash
python scripts/print_erl_model.py models/erl_model.json --top 10 --verbosity 1
```

## Save and restore

After fitting and selecting final candidates:

```python
from set_based_models import ExistentialRuleList

model.prune_candidates(similarity_filter=False, stricter_score_improvement=0.0)
model.save("models/erl_model.json")
restored = ExistentialRuleList.load("models/erl_model.json")
```

Saving creates parent directories and writes atomically. Existing files are
overwritten with a warning by default; use `overwrite=False` to protect them.
Invalid/nonfinite model JSON and failed writes do not replace an existing file.

`model.to_dict()` and `ExistentialRuleList.from_dict(payload)` provide the same
round trip without file I/O. Returned/loaded dictionaries do not share mutable
containers with the original model.

## Inference on new events

```python
# OR of the final selected rules; output is a Boolean vector, one value per event.
accepted = restored.predict(
    test_objects, slot_mask=test_slot_mask, event_features=test_events,
)

# A specific candidate, whether or not it survived final selection.
accepted_by_rule = restored.get_rule(0).predict(
    test_objects, slot_mask=test_slot_mask, event_features=test_events,
)

# An explicitly chosen OR of candidate rule IDs.
accepted_by_menu = restored.predict(
    test_objects, slot_mask=test_slot_mask, event_features=test_events,
    rule_ids=[0, 2],
)

# Native event-only models do not need an object tensor or dummy slots.
accepted_events = restored.predict(event_features=test_events)
```

Use the relevant example for the model's input mode. Pair tensors are ordinary
slot tensors to the core: pass the pair values and their pair-presence mask in
place of single-object inputs. Same-slot conjunctions, per-rule minimum counts,
event cuts, and automatic validity guards retain their training semantics.
Each rule is aggregated independently before applying OR, so two different
rules matching different slots cannot jointly satisfy one rule's count.

Use exactly the saved schema's column order, category codes, preprocessing, and
mask policy. The schemas are accessible as `restored.observables` and
`restored.event_observables`. Unnamed numeric arrays cannot be automatically
checked for semantic column reordering. In particular, the core does not infer
padding from PT: provide the same slot-mask convention used for training.
The number of events and slots may differ from training; the feature schemas
must still match.

Default `predict()` requires a final selection made by `prune_candidates()`.
It deliberately does not silently OR all unpruned beam candidates. For an
unpruned model, use explicit `rule_ids` or `build_union(ids)` instead. An empty
final selection, or explicit `rule_ids=[]`, accepts no events. Single-rule
prediction and `build_union()` remain available with or without final pruning.

## Evaluate held-out data

```python
performance = restored.evaluate(
    signal_objects, reference_objects,
    signal_slot_mask=signal_mask,
    background_slot_mask=reference_mask,
    signal_event_features=signal_events,
    background_event_features=reference_events,
    # Optional: rule_ids=[0, 2], signal_weight=..., background_weight=...
)
print(performance.true_positive_rate, performance.false_positive_rate)
```

For event-only evaluation omit the two positional object arrays and slot masks.
`evaluate_rule(rule_id, ...)` evaluates one candidate using the same inputs.
The term `signal` means the supplied comparison population, which can be a
background-contaminated mixture; no process labels are inferred or required.
These methods calculate new metrics and do not overwrite stored training
statistics. They do not estimate physical rates or confidence intervals.

## Recalculate candidates before validation-based pruning

`recalculate_candidates()` explicitly replaces the active candidate measurements
without retraining or optimizing any cuts:

```python
restored = ExistentialRuleList.load("models/erl_model.json")
restored.recalculate_candidates(
    validation_mixture_objects, validation_reference_objects,
    signal_slot_mask=validation_mixture_mask,
    background_slot_mask=validation_reference_mask,
    signal_event_features=validation_mixture_events,
    background_event_features=validation_reference_events,
    scoring_function=my_score,
)
restored.prune_candidates(similarity_filter=True, stricter_score_improvement=0.001)
restored.save("models/erl_validation_selected.json")
```

For event-only models omit the object arrays and slot masks. Optional
`signal_weight` and `background_weight` work as in `fit()`. The scorer defaults
to the model's current callable, so supply it here or at `load()` for a loaded
model. A scorer explicitly passed to a successful `fit()` or recalculation
becomes that model's current callable; it is still never serialized.

- Every candidate and every prefix gets new counts, weights and scores;
  final event masks and full event Jaccard matrices are regenerated for
  similarity pruning. Previously
  discarded candidates are included. No search budgets or efficiency filters
  remove candidates during this pass.
- Predicates, guards, thresholds, cut order, minimum counts, rule IDs and
  candidate order stay fixed. Final pruning ranks by the new scores rather
  than the original training order.
- The previous selected list and pruning audit/configuration are cleared.
  Call `prune_candidates()` again before default `predict()`; explicit
  `rule_ids` remain available immediately.
- Updates commit only after all candidates succeed. Invalid data, scorer
  errors or nonfinite scores leave the existing candidate/pruning state intact.
- Existing prefix statistics and `generation_score` values are **replaced**,
  not archived. Save the training model first to retain the original state.
  `statistics_source_` becomes `"recalculated"` (versus `"training"` after fit).
  This marker does not establish that the supplied data are independent.

The method returns `self`. `evaluate()` and `evaluate_rule()` remain read-only.
`train_ERL.py` does not automatically split data or invoke recalculation.
The optional [train/validation trainer](../casual_tests/ERL_VALIDATION.md) saves
unpruned training candidates, recalculates them on validation, then prunes and
saves the validation-selected model separately. Neither trainer splits inputs.
Use validation for rule selection and an untouched test sample for final evaluation.

## Shorten each rule before final pruning

`self_prune_candidates()` returns a **new** `ExistentialRuleList`. It evaluates
shortened versions of each original candidate on validation data, prunes within
each family, and pools the survivors before any final list selection:

```python
source = ExistentialRuleList.load("models/erl_training.model.json")
shortened = source.self_prune_candidates(
    signal_event_features=validation_mixture_events,
    background_event_features=validation_reference_events,
    scoring_function=my_score,
    stricter_score_improvement=0.001,
    max_reference_acceptance=0.001,
)
shortened.save("models/erl_shortened_candidates.json")

# Final comparisons across all families are a separate step.
shortened.prune_candidates(
    stricter_score_improvement=0.001,
    max_reference_acceptance=0.001,
    similarity_filter=True,
    signal_similarity_threshold=0.9,
    background_similarity_threshold=0.9,
)
shortened.save("models/erl_shortened_selected.json")
```

Object-only, pair and mixed models accept the same object arrays, slot masks,
event arrays and sample weights as `recalculate_candidates()`. The scorer
defaults to the source model's current callable; loaded models need one supplied.

- Every nonempty subset of cuts is considered, including non-prefix subsets.
  A guarded measurement has three choices: keep the measurement with its
  automatic guard, keep only the True guard, or drop both. No False guard cuts
  are introduced. Guard-only alternatives are available even when discovery's
  `allow_validity_cuts` is False. Shared guards are deduplicated through logical
  normalization; remaining measurements always enforce their guards.
- Within each source rule, optional reference-acceptance and priority requirements
  are applied before equivalence and stricter-score pruning. A stricter variant
  survives only when its score improvement over a retained looser variant exceeds
  `stricter_score_improvement`, with the same signed-gap semantics as final pruning.
  `None` disables local score pruning. No similarity pruning is applied locally.
- By default **all** source candidates are considered, even after an earlier final
  selection. Use `rule_ids=[...]` to choose specific source candidates. The original
  version is included in each family and can survive alongside several shorter
  versions. TRUE is excluded; no thresholds or new feature cuts are searched.
- Remaining cuts preserve their order and same-slot semantics. Minimum object
  counts are preserved; a variant with only event cuts is evaluated as event-only.
  `require_complete_priority=True` restricts variants to the complete source
  priority sequence, including repeated features and directions. Set it during
  both local and final pruning if completion is required.
- Equivalent pooled variants are deduplicated, retaining the shortest version
  and then the highest score on depth ties, as in existing equivalence pruning.
  Different source priorities and
  priority progress stay distinct. IDs are reassigned consecutively. Original
  source configurations/origins are retained through `merge_provenance_`, and
  `self_pruning_provenance_` records parent candidate IDs, local settings and
  variant counts. Both are saved and restored by the core model format.
- Every retained prefix and rule has fresh validation counts, weighted statistics,
  scores and masks. `statistics_source_` is `"recalculated"`. The returned model
  has no final selection or final pruning audit; call `prune_candidates()` before
  default prediction. The source model, its statistics and selection stay unchanged.

Subset generation is exponential: up to `2**depth` choices for ordinary cuts,
or `3**depth` when every cut has a guard. It is intended for short conjunctions.
Only one rule family is expanded at a time. Full Jaccard matrices are calculated
after local pruning and pooled deduplication, rather than for all expansions.
Set `compute_jaccard=False` to omit their quadratic storage; final similarity
pruning can still use the returned event masks in memory. Call
`compute_jaccard_matrices()` before saving if mask-free similarity repruning is
needed after loading.

## Saved event Jaccard matrices

`fit()` and `recalculate_candidates()` automatically compute two full matrices:

```python
mixture_jaccard = model.signal_jaccard_
reference_jaccard = model.background_jaccard_
overlap = mixture_jaccard[rule_id_a, rule_id_b]
```

Both have shape `(candidate_count, candidate_count)`, include every candidate,
and use candidate rule IDs for both axes. They compare **event** selections,
after applying object validity, same-slot conjunctions, minimum counts, and event
cuts. Object-slot matrices are not stored. Similarity is unweighted, matching
existing final pruning even when scores/statistics use sample weights. The
diagonal is 1; two empty selections also have similarity 1.

The arrays are read-only float64 values. Bit packing and bounded row batches
limit temporary memory during calculation; the stored matrices still require
quadratic space. Two matrices use `16 * candidate_count**2` bytes before JSON
overhead. `compute_jaccard_matrices()` explicitly rebuilds them from current
event masks and returns `(signal_matrix, background_matrix)`.

`save()` embeds both matrices directly in the model JSON as `event_jaccard`.
Loading restores them without restoring masks. Final similarity pruning uses
these cached values automatically, so it can run on the saved statistics:

```python
restored = ExistentialRuleList.load("models/erl_model.json")
restored.prune_candidates(similarity_filter=True,
                          signal_similarity_threshold=0.9,
                          background_similarity_threshold=0.9)
```

The cache describes the same input rows as the recorded statistics. Prediction
or read-only evaluation on another sample does not update it. Call
`recalculate_candidates()` to replace both statistics and similarities when
selecting on validation or another dataset. Older files without matrices still
load; recalculation supplies their missing cache.

## Pool candidates from multiple searches

Save the unpruned training model from each hyperparameter configuration, then
load their full candidate pools together:

```python
combined = ExistentialRuleList.load_many([
    "models/run_a.training.model.json",
    "models/run_b.training.model.json",
], scoring_function=my_validation_score)

# Event-only example; object/pair/mixed inputs follow recalculate_candidates().
combined.recalculate_candidates(
    signal_event_features=validation_mixture_events,
    background_event_features=validation_reference_events,
)
combined.prune_candidates(
    similarity_filter=True,
    stricter_score_improvement=0.001,
    max_reference_acceptance=0.001,
    require_complete_priority=True,
)
combined.save("models/combined.validation.model.json")
```

For models already in memory, use
`ExistentialRuleList.merge([model_a, model_b], scoring_function=my_score)`.
Optional `source_names=["run_a", "run_b"]` labels these sources; `load_many()`
uses their filenames. Both methods return a normal `ExistentialRuleList`.

- **All candidates are pooled**, including those discarded by a source's final
  selection. There is no new candidate cap, score comparison, or deduplication
  during merging. The sources stay unchanged.
- Ordered object and event schemas must match exactly: names, kinds, categories,
  and validity guards. Different search settings and per-rule minimum counts
  are supported. Different feature projections, or single-object versus pair
  schemas, require separate models. Use the same preprocessing and slot meaning;
  schema equality cannot check how the original arrays were constructed.
- Rules receive consecutive IDs in input-model/candidate order. Cuts, prefix
  histories, and minimum counts are preserved. `merge_provenance_` records each
  original source configuration and each rule's original source index/ID.
  Repeated merges flatten origins back to the original searches.
- Selections, pruning records, and event masks are cleared. Existing source
  statistics/scores remain readable, but `statistics_source_` becomes `"merged"`.
  Every pruning call requires successful recalculation on common data first.
  Supply one common scorer at merge/load time or during recalculation; source
  scoring callables are not inherited. Explicit-rule prediction remains available.
- Each Jaccard matrix becomes block diagonal: input models' matrices occupy the
  diagonal blocks, and entries between input models are **assumed zero**, even
  for identical cuts. `jaccard_source_` is `"block_diagonal"`, and
  `jaccard_blocks_` gives block sizes. These cross-model zeros are assumptions,
  not measured disjointness. Nested merges preserve existing blocks. If a source
  has masks but no cache, its block is calculated without changing that source.
  If an older source has neither matrices nor masks, the merged cache remains
  unavailable until recalculation.
- Validation recalculation replaces the block-diagonal approximation with full
  measured overlaps across the pooled candidates. Its `jaccard_source_` becomes
  `"event_masks"`, and `jaccard_blocks_` becomes one full block. The existing
  requirement to recalculate merged statistics before pruning remains unchanged.
- The combined model has no global object priority. With
  `require_complete_priority=True`, each candidate is checked against its own
  source priority, including repeated names and directions. Final pruning then
  compares eligible candidates across sources using their refreshed statistics.
- Source search budgets and novelty weights are provenance. They impose no new
  restrictions during validation selection. Refitting the combined object starts
  a fresh search with its current constructor settings and clears merge provenance.

Saving/loading retains provenance and the requirement to recalculate an untouched
merged pool. After validation selection, saving and inference work normally.
Validation is used for selection; an independent test sample is still needed for
final performance. A per-rule acceptance limit is not a limit on their combined OR.

## Format, version 1

The root `format` is `set_based_models_erl`, and `schema_version` is `1`.

| Field | Content |
| --- | --- |
| `observables`, `event_observables` | Ordered names, feature kinds, category code/label records, and validity-column associations. |
| `configuration` | Core constructor settings, excluding the scoring callable; repeated/directed object priorities and `beam_novelty_weight` are preserved. Older version-1 artifacts missing the novelty weight load with zero. |
| `input_mode` | Fitted `event_only`, `object_only`, or `combined` mode; `null` for an unfitted model. |
| `statistics_source` | `training`, `recalculated`, `merged` (requires recalculation before pruning), or `null`; older version-1 artifacts without this optional field default to `training` for fitted models. |
| `rules` | All candidate rules in ID order, not just retained rules; each stores its ID, minimum count, ordered steps, prefix scores, and complete count/weight statistics. |
| `pruning` | `null` before final selection; otherwise the selection configuration, ordered selected-rule IDs, pruning decisions, and normalized cuts. |
| `merge_provenance` | Optional source records and rule origins, aligned with candidate IDs. Source records contain `name`, `configuration`, `input_mode`, and `statistics_source`; each origin stores `source_index` and the original `rule_id`. Absent in ordinary models and older artifacts. |
| `self_pruning_provenance` | Optional local-shortening settings, source candidate IDs, variant counts and per-output `parent_rule_ids`; created by `self_prune_candidates()`. Refitting clears it. |
| `event_jaccard` | Optional event-level similarity cache: `metric`, `scope`, `source`, `statistics_source`, `rule_ids`, `blocks`, and square `signal`/`background` matrices. Source is `event_masks` or `block_diagonal`; block-diagonal cross-block values are assumed zero. Older artifacts may omit this field. |

Cuts declare `scope` (`slot` or `event`) and an observable name within that
scope. Threshold cuts store the exact comparison and threshold. Category cuts
store accepted integer codes. Both scopes can contain a feature with the same
name. JSON preserves the original floating-point thresholds without display
rounding. Candidate IDs remain zero-based and correspond to positions in
`candidate_rules_`.

The loader rejects different formats or unsupported schema versions rather
than interpreting a runner report as a model. It reconstructs declarative core
objects only: it never imports a scorer or executes Python from the file.

## Intentionally omitted state

- No input tensors, training event/slot masks, or intermediate beam states.
- No scoring callable or automatic reconstruction of custom scorer parameters.
- No HEP-specific conversion layout, process labels, tensor input paths, or rate units.
  Merged artifacts may record source model filenames as provenance labels.

The fitted scores and statistics remain available, but loaded rule
`signal_selection` and `background_selection` are `None`. Recompute selections
on the appropriate arrays with `predict()` when needed. Logical, rate-fraction,
and priority-completion pruning use saved rules/statistics; final similarity
pruning can use the saved Jaccard matrices. Older mask-free files without a cache
need recalculation before similarity pruning. To run pruning on new data, call
`recalculate_candidates()` first. Saving still omits the regenerated event masks
while retaining their pairwise similarities. Pairwise matrices do not determine
the combined OR acceptance or higher-order intersections. A model artifact is
not a checkpoint for continuing an interrupted beam search.

To fit again, supply a scoring callable explicitly, either when loading:

```python
restored = ExistentialRuleList.load("models/erl_model.json", scoring_function=my_score)
```

or on the next `fit(..., scoring_function=my_score)`. Refitting resets the
previous final selection and pruning configuration as usual.

`casual_tests/train_ERL.py` saves unpruned training candidates through the core
model's `save()`. By default it writes `.training.model.json` and
`.training.report.json` files under
`training_outputs/`; `--output-prefix PATH` names both. The report remains the
separate quantitative format, with study configuration and training diagnostics.
`--model FILE` and `--training-report FILE` override individual paths; `--report`
is a compatibility alias. Existing training outputs require `--overwrite-output`.
See [the trainer guide](../casual_tests/ERL_GINI_INPUT_MODES.md#training-outputs).
`train_ERL_validation.py` calls this trainer once, then `validate_ERL.py` to
recalculate and prune using an independent validation score. Validation-selected
models and reports use `.validation.model.json` and `.validation.report.json`.
