# Visualize ERL rule overlaps with metric MDS

`scripts/plot_erl_mds.py` reads a core ERL model's saved `signal_jaccard_`
matrix and places rules on a 2D plane. Here, signal means the entire
anomaly-tagged mixture. No input tensors or true process labels are needed.

The standalone functions keep fitting, persistence and plotting separate.
The optimizer uses only NumPy; plotting additionally requires Matplotlib.
There are no changes to ERL training, pruning or its model format.

## Fit and save

```bash
python scripts/plot_erl_mds.py \
  training_outputs/ggHgammagamma_scan.validation.model.json
```

This creates `ggHgammagamma_scan.validation.mds.json` and
`ggHgammagamma_scan.validation.mds.png` beside the model. The JSON is a separate
layout artifact; the model is unchanged. Existing layout files are overwritten
with a warning. Figure files follow Matplotlib's normal overwrite behavior.

By default, a pruned model supplies its retained rules in retained order.
An unpruned model supplies all candidates. Use `--all-candidates` or
`--rule-ids 440 123 742` to choose a different point set. Maps with up to 40 rules
annotate original candidate IDs, not ranking numbers. Larger maps omit rule IDs
and, when clusters are supplied, label up to eight highlighted clusters as `C0`,
`C1`, etc. Every rule remains plotted.

Optional controls:

```bash
python scripts/plot_erl_mds.py MODEL.json \
  --n-init 4 --max-iter 500 --tol 1e-6 --seed 42 \
  --layout-output plots/rules.mds.json --output plots/rules.pdf
```

Use `--label-all` to annotate every rule ID, `--no-labels` to hide all
annotations, or `--title "My rule map"` to change the title.

## Replot without optimization

```bash
python scripts/plot_erl_mds.py \
  training_outputs/ggHgammagamma_scan.validation.mds.json \
  --output training_outputs/ggHgammagamma_scan.validation.mds.pdf
```

A saved layout contains everything needed for plotting; neither the original
model nor the ERL package is loaded during replotting. Its coordinates remain
unchanged. Use `--layout-output another.mds.json` to save an updated copy.

## Supply cluster indices

No clustering is performed. Every rule defaults to cluster `0`. Cluster
indices affect colors only and never enter optimization.

Prefer a JSON mapping from original rule IDs to cluster indices:

```json
{"440": 0, "123": 0, "742": 1}
```

Supply a label for every plotted rule; extra IDs in a mapping are ignored.
Alternatively, use a JSON list in the layout's exact `rule_ids` order, for
example `[0, 0, 1]`. Integer labels, including `-1` for a noise group if supplied
by a future clustering method, are supported.

```bash
python scripts/plot_erl_mds.py plots/rules.mds.json \
  --clusters clusters.json --output plots/rules_clustered.png \
  --layout-output plots/rules_clustered.mds.json
```

The standalone [ERL clusterizer](ERL_CLUSTERING.md) can generate these labels
and call this visualizer directly. It accepts a model or a saved MDS layout:

```bash
python scripts/cluster_erl_rules.py plots/rules.mds.json --distance-threshold 0.5
```

Clustering uses the original distances. Clustering the 2D coordinates would
introduce embedding distortion into the clustering decisions. Layouts produced
by the clusterizer display singleton rules as gray crosses. The eight largest
non-singleton clusters have distinct colors; smaller clusters use muted gray
circles. A bounded legend sits outside the data area. Saved cluster indices
remain distinct even where display styles are shared.

## Geometry and saved diagnostics

For the chosen rule IDs, use `D = 1 - signal_jaccard`. The optimizer is complete,
unweighted metric SMACOF: it minimizes
`sum((embedded_distance - D)**2 for i < j)` directly. No square-root transform
or positive-semidefinite correction is applied. Initialization uses centered
Gaussian coordinates, rescaled to the target distances. The lowest-stress
result from `n_init` starts is retained.

Stop when `previous_stress - stress <= tolerance * previous_stress`, or after
`max_iter` updates. Zero stress stops immediately. Convergence means the
objective has stopped improving; it does not establish an accurate embedding.
The plot uses equal axis scaling. Its axes have no physical interpretation.

The version 1 JSON format, `set_based_models_erl_mds`, stores:

- Ordered `rule_ids`, aligned `coordinates` and `cluster_indices`.
- The selected `signal_jaccard` submatrix for later distance diagnostics or clustering.
- Rule definitions, scores, mixture acceptance and reference acceptance.
- Source model path, statistics source and Jaccard provenance.
  `source_jaccard_blocks` describes the full source model, before rule selection.
- Optimization settings, best start, iteration count, convergence status,
  winning stress history and summaries of all initializations.
- Raw stress; normalized stress defined as
  `sqrt(stress / sum(D**2 for i < j))`; pairwise-distance RMSE and maximum error.

The optimizer accepts non-Euclidean distances; some distortion may persist even
at convergence. A model with `block_diagonal` Jaccard provenance produces a
warning because its cross-model similarities are assumptions. Prefer the
measured matrices obtained after recalculation on common validation events.

## Python functions

```python
from scripts.plot_erl_mds import fit_erl_mds, save_layout, load_layout, plot_mds

layout = fit_erl_mds(model, cluster_indices=None, seed=42)
save_layout(layout, "plots/rules.mds.json")

layout = load_layout("plots/rules.mds.json")
figure = plot_mds(layout, cluster_indices=clusters_by_rule_id)
figure.savefig("plots/rules.png", dpi=180)
```

`fit_erl_mds()` accepts an ERL model instance, model dictionary, or model JSON
path. Explicit selections use `rule_ids=[...]`; `all_candidates=True` overrides
the default retained selection. Cluster sequences align with the selected IDs;
mappings use original rule IDs.

The lower-level `metric_mds(distances, ...)` returns NumPy coordinates and a
diagnostic dictionary without importing ERL or Matplotlib. This separation
allows later integration into the core without adding plotting dependencies.
