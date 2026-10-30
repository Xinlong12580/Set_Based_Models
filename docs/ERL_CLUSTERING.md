# Cluster ERL rules by their event selections

`scripts/cluster_erl_rules.py` reads a core ERL model, clusters its rules using
the saved **mixture-event Jaccard matrix**, and calls `plot_erl_mds.py` to draw
the labeled 2D map. It needs NumPy and Matplotlib, with no SciPy dependency.
It does not change the model, remove rules or use process labels.

## Run from a model

```bash
python scripts/cluster_erl_rules.py \
  training_outputs/ggHgammagamma_scan.validation.model.json \
  --distance-threshold 0.5 --max-iter 2000
```

The default rule selection matches the visualizer: retained rules after
pruning, or all candidates for an unpruned model. Override it with
`--all-candidates` or `--rule-ids 440 123 742`.

The script fits the MDS layout, calculates clusters from the original matrix
and draws the cluster labels. MDS coordinates never enter clustering.
The default distance threshold is an illustrative choice, not an optimized
physics setting. A smaller threshold makes grouping stricter.

## Reuse existing coordinates

```bash
python scripts/cluster_erl_rules.py \
  training_outputs/ggHgammagamma_scan.validation.mds.json \
  --distance-threshold 0.5
```

A saved layout already contains the chosen Jaccard submatrix and rule
metadata. This route only clusters and plots; it neither loads the original
model nor repeats MDS. You can change the threshold while keeping exactly
the same coordinates. Rule selection on a layout is fixed.

Use `--output-prefix plots/rules` to choose an output prefix, and
`--output plots/rules.pdf` to override just the figure path. Labels are automatic:
maps with up to 40 rules show rule IDs; larger maps label at most eight
highlighted clusters. `--label-all` shows every rule ID and `--no-labels` hides
all annotations. `--title` changes the plot title. Model inputs also
accept the visualizer's `--n-init`, `--max-iter`, `--tol` and `--seed` controls.

## Clustering definition

The input distances are `D = 1 - signal_jaccard`. Complete linkage starts
with one cluster per rule and repeatedly merges the nearest pair of clusters.
The distance between two clusters is the **largest** distance between their
members. Merging stops when that distance exceeds the cutoff. See the
[complete-linkage definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html).

At threshold 0.5, every pair of distinct rules within a cluster has Jaccard
similarity at least 0.5. This avoids grouping an entire chain merely because
successive neighbors overlap. Weakly connected rules remain singleton clusters;
every rule still receives its own cluster index.

Cluster indices are zero-based and ordered by the first member's position in
the selected input. Equal-distance merge ties follow that input order.
The representative is the highest-scoring member, with score ties resolved by
input order. The original rule IDs remain unchanged.

The compact NumPy implementation uses quadratic memory and cubic worst-case
time. It is intended primarily for retained rule lists; large candidate pools
are more expensive.

## Outputs

For prefix `training_outputs/ggHgammagamma_scan.validation`:

- `.clusters.json`: a rule-ID-to-cluster mapping, directly accepted by
  `plot_erl_mds.py --clusters`.
- `.clustered.mds.json`: the full reusable MDS layout with cluster indices and
  a `clustering` record. Each cluster records member IDs, size, representative,
  minimum/mean pairwise Jaccard and maximum distance. Singleton pairwise
  similarities are `null`. The merge history contains only merges through
  the chosen cutoff; its leaf indices refer to positions in `rule_ids`.
- `.clustered.mds.png`: the MDS figure. The eight largest non-singleton clusters
  have distinct colors and a fixed legend outside the plot. Remaining clusters
  use muted gray circles; singleton rules use gray crosses. Every rule remains
  plotted and keeps its original cluster index in the JSON. On dense maps,
  `C<number>` callouts identify highlighted clusters at their highest-scoring
  member. Rule IDs and full cluster membership remain available in the JSON.

Replot the full labeled layout directly:

```bash
python scripts/plot_erl_mds.py \
  training_outputs/ggHgammagamma_scan.validation.clustered.mds.json \
  --output plots/clustered_rules.pdf
```

Replotting applies the current display layout without repeating MDS or changing
the stored coordinates, similarities, clusters or model.

Existing derived JSON outputs are overwritten with warnings. Figure files
follow Matplotlib's normal overwrite behavior. Re-clustering a labeled layout
can update that derived layout, while preserving its coordinates.

Clusters describe similar selected event populations; they do not establish
physics-process identity. Narrow selections contained within broad selections
can still have low Jaccard similarity. Prefer measured validation matrices over
assumed cross-model block-diagonal similarities. MDS distortion changes the
display, not cluster membership.

The saved pairwise matrix does not determine cluster OR acceptance, incremental
coverage or combined trigger rate; those require event masks or reevaluation.

## Python functions

```python
from scripts.cluster_erl_rules import cluster_erl_model, cluster_layout
from scripts.plot_erl_mds import load_layout, save_layout, plot_mds

layout = cluster_erl_model(model, distance_threshold=0.5, max_iter=2000)
save_layout(layout, "plots/rules.clustered.mds.json")
figure = plot_mds(layout)
figure.savefig("plots/rules.png", dpi=180)

# Try another cutoff without MDS optimization.
layout = cluster_layout(load_layout("plots/rules.clustered.mds.json"), distance_threshold=0.3)
```

`cluster_erl_model()` accepts a core model instance, dictionary or model JSON
path. The lower-level `complete_linkage(distances, distance_threshold)` returns
an aligned NumPy cluster-index array and merge records, without importing ERL
or Matplotlib. These functions remain separate from core training and pruning.
