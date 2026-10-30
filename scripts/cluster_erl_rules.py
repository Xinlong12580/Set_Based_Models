"""Cluster ERL rules by mixture-event Jaccard distance and plot their MDS map.

    python scripts/cluster_erl_rules.py model.validation.model.json
    python scripts/cluster_erl_rules.py model.validation.mds.json --distance-threshold 0.3

Complete linkage uses the original distances, never the MDS coordinates.
A saved MDS layout can be reused without optimization. No rules are pruned.
"""

import argparse
import json
from pathlib import Path
import sys
import warnings

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import plot_erl_mds as visualizer


def complete_linkage(distances, distance_threshold=0.5):
    """Return cluster indices and merges through an inclusive distance cutoff.

    Start with singleton clusters and merge the nearest pair, measuring cluster
    distance by the largest member-to-member distance. Ties follow input order.
    This NumPy implementation uses O(n**2) memory and O(n**3) worst-case time.
    """
    work = np.array(distances, dtype=float, copy=True)
    n = len(work)
    np.fill_diagonal(work, np.inf)
    members = {index: [index] for index in range(n)}
    nodes = list(range(n))
    merges = []
    while len(members) > 1:
        left, right = np.unravel_index(np.argmin(work), work.shape)
        distance = float(work[left, right])
        if distance > distance_threshold:
            break
        members[left].extend(members.pop(right))
        node = n + len(merges)
        merges.append({"node": node, "left": nodes[left], "right": nodes[right],
                       "distance": distance, "size": len(members[left])})
        nodes[left] = node
        updated = np.maximum(work[left], work[right])
        work[left] = updated
        work[:, left] = updated
        work[right, :] = work[:, right] = np.inf
        work[left, left] = np.inf
    indices = np.empty(n, dtype=int)
    for cluster, positions in enumerate(members.values()):
        indices[positions] = cluster
    return indices, merges


def cluster_layout(layout, *, distance_threshold=0.5):
    """Label a layout using its original Jaccard matrix; preserve coordinates.

    Return a new layout with cluster membership, coherence and representatives.
    The representative is the highest-scoring member; score ties follow the
    selected rule order. Singleton clusters remain independent clusters.
    """
    similarity = np.asarray(layout["signal_jaccard"], dtype=float)
    indices, merges = complete_linkage(1. - similarity, distance_threshold)
    metadata = {rule["rule_id"]: rule for rule in layout["rules"]}
    clusters = []
    for index in range(len(set(indices))):
        positions = np.flatnonzero(indices == index)
        rule_ids = [layout["rule_ids"][position] for position in positions]
        representative = max(rule_ids, key=lambda rule_id: metadata[rule_id]["score"])
        block = similarity[np.ix_(positions, positions)]
        pairs = block[np.triu_indices(len(positions), 1)]
        clusters.append({
            "cluster_index": index, "rule_ids": rule_ids, "size": len(rule_ids),
            "representative_rule_id": representative,
            "representative": dict(metadata[representative]),
            "minimum_jaccard": float(pairs.min()) if len(pairs) else None,
            "mean_jaccard": float(pairs.mean()) if len(pairs) else None,
            "maximum_distance": float(1. - pairs.min()) if len(pairs) else 0.,
        })
    return {
        **layout, "cluster_indices": indices.tolist(),
        "clustering": {
            "method": "complete_linkage", "distance": "1 - signal_jaccard",
            "distance_threshold": distance_threshold, "threshold_inclusive": True,
            "representative_selection": "highest_score_then_input_order",
            "cluster_count": len(clusters), "singleton_count": sum(c["size"] == 1 for c in clusters),
            "clusters": clusters, "merges": merges,
        },
    }


def cluster_erl_model(model, *, distance_threshold=0.5, rule_ids=None, all_candidates=False, **mds_options):
    """Accept a core model, model dictionary or JSON path and return a labeled MDS layout."""
    layout = visualizer.fit_erl_mds(model, rule_ids=rule_ids, all_candidates=all_candidates, **mds_options)
    return cluster_layout(layout, distance_threshold=distance_threshold)


def save_clusters(layout, path):
    """Save a rule-ID-to-cluster mapping accepted by plot_erl_mds.py --clusters."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        warnings.warn(f"Overwriting cluster indices: {path}", stacklevel=2)
    mapping = dict(zip(map(str, layout["rule_ids"]), layout["cluster_indices"]))
    path.write_text(json.dumps(mapping, indent=2) + "\n")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="Core ERL model JSON or saved MDS layout JSON")
    parser.add_argument("--distance-threshold", type=float, default=0.5,
                        help="Largest within-cluster Jaccard distance; default 0.5 means Jaccard >= 0.5")
    parser.add_argument("--output-prefix", type=Path, help="Output prefix; default: beside the input")
    parser.add_argument("--output", type=Path, help="Override the figure path, e.g. clusters.pdf")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--all-candidates", action="store_true")
    selection.add_argument("--rule-ids", type=int, nargs="+")
    parser.add_argument("--n-init", type=int, default=4)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    visualizer.add_label_arguments(parser)
    parser.add_argument("--title")
    args = parser.parse_args(argv)
    source = args.input.expanduser().resolve()
    payload = json.loads(source.read_text())
    saved_layout = payload.get("format") == visualizer.FORMAT
    if saved_layout:
        if args.all_candidates or args.rule_ids is not None:
            parser.error("Rule selection requires a model input; a saved layout already fixes the point set")
        layout = cluster_layout(visualizer.load_layout(source), distance_threshold=args.distance_threshold)
    else:
        layout = cluster_erl_model(payload, distance_threshold=args.distance_threshold,
                                   rule_ids=args.rule_ids, all_candidates=args.all_candidates,
                                   n_init=args.n_init, max_iter=args.max_iter, tolerance=args.tol, seed=args.seed)
        layout["source_model"] = str(source)
    base = source.with_suffix("")
    for ending in (".clustered.mds.json", ".mds.json", ".model.json"):
        if source.name.endswith(ending):
            base = source.with_name(source.name.removesuffix(ending))
            break
    prefix = (args.output_prefix or base).expanduser()
    clusters_path = Path(f"{prefix}.clusters.json")
    layout_path = Path(f"{prefix}.clustered.mds.json")
    output = (args.output or Path(f"{prefix}.clustered.mds.png")).expanduser()
    targets = [path.resolve() for path in (clusters_path, layout_path, output)]
    protected = [targets[0], targets[2]] if saved_layout else targets
    if source in protected or len(set(targets)) != len(targets):
        parser.error("Output paths must be distinct and differ from the input model/layout")
    save_clusters(layout, clusters_path)
    visualizer.save_layout(layout, layout_path)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure = visualizer.plot_mds(layout, labels=args.labels, title=args.title)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    result = layout["clustering"]
    print(f"{len(layout['rule_ids'])} rules -> {result['cluster_count']} clusters "
          f"({result['singleton_count']} singletons); complete-linkage distance <= {args.distance_threshold:g}")
    fit = layout["optimization"]
    print(f"MDS normalized stress={fit['normalized_stress']:.6g}; iterations={fit['n_iter']}; converged={fit['converged']}")
    for cluster in result["clusters"]:
        print(f"Cluster {cluster['cluster_index']}: rules={cluster['rule_ids']}; "
              f"representative={cluster['representative_rule_id']}; max distance={cluster['maximum_distance']:.4f}")
    print(f"Cluster indices saved to: {clusters_path}\nClustered layout saved to: {layout_path}\nPlot saved to: {output}")
    return layout


if __name__ == "__main__":
    main()
