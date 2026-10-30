"""Fit or replot a 2D metric-MDS map of an ERL model's mixture-event overlaps.

    python scripts/plot_erl_mds.py model.validation.model.json
    python scripts/plot_erl_mds.py model.validation.mds.json --clusters clusters.json

Only fitting needs the ERL package. The optimizer uses NumPy; plotting needs
Matplotlib. Cluster labels color points and never enter the optimization.
"""

import argparse
import json
from pathlib import Path
import sys
import warnings

import numpy as np


FORMAT = "set_based_models_erl_mds"


def pairwise_distances(coordinates):
    coordinates = np.asarray(coordinates, dtype=float)
    return np.hypot(coordinates[:, None, 0] - coordinates[None, :, 0],
                    coordinates[:, None, 1] - coordinates[None, :, 1])


def metric_mds(distances, *, n_init=4, max_iter=500, tolerance=1e-6, seed=42):
    """Return coordinates and fit diagnostics using complete, unweighted SMACOF.

    Minimize sum((embedded_distance - target_distance)**2) over unique pairs.
    Stop when the stress improvement is <= tolerance * previous_stress, or
    max_iter is reached. Keep the lowest-stress random initialization.
    """
    distances = np.asarray(distances, dtype=float)
    n = len(distances)
    if not n or distances.shape != (n, n):
        raise ValueError("MDS requires a nonempty square distance matrix")
    if n_init < 1:
        raise ValueError("At least one initialization is required")
    rng = np.random.default_rng(seed)
    best = None
    starts = []
    for start in range(n_init):
        coordinates = rng.normal(size=(n, 2))
        coordinates -= coordinates.mean(axis=0)
        embedded = pairwise_distances(coordinates)
        denominator = np.sum(embedded**2)
        coordinates *= np.sum(embedded * distances) / denominator if denominator else 0.
        embedded = pairwise_distances(coordinates)
        stress = float(np.sum((embedded - distances)**2) / 2)
        history = [stress]
        converged = stress == 0.
        for _ in range(max_iter):
            if converged:
                break
            ratio = np.divide(distances, embedded, out=np.zeros_like(distances), where=embedded > 0)
            coordinates = (ratio.sum(axis=1)[:, None] * coordinates - ratio @ coordinates) / n
            coordinates -= coordinates.mean(axis=0)
            embedded = pairwise_distances(coordinates)
            previous = stress
            stress = float(np.sum((embedded - distances)**2) / 2)
            history.append(stress)
            converged = previous - stress <= tolerance * previous
        starts.append({"start": start, "stress": stress, "n_iter": len(history) - 1, "converged": converged})
        if best is None or stress < best["stress"]:
            best = {**starts[-1], "coordinates": coordinates.copy(), "stress_history": history}
    coordinates = best.pop("coordinates")
    error = pairwise_distances(coordinates) - distances
    target_norm = float(np.sum(distances**2) / 2)
    diagnostics = {
        "algorithm": "SMACOF", "dimensions": 2, "initialization": "centered_rescaled_gaussian",
        "seed": seed, "n_init": n_init, "max_iter": max_iter, "tolerance": tolerance,
        "stopping_rule": "previous_stress - stress <= tolerance * previous_stress",
        **best, "starts": starts,
        "normalized_stress": float(np.sqrt(best["stress"] / target_norm)) if target_norm else 0.,
        "normalized_stress_definition": "sqrt(stress / sum(target_distance**2 over i < j))",
        "distance_rmse": float(np.sqrt(best["stress"] / max(1, n * (n - 1) // 2))),
        "max_absolute_distance_error": float(np.max(np.abs(error))),
    }
    return coordinates, diagnostics


def _cluster_indices(rule_ids, cluster_indices):
    if cluster_indices is None:
        return [0] * len(rule_ids)
    if isinstance(cluster_indices, dict):
        values = [cluster_indices[str(rule_id)] if str(rule_id) in cluster_indices else cluster_indices[rule_id]
                  for rule_id in rule_ids]
    else:
        values = list(cluster_indices)
    if len(values) != len(rule_ids):
        raise ValueError("Cluster indices must match the plotted rule IDs")
    return [value.item() if isinstance(value, np.generic) else value for value in values]


def fit_erl_mds(model, *, rule_ids=None, all_candidates=False, cluster_indices=None, **mds_options):
    """Accept a core model, model dictionary or JSON path; return a portable layout.

    Default to retained rules after pruning, or every candidate before pruning.
    Cluster indices may be a sequence aligned with the output rule_ids, or a
    mapping keyed by rule ID. Omitting cluster_indices gives one shared cluster (0).
    """
    source = None
    if isinstance(model, (str, Path)):
        source = str(Path(model).expanduser().resolve())
        model = json.loads(Path(source).read_text())
    if isinstance(model, dict):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from set_based_models import ExistentialRuleList
        model = ExistentialRuleList.from_dict(model)
    if model.signal_jaccard_ is None:
        raise ValueError("The model has no saved mixture Jaccard matrix; recalculate candidates and save it first")
    if model.jaccard_source_ == "block_diagonal":
        warnings.warn("Cross-model overlaps are assumed zero; recalculate candidates on common events for measured overlaps",
                      stacklevel=2)
    selection = "explicit"
    if rule_ids is None:
        retained = model.pruning_result_ is not None and not all_candidates
        rules = model.selected_rules_ if retained else model.candidate_rules_
        rule_ids = [rule.rule_id for rule in rules]
        selection = "retained" if retained else "all_candidates"
    rule_ids = list(rule_ids)
    if not rule_ids:
        raise ValueError("No rules selected; use all_candidates=True or --all-candidates to inspect the candidate pool")
    similarity = model.signal_jaccard_[np.ix_(rule_ids, rule_ids)]
    coordinates, diagnostics = metric_mds(1. - similarity, **mds_options)
    rules = [model.get_rule(rule_id) for rule_id in rule_ids]
    return {
        "format": FORMAT, "schema_version": 1, "source_model": source,
        "statistics_source": model.statistics_source_, "jaccard_source": model.jaccard_source_,
        "source_jaccard_blocks": list(model.jaccard_blocks_), "selection": selection,
        "distance": "1 - signal_jaccard", "rule_ids": rule_ids,
        "cluster_indices": _cluster_indices(rule_ids, cluster_indices),
        "coordinates": coordinates.tolist(), "signal_jaccard": similarity.tolist(),
        "optimization": diagnostics,
        "rules": [{"rule_id": rule.rule_id, "rule": rule.to_text(), "score": rule.generation_score,
                   "signal_acceptance": rule.statistics.true_positive_rate,
                   "reference_acceptance": rule.statistics.false_positive_rate} for rule in rules],
    }


def save_layout(layout, path):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        warnings.warn(f"Overwriting MDS layout: {path}", stacklevel=2)
    path.write_text(json.dumps(layout, indent=2, allow_nan=False) + "\n")
    return path


def load_layout(path):
    layout = json.loads(Path(path).expanduser().read_text())
    if layout.get("format") != FORMAT or layout.get("schema_version") != 1:
        raise ValueError("Expected a version 1 ERL MDS layout")
    return layout


def _annotate_clusters(figure, axes, labels):
    """Place a few cluster callouts without covering one another."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    occupied = []
    bounds = axes.get_window_extent(renderer)
    offsets = [(8, 8), (8, -20), (-36, 8), (-36, -20), (8, 30),
               (-36, 30), (30, -36), (-54, -36), (30, 48), (-54, 48)]
    for text, coordinate, color in labels:
        annotation = axes.annotate(text, coordinate, xytext=offsets[0], textcoords="offset points", fontsize=9,
                                   fontweight="bold", color=color, zorder=5,
                                   bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, lw=.7),
                                   arrowprops=dict(arrowstyle="-", color=color, lw=.6))
        for offset in offsets:
            annotation.set_position(offset)
            box = annotation.get_window_extent(renderer).expanded(1.1, 1.1)
            if (bounds.contains(box.x0, box.y0) and bounds.contains(box.x1, box.y1)
                    and not any(box.overlaps(other) for other in occupied)):
                break
        occupied.append(box)


def plot_mds(layout, *, cluster_indices=None, labels=None, title=None):
    """Return a figure without fitting; labels=None keeps dense maps readable.

    Small maps label every rule. Dense maps highlight at most eight clusters,
    label those clusters, and mute smaller groups. All rules remain plotted.
    Pass labels=True to annotate every rule, or False to hide all annotations.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    coordinates = np.asarray(layout["coordinates"])
    clusters = _cluster_indices(layout["rule_ids"], layout["cluster_indices"] if cluster_indices is None else cluster_indices)
    groups = list(dict.fromkeys(clusters))
    sizes = {group: clusters.count(group) for group in groups}
    grouped = bool(layout.get("clustering")) or len(groups) > 1
    figure, axes = plt.subplots(figsize=(12, 8))
    figure.subplots_adjust(left=.08, right=.73, bottom=.13, top=.85)
    if layout.get("clustering"):
        singletons = np.array([sizes[group] == 1 for group in clusters])
        if singletons.any():
            axes.scatter(coordinates[singletons, 0], coordinates[singletons, 1], s=24, color="#aab2bd",
                         marker="x", linewidths=.9, label=f"Singletons ({singletons.sum()} rules)", zorder=2)
        groups = [group for group in groups if sizes[group] > 1]
    groups.sort(key=lambda group: -sizes[group])
    highlighted, other = groups[:8], groups[8:]
    if other:
        selected = np.array([group in other for group in clusters])
        axes.scatter(coordinates[selected, 0], coordinates[selected, 1], s=30, color="#788797", alpha=.55,
                     label=f"Other clusters ({len(other)})", zorder=2)
    colors = plt.get_cmap("tab10")
    callouts = []
    metadata = {rule["rule_id"]: rule for rule in layout["rules"]}
    for index, cluster in enumerate(highlighted):
        selected = np.array([value == cluster for value in clusters])
        color = colors([0, 1, 2, 3, 4, 5, 6, 9][index])
        label = f"C{cluster}: {sizes[cluster]} rules" if grouped else f"All rules ({len(clusters)})"
        axes.scatter(coordinates[selected, 0], coordinates[selected, 1], s=48, color=color, alpha=.85,
                     edgecolors="white", linewidths=.5, label=label, zorder=3)
        representative = max(np.flatnonzero(selected), key=lambda position: metadata[layout["rule_ids"][position]]["score"])
        callouts.append((f"C{cluster}", coordinates[representative], color))
    if labels is True or (labels is None and len(clusters) <= 40):
        for rule_id, coordinate in zip(layout["rule_ids"], coordinates):
            axes.annotate(str(rule_id), coordinate, xytext=(5, 5), textcoords="offset points", fontsize=8)
    fit = layout["optimization"]
    figure.suptitle(title or "ERL rule overlaps", x=.08, y=.97, ha="left", fontsize=18, fontweight="bold")
    description = f"{len(clusters)} rules"
    if grouped:
        description += f"  |  {len(sizes)} clusters"
    figure.text(.08, .91, description + f"  |  MDS normalized stress: {fit['normalized_stress']:.3f}",
                fontsize=11, color="#536171")
    axes.set_xlabel("MDS coordinate 1")
    axes.set_ylabel("MDS coordinate 2")
    axes.set_aspect("equal", adjustable="box")
    axes.set_axisbelow(True)
    axes.grid(alpha=.18)
    axes.spines[["top", "right"]].set_visible(False)
    axes.xaxis.set_major_locator(MaxNLocator(5))
    axes.yaxis.set_major_locator(MaxNLocator(5))
    axes.margins(.12)
    handles, legend_labels = axes.get_legend_handles_labels()
    # Highlighted groups precede the muted context groups in the fixed sidebar.
    order = list(range(len(handles) - len(highlighted), len(handles))) + list(range(len(handles) - len(highlighted)))
    figure.legend([handles[i] for i in order], [legend_labels[i] for i in order],
                  title="Largest clusters" if other else "Rule groups", loc="upper left",
                  bbox_to_anchor=(.76, .85), frameon=False, fontsize=10, title_fontsize=11,
                  labelspacing=.9, handletextpad=.5)
    figure.text(.08, .055, "MDS distances are approximate; clustering uses the original Jaccard distances.",
                fontsize=9, color="#536171")
    if labels is None and len(clusters) > 40 and grouped:
        _annotate_clusters(figure, axes, callouts)
    return figure


def add_label_arguments(parser):
    labels = parser.add_mutually_exclusive_group()
    labels.add_argument("--label-all", dest="labels", action="store_true", help="Annotate every rule ID, including dense maps")
    labels.add_argument("--no-labels", dest="labels", action="store_false", help="Hide rule and cluster annotations")
    parser.set_defaults(labels=None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="Core ERL model JSON or previously saved MDS layout JSON")
    parser.add_argument("--output", type=Path, help="PNG/PDF/SVG output; default: INPUT.mds.png")
    parser.add_argument("--layout-output", type=Path, help="Save fit/updated layout JSON; automatic when fitting a model")
    parser.add_argument("--clusters", type=Path, help="JSON rule-ID-to-cluster mapping, or list aligned with plotted rule_ids")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--all-candidates", action="store_true")
    selection.add_argument("--rule-ids", type=int, nargs="+")
    parser.add_argument("--n-init", type=int, default=4)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    add_label_arguments(parser)
    parser.add_argument("--title")
    args = parser.parse_args(argv)
    source = args.input.expanduser().resolve()
    payload = json.loads(source.read_text())
    clusters = None if args.clusters is None else json.loads(args.clusters.expanduser().read_text())
    replot = payload.get("format") == FORMAT
    if replot:
        layout = load_layout(source)
        if args.all_candidates or args.rule_ids is not None:
            parser.error("Rule selection requires a model input; a saved layout already fixes the point set")
        if clusters is not None:
            layout["cluster_indices"] = _cluster_indices(layout["rule_ids"], clusters)
        base = source.with_name(source.name.removesuffix(".mds.json")) if source.name.endswith(".mds.json") else source.with_suffix("")
    else:
        layout = fit_erl_mds(payload, rule_ids=args.rule_ids, all_candidates=args.all_candidates,
                             cluster_indices=clusters, n_init=args.n_init, max_iter=args.max_iter,
                             tolerance=args.tol, seed=args.seed)
        layout["source_model"] = str(source)
        base = source.with_name(source.name.removesuffix(".model.json")) if source.name.endswith(".model.json") else source.with_suffix("")
    layout_path = args.layout_output or (None if replot else Path(f"{base}.mds.json"))
    output = (args.output or Path(f"{base}.mds.png")).expanduser()
    if output.resolve() == source or (not replot and layout_path.expanduser().resolve() == source):
        parser.error("Output paths must differ from the input model/layout")
    if layout_path is not None:
        if layout_path.expanduser().resolve() == output.resolve():
            parser.error("Layout and figure output paths must differ")
        save_layout(layout, layout_path)
        print(f"Layout saved to: {layout_path}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure = plot_mds(layout, labels=args.labels, title=args.title)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    fit = layout["optimization"]
    print(f"{'Loaded' if replot else 'Fitted'} {len(layout['rule_ids'])} rules; stress={fit['stress']:.6g}; "
          f"normalized stress={fit['normalized_stress']:.6g}; iterations={fit['n_iter']}; converged={fit['converged']}")
    print(f"Plot saved to: {output}")
    return layout


if __name__ == "__main__":
    main()
