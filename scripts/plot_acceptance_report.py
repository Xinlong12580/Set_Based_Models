"""Plot trigger paths from a structured EDT or ERL quantitative report.

The left panel places each selected trigger path on the acceptance plane:
reference acceptance on the horizontal axis and anomalous-mixture acceptance
on the vertical axis.  The right panel maps the numbered points to readable
trigger definitions and their quantitative metrics.

Example::

    python scripts/plot_acceptance_report.py \
        reports/ERL_ggHgammagamma_object_only.txt \
        --output reports/ERL_ggHgammagamma_object_only.png

By default, ERL plots the ten highest-ranked, similarity-pruned,
mixture-enriched rules. EDT plots its ten highest-ranked mixture-enriched
leaves. Use ``--rules all`` or ``--direction all`` to change that selection.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


REPORT_FORMAT = "set_based_models_quantitative_report"
DIRECTION_NAMES = {
    "mixture": "mixture-enriched",
    "reference": "reference-enriched",
}


def load_report(path: str | Path) -> dict:
    """Load and minimally validate a quantitative runner report."""
    path = Path(path).expanduser().resolve()
    try:
        with path.open(encoding="utf-8") as handle:
            report = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(report, dict) or report.get("format") != REPORT_FORMAT:
        raise ValueError(f"{path} is not a {REPORT_FORMAT!r} report.")
    if report.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported report schema version {report.get('schema_version')!r}."
        )
    results = report.get("results")
    if not isinstance(results, dict) or not ({"rules", "leaves"} & results.keys()):
        raise ValueError(f"{path} does not contain rule or leaf results.")
    return report


def select_paths(report: dict, *, rules: str, direction: str, top: int) -> tuple[list, list]:
    """Return all eligible paths and the ranked subset to annotate."""
    results = report["results"]
    is_erl = "rules" in results
    records = list(results["rules"] if is_erl else results["leaves"])
    requested_direction = DIRECTION_NAMES.get(direction)

    eligible = []
    for record in records:
        metrics = _metrics(record)
        if requested_direction is not None and metrics["direction"] != requested_direction:
            continue
        if is_erl and rules == "pruned" and not record.get(
            "retained_after_similarity_pruning", False
        ):
            continue
        eligible.append(record)

    if not eligible:
        qualifier = direction if direction != "all" else "requested"
        raise ValueError(f"The report contains no {qualifier} paths for this selection.")
    selected = eligible[:top]
    return eligible, selected


def make_plot(
    report: dict,
    *,
    rules: str = "pruned",
    direction: str = "mixture",
    top: int = 10,
    title: str | None = None,
):
    """Create and return the two-panel acceptance figure."""
    eligible, selected = select_paths(report, rules=rules, direction=direction, top=top)
    model = report.get("model", "unknown model")
    is_erl = "rules" in report["results"]
    if not is_erl:
        rules = "all"

    figure_height = max(6.2, 1.02 * len(selected) + 1.8)
    figure = plt.figure(figsize=(15.5, figure_height), constrained_layout=True)
    axes = figure.subplots(1, 2, gridspec_kw={"width_ratios": (1.08, 1.0)})
    plane, definitions = axes

    context_x = [100.0 * _metrics(record)["background_acceptance"] for record in eligible]
    context_y = [100.0 * _metrics(record)["signal_acceptance"] for record in eligible]
    plane.scatter(
        context_x,
        context_y,
        s=30,
        c="#b8bec7",
        alpha=0.52,
        linewidths=0,
        label="eligible paths",
        zorder=1,
    )

    colors = plt.get_cmap("viridis")(
        [0.12 + 0.76 * index / max(1, len(selected) - 1) for index in range(len(selected))]
    )
    coordinate_uses: dict[tuple[float, float], int] = {}
    for number, (record, color) in enumerate(zip(selected, colors), start=1):
        metrics = _metrics(record)
        x = 100.0 * metrics["background_acceptance"]
        y = 100.0 * metrics["signal_acceptance"]
        plane.scatter(
            [x],
            [y],
            s=105,
            c=[color],
            edgecolors="black",
            linewidths=0.7,
            zorder=3,
        )
        coordinate = (round(x, 12), round(y, 12))
        occurrence = coordinate_uses.get(coordinate, 0)
        coordinate_uses[coordinate] = occurrence + 1
        plane.annotate(
            str(number),
            (x, y),
            xytext=(7, 7 + 13 * occurrence),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color=color,
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": color, "alpha": 0.95},
            zorder=4,
        )

    upper = max(context_x + context_y + [0.1]) * 1.08
    plane.plot([0, upper], [0, upper], linestyle="--", color="#666666", linewidth=1.1)
    plane.set_xlim(left=-0.02 * upper, right=upper)
    plane.set_ylim(bottom=-0.02 * upper, top=upper)
    plane.set_aspect("equal", adjustable="box")
    plane.set_anchor("N")
    plane.grid(True, color="#d9dde3", linewidth=0.7, alpha=0.75)
    plane.set_xlabel("Reference acceptance [%]", fontsize=12)
    plane.set_ylabel("Anomalous-mixture acceptance [%]", fontsize=12)
    plane.set_title("Acceptance plane", fontsize=14, fontweight="bold")
    plane.text(
        0.98,
        0.02,
        "Above diagonal: mixture-enriched\nBelow diagonal: reference-enriched",
        transform=plane.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#4d535b",
    )
    plane.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="", color="#b8bec7", label="eligible path"),
            Line2D(
                [0], [0], marker="o", linestyle="", markeredgecolor="black",
                color=colors[0], markersize=8, label="numbered path",
            ),
            Line2D([0], [0], linestyle="--", color="#666666", label="equal acceptance"),
        ],
        loc="upper left",
        frameon=True,
        fontsize=9,
    )

    definitions.axis("off")
    definitions.set_title("Trigger paths", fontsize=14, fontweight="bold", loc="left")
    y_position = 0.985
    line_height = min(0.026, 0.72 / max(12, sum(_entry_lines(record) for record in selected)))
    for number, (record, color) in enumerate(zip(selected, colors), start=1):
        metrics = _metrics(record)
        path = _path_text(record)
        header = _record_header(record, is_erl)
        metric_line = (
            f"A={100 * metrics['signal_acceptance']:.3f}%   "
            f"R={100 * metrics['background_acceptance']:.3f}%   "
            f"Δ={100 * metrics['acceptance_difference']:+.3f}%"
        )
        score_line = _score_text(record, is_erl)
        wrapped_path = textwrap.wrap(path, width=62) or ["(empty path)"]
        definitions.text(
            0.0,
            y_position,
            f"{number}",
            transform=definitions.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            fontweight="bold",
            color="white",
            bbox={"boxstyle": "circle,pad=0.28", "fc": color, "ec": color},
        )
        definitions.text(
            0.065,
            y_position,
            header,
            transform=definitions.transAxes,
            va="top",
            fontsize=9.5,
            fontweight="bold",
            color="#222222",
        )
        y_position -= line_height * 1.2
        for line in wrapped_path:
            definitions.text(
                0.065,
                y_position,
                line,
                transform=definitions.transAxes,
                va="top",
                fontsize=9.1,
                family="monospace",
                color="#222222",
            )
            y_position -= line_height
        definitions.text(
            0.065,
            y_position,
            f"{metric_line}   {score_line}",
            transform=definitions.transAxes,
            va="top",
            fontsize=8.7,
            color="#4d535b",
        )
        y_position -= line_height * 1.85

    dataset = Path(report["inputs"]["signal"]["path"]).parent.name
    selection_text = f"{rules}, {direction}"
    default_title = f"{model}: {dataset} ({selection_text})"
    figure.suptitle(title or default_title, fontsize=16, fontweight="bold")
    return figure


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Structured JSON .txt report to plot.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output image path (default: <report stem>_acceptance.png).",
    )
    parser.add_argument(
        "--top",
        type=_positive_int,
        default=10,
        help="Number of ranked paths to number and define (default: 10).",
    )
    parser.add_argument(
        "--rules",
        choices=("pruned", "all"),
        default="pruned",
        help="For ERL, plot similarity-pruned or all candidates (default: pruned).",
    )
    parser.add_argument(
        "--direction",
        choices=("mixture", "reference", "all"),
        default="mixture",
        help="Path enrichment direction to include (default: mixture).",
    )
    parser.add_argument("--title", help="Optional custom figure title.")
    parser.add_argument("--dpi", type=_positive_int, default=180)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permit replacement of an existing output image.",
    )
    args = parser.parse_args(argv)

    report_path = args.report.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else report_path.with_name(f"{report_path.stem}_acceptance.png")
    )
    if output.exists() and not args.overwrite:
        parser.error(f"Refusing to overwrite {output}; use --overwrite.")
    try:
        report = load_report(report_path)
        figure = make_plot(
            report,
            rules=args.rules,
            direction=args.direction,
            top=args.top,
            title=args.title,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=args.dpi, bbox_inches="tight")
        plt.close(figure)
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Saved acceptance plot to {output}")


def _metrics(record):
    metrics = record.get("metrics")
    required = {
        "signal_acceptance",
        "background_acceptance",
        "acceptance_difference",
        "direction",
    }
    if not isinstance(metrics, dict) or not required <= metrics.keys():
        raise ValueError("A result record is missing required acceptance metrics.")
    for key in ("signal_acceptance", "background_acceptance", "acceptance_difference"):
        value = metrics[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Result metric {key!r} must be finite and numeric.")
    return metrics


def _path_text(record):
    if "simplified_rule" in record:
        return record["simplified_rule"]
    path = record.get("path", [])
    return " AND ".join(step["selected_condition"] for step in path) or "ROOT"


def _record_header(record, is_erl):
    if is_erl:
        return f"Rule {record['rule_id']} · global rank {record['rank']}"
    return f"Leaf {record['leaf_id']} · rank {record['rank']}"


def _score_text(record, is_erl):
    if is_erl:
        return f"Gini={record['gini_gain']:.5g}"
    return f"mixture fraction={record['leaf_signal_fraction']:.4f}"


def _entry_lines(record):
    return len(textwrap.wrap(_path_text(record), width=62) or [""]) + 3


def _positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


if __name__ == "__main__":
    main()
