"""Run Gini-scored ERL with object-level and event-level inputs.

The input ``.pt`` files must contain the original ``obj``, ``label``, and
``obj_columns`` entries plus ``event_features`` and
``event_feature_columns``.  Event features are interpreted as non-negative
integer-valued ordinal observables, matching the output of
``scripts/add_event_multiplicities.py``.  At every rule-search layer,
event-level and object-level cuts compete using Gini gain.

Run, for example::

    python casual_tests/stale/run_ERL_Gini_with_event_features.py \
        --signal data/with_event_multiplicities/anomalous.pt \
        --background data/with_event_multiplicities/reference.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from set_based_models import (  # noqa: E402
    EventCut,
    ExistentialRuleList,
    ObservableSpec,
    PartitionStatistics,
    RuleStatistics,
    SplitStatistics,
    gini_gain,
)


OBJECT_COLUMNS = ["pt", "eta", "phi", "type_id"]
OBSERVABLES = (
    ObservableSpec("pt", "continuous"),
    ObservableSpec("eta", "continuous"),
    ObservableSpec("phi", "continuous"),
    ObservableSpec(
        "type_id",
        "categorical",
        {0: "jet", 1: "muon", 2: "electron", 3: "photon", 4: "met"},
    ),
)
SIGNAL_SIMILARITY_THRESHOLD = 0.2
BACKGROUND_SIMILARITY_THRESHOLD = 0.2


def gini_rule_score(statistics: RuleStatistics) -> float:
    """Score a rule's pass/fail partition using weighted Gini gain."""
    parent = PartitionStatistics(
        signal_count=statistics.signal_total,
        background_count=statistics.background_total,
        signal_weight=statistics.signal_weight_total,
        background_weight=statistics.background_weight_total,
    )
    positive = PartitionStatistics(
        signal_count=statistics.signal_accepted,
        background_count=statistics.background_accepted,
        signal_weight=statistics.signal_weight_accepted,
        background_weight=statistics.background_weight_accepted,
    )
    negative = PartitionStatistics(
        signal_count=parent.signal_count - positive.signal_count,
        background_count=parent.background_count - positive.background_count,
        signal_weight=parent.signal_weight - positive.signal_weight,
        background_weight=parent.background_weight - positive.background_weight,
    )
    return gini_gain(SplitStatistics(parent, positive, negative, root=parent))


def load_dataset(path: str | Path):
    """Load one augmented dataset and return NumPy model inputs."""
    path = Path(path).expanduser().resolve()
    raw = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "obj",
        "label",
        "obj_columns",
        "event_features",
        "event_feature_columns",
    }
    if not isinstance(raw, dict) or not required <= raw.keys():
        missing = required - raw.keys() if isinstance(raw, dict) else required
        raise ValueError(f"{path} is missing required keys: {sorted(missing)}")
    if raw["obj_columns"] != OBJECT_COLUMNS:
        raise ValueError(f"Unexpected object columns in {path}: {raw['obj_columns']!r}")

    obj = raw["obj"]
    label = raw["label"]
    event_features = raw["event_features"]
    event_columns = raw["event_feature_columns"]
    if not isinstance(obj, torch.Tensor) or obj.ndim != 3 or obj.shape[1:] != (23, 4):
        shape = tuple(obj.shape) if isinstance(obj, torch.Tensor) else type(obj).__name__
        raise ValueError(f"Expected obj shape (events, 23, 4) in {path}, got {shape}")
    if not isinstance(label, torch.Tensor) or label.ndim != 1 or label.shape[0] != obj.shape[0]:
        raise ValueError(f"label must contain one entry per obj row in {path}.")
    if not isinstance(event_features, torch.Tensor) or event_features.ndim != 2:
        raise ValueError(f"event_features must be a two-dimensional tensor in {path}.")
    if event_features.shape[0] != obj.shape[0]:
        raise ValueError(f"event_features must contain one row per obj row in {path}.")
    if (
        not isinstance(event_columns, list)
        or not event_columns
        or not all(isinstance(name, str) and name.strip() for name in event_columns)
        or len(event_columns) != event_features.shape[1]
    ):
        raise ValueError(
            f"event_feature_columns must name every event_features column in {path}."
        )
    if len(set(event_columns)) != len(event_columns):
        raise ValueError(f"event_feature_columns contains duplicate names in {path}.")
    if not bool(torch.isfinite(event_features).all()):
        raise ValueError(f"event_features must contain only finite values in {path}.")
    if not bool((event_features == torch.floor(event_features)).all()):
        raise ValueError(f"event_features must contain integer-valued multiplicities in {path}.")
    if not bool((event_features >= 0).all()):
        raise ValueError(f"event_features must contain non-negative multiplicities in {path}.")

    obj = torch.nan_to_num(obj, nan=0.0, posinf=0.0, neginf=0.0)
    slot_mask = obj[..., 0] > 0
    return (
        obj.numpy(),
        slot_mask.numpy(),
        event_features.numpy(),
        label,
        tuple(event_columns),
    )


def print_rules(title, rules, model) -> None:
    print(f"\n{title} ({len(rules)} rules):")
    for rank, rule in enumerate(rules, start=1):
        stats = rule.statistics
        tpr, fpr = stats.true_positive_rate, stats.false_positive_rate
        direction = "mixture-enriched" if tpr > fpr else (
            "reference-enriched" if tpr < fpr else "equal acceptance rates"
        )
        print(f"\n[{rank}] Rule {rule.rule_id}")
        print(f"  Gini gain: {rule.generation_score:.6g}")
        print(f"  rule:      {rule.to_text()}")
        print(f"  direction: {direction}")
        print(f"  TPR (mixture): {tpr:.6f}; FPR (reference): {fpr:.6f}")
        print(f"  mixture-label purity: {stats.precision:.6f}")
        print(
            f"  accepted: mixture {stats.signal_accepted}/{stats.signal_total}, "
            f"reference {stats.background_accepted}/{stats.background_total}"
        )
        print("  cut chain:")
        for step in rule.steps:
            schema = (
                model.event_observables
                if isinstance(step.cut, EventCut)
                else model.observables
            )
            print(
                f"    {step.depth}. {step.cut.to_text(schema)} "
                f"(prefix Gini gain={step.generation_score:.6g})"
            )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--signal",
        type=Path,
        default=ROOT / "data/with_event_multiplicities/anomalous.pt",
        help="Signal/mixture augmented .pt file.",
    )
    parser.add_argument(
        "--background",
        type=Path,
        default=ROOT / "data/with_event_multiplicities/reference.pt",
        help="Reference/background augmented .pt file.",
    )
    args = parser.parse_args(argv)
    signal_path = args.signal.expanduser().resolve()
    background_path = args.background.expanduser().resolve()

    try:
        signal, signal_slot_mask, signal_events, signal_labels, signal_event_columns = (
            load_dataset(signal_path)
        )
        (
            background,
            background_slot_mask,
            background_events,
            background_labels,
            background_event_columns,
        ) = load_dataset(background_path)
        if signal_event_columns != background_event_columns:
            raise ValueError(
                "Signal and background event_feature_columns must match exactly; "
                f"received {list(signal_event_columns)!r} and "
                f"{list(background_event_columns)!r}."
            )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    event_observables = tuple(
        ObservableSpec(name, "ordinal") for name in signal_event_columns
    )

    print(f"Signal input: {signal_path}")
    print(f"Background input: {background_path}")
    print(f"Event observables: {', '.join(signal_event_columns)}")
    print(
        f"Signal sample: {len(signal)} events; stored labels:",
        torch.unique(signal_labels, return_counts=True),
    )
    print(
        f"Reference background: {len(background)} events; stored labels:",
        torch.unique(background_labels, return_counts=True),
    )
    print("All rows in the signal input are treated as signal, regardless of stored process labels.")
    print("Scoring: weighted Gini gain of each rule versus its complement.")
    print("Gini is symmetric; each rule's enrichment direction is reported below.")

    model = ExistentialRuleList(
        OBSERVABLES,
        event_observables=event_observables,
        max_rule_depth=3,
        beam_width=40,
        max_candidate_rules=200,
        max_thresholds=32,
        minimum_count=1,
        beam_similarity_filter=True,
        scoring_function=gini_rule_score,
        beam_signal_similarity_threshold=SIGNAL_SIMILARITY_THRESHOLD,
        min_signal_efficiency=0.01,
        beam_background_similarity_threshold=BACKGROUND_SIMILARITY_THRESHOLD,
    )
    model.fit(
        signal,
        background,
        signal_slot_mask=signal_slot_mask,
        background_slot_mask=background_slot_mask,
        signal_event_features=signal_events,
        background_event_features=background_events,
    )

    print(
        f"Generated {len(model.candidate_rules_)} candidate rules from "
        f"{len(signal)} signal and {len(background)} background events."
    )
    if not model.candidate_rules_:
        print("No valid rules were generated.")
        return

    print_rules(
        "Top unpruned rules (higher Gini gain is better)",
        model.candidate_rules_[:100],
        model,
    )

    pruned_rules = model.prune_candidates(
        signal_similarity_threshold=SIGNAL_SIMILARITY_THRESHOLD,
        background_similarity_threshold=BACKGROUND_SIMILARITY_THRESHOLD,
    )
    print(
        "\nPost-training pruning: "
        f"{len(model.candidate_rules_)} -> {len(pruned_rules)} rules "
        f"(signal Jaccard < {SIGNAL_SIMILARITY_THRESHOLD:g} or "
        f"background Jaccard < {BACKGROUND_SIMILARITY_THRESHOLD:g})."
    )
    print_rules("Top pruned rules", pruned_rules[:10], model)


if __name__ == "__main__":
    main()
