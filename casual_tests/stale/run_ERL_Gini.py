"""Run with ``python casual_tests/stale/run_ERL_Gini.py``.

Use --signal and --background to choose .pt files; explicit relative paths are
resolved from the shell's current directory. Defaults are the repository's
data/anomalous.pt and data/reference.pt, independent of the current directory.
All input rows are used, with the search settings of run_ERL_reference_anomalous.py.
Gini gain scores every rule's pass/fail partition of the full training sample;
it is used during beam search as well as for the final candidate ranking.
"""

import argparse
from pathlib import Path
import sys

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from set_based_models import (  # noqa: E402
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
    """Adapt ERL statistics to EDT's weighted Gini impurity reduction.

    The parent is the full mixture/reference training sample, the positive
    branch passes the entire rule, and the negative branch is its complement.
    Each prefix is scored against that same full sample, not just its previous
    prefix. Event weights determine both class purity and branch size.

    This is the ordinary symmetric Gini gain: either mixture-enriched or
    reference-enriched pass branches can score highly. No direction filter or
    rule-length penalty is applied, and empty branches have zero impurity.
    """
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


def load_obj_dataset(path: str | Path):
    path = Path(path).expanduser().resolve()
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict) or not {"obj", "label", "obj_columns"} <= raw.keys():
        raise ValueError(f"{path} must contain obj, label, and obj_columns.")
    if raw["obj_columns"] != OBJECT_COLUMNS:
        raise ValueError(f"Unexpected object columns in {path}: {raw['obj_columns']!r}")
    obj = raw["obj"]
    label = raw["label"]
    if obj.ndim != 3 or obj.shape[1:] != (23, 4):
        raise ValueError(f"Expected obj shape (events, 23, 4) in {path}, got {tuple(obj.shape)}")
    if label.ndim != 1 or label.shape[0] != obj.shape[0]:
        raise ValueError(f"label must contain one entry per obj row in {path}.")
    obj = torch.nan_to_num(obj, nan=0.0, posinf=0.0, neginf=0.0)
    slot_mask = obj[..., 0] > 0
    return obj.numpy(), slot_mask.numpy(), label


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
            print(
                f"    {step.depth}. {step.cut.to_text(model.observables)} "
                f"(prefix Gini gain={step.generation_score:.6g})"
            )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--signal", type=Path, default=ROOT / "data/anomalous.pt",
        help="Signal/mixture .pt file (default: repository data/anomalous.pt).",
    )
    parser.add_argument(
        "--background", type=Path, default=ROOT / "data/reference.pt",
        help="Reference/background .pt file (default: repository data/reference.pt).",
    )
    args = parser.parse_args(argv)
    signal_path = args.signal.expanduser().resolve()
    background_path = args.background.expanduser().resolve()
    signal, signal_slot_mask, signal_labels = load_obj_dataset(signal_path)
    background, background_slot_mask, background_labels = load_obj_dataset(background_path)

    print(f"Signal input: {signal_path}")
    print(f"Background input: {background_path}")
    print(f"Signal sample: {len(signal)} events; stored labels:",
          torch.unique(signal_labels, return_counts=True))
    print(f"Reference background: {len(background)} events; stored labels:",
          torch.unique(background_labels, return_counts=True))
    print("All rows in the signal input are treated as signal, regardless of stored process labels.")
    print("Scoring: weighted Gini gain of each rule versus its complement.")
    print("Gini is symmetric; each rule's enrichment direction is reported below.")

    model = ExistentialRuleList(
        OBSERVABLES,
        # Match run_ERL_reference_anomalous.py apart from the scoring callback.
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
