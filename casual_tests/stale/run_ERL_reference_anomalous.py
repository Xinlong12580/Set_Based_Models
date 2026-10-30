"""Run with ``python casual_tests/stale/run_ERL_reference_anomalous.py``.

Fits the original ERL settings to all anomalous.pt rows as signal and all
reference.pt rows as background. Input files are read from the repository's
data/ directory, independent of the shell's current directory.
"""

from pathlib import Path
import sys

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from set_based_models import (  # noqa: E402
    ExistentialRuleList,
    ObservableSpec,
    TPRFPRRuleScore,
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


def load_obj_dataset(filename: str):
    path = ROOT / "data" / filename
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
        print(f"\n[{rank}] Rule {rule.rule_id}")
        print(f"  score:    {rule.generation_score:.6g}")
        print(f"  rule:     {rule.to_text()}")
        print("  cut chain:")
        for step in rule.steps:
            print(
                f"    {step.depth}. {step.cut.to_text(model.observables)} "
                f"(prefix score={step.generation_score:.6g})"
            )


def main() -> None:
    signal, signal_slot_mask, signal_labels = load_obj_dataset("anomalous.pt")
    background, background_slot_mask, background_labels = load_obj_dataset("reference.pt")

    print(f"Anomalous signal sample: {len(signal)} events; stored labels:",
          torch.unique(signal_labels, return_counts=True))
    print(f"MinBias reference background: {len(background)} events; stored labels:",
          torch.unique(background_labels, return_counts=True))
    print("The full anomalous.pt mixture is treated as signal, including its MinBias events.")

    model = ExistentialRuleList(
        OBSERVABLES,
        # Keep the original casual_tests/stale/run_ERL.py search configuration.
        max_rule_depth=3,
        beam_width=40,
        max_candidate_rules=200,
        max_thresholds=32,
        minimum_count=1,
        beam_similarity_filter=True,
        scoring_function=TPRFPRRuleScore(lambda_fpr=1.0),
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
        "Top unpruned rules (higher score is better)",
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
