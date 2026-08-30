"""Fit an ERL and compare its raw and similarity-pruned rule rankings."""

from pathlib import Path
import sys

import torch


HERE = Path(__file__).resolve().parent

# Allow ``python casual_tests/run_ERL.py`` before the package is installed.
sys.path.insert(0, str(HERE.parent / "src"))

from set_based_models import (  # noqa: E402
    ExistentialRuleList,
    ObservableSpec,
    TPRFPRRuleScore,
)


SIGNAL_SIMILARITY_THRESHOLD = 0.2
BACKGROUND_SIMILARITY_THRESHOLD = 0.2


def print_rules(title, rules, model) -> None:
    print(f"\n{title} ({len(rules)} rules):")
    for rank, rule in enumerate(rules, start=1):
        stats = rule.statistics
        print(f"\n[{rank}] Rule {rule.rule_id}")
        print(f"  score:    {rule.generation_score:.6g}")
        #print(f"  depth:    {rule.depth}")
        #print(f"  TPR:      {stats.true_positive_rate:.6f}")
        #print(f"  FPR:      {stats.false_positive_rate:.6f}")
        #print(f"  purity:   {stats.precision:.6f}")
        #print(
        #    "  accepted: "
        #    f"signal {stats.signal_accepted}/{stats.signal_total}, "
        #    f"background {stats.background_accepted}/{stats.background_total}"
        #)
        print(f"  rule:     {rule.to_text()}")
        print("  cut chain:")
        for step in rule.steps:
            print(
                f"    {step.depth}. {step.cut.to_text(model.observables)} "
                f"(prefix score={step.generation_score:.6g})"
            )


def main() -> None:
    raw = torch.load(HERE / "data" / "test.pt", map_location="cpu")
    obj = torch.nan_to_num(raw["obj"], nan=0.0, posinf=0.0, neginf=0.0)
    label = torch.nan_to_num(raw["label"], nan=0.0, posinf=0.0, neginf=0.0)

    # Keep the same class convention as run_EDT.py: TT is signal and QCD is
    # background. NumPy arrays have shape (sample, slot, observable).
    signal = obj[label == 2].detach().cpu().numpy()
    background = obj[label == 0].detach().cpu().numpy()

    observables = (
        ObservableSpec("pt", "continuous"),
        ObservableSpec("eta", "continuous"),
        ObservableSpec("phi", "continuous"),
        ObservableSpec(
            "type_id",
            "categorical",
            {0: "jet", 1: "muon", 2: "electron", 3: "photon", 4: "met"},
        ),
    )

    model = ExistentialRuleList(
        observables,
        # A deliberately modest search budget keeps this casual example fast
        # on the 20k-event test file while still producing many candidates.
        max_rule_depth=3,
        beam_width=40,
        max_candidate_rules=200,
        max_thresholds=32,
        minimum_count=1,
        beam_similarity_filter=True,
        scoring_function=TPRFPRRuleScore(lambda_fpr=1.0),
        beam_signal_similarity_threshold=SIGNAL_SIMILARITY_THRESHOLD,
        min_signal_efficiency=0.01,
        beam_background_similarity_threshold=BACKGROUND_SIMILARITY_THRESHOLD
    )
    model.fit(signal, background)

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
