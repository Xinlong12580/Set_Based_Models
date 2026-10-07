"""Run both models on synthetic data requiring event and slot cuts.

Run from the repository root: python casual_tests/run_mixed_observables.py
Only NumPy is required; no external dataset is loaded.
"""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from set_based_models import (
    ExistentialDecisionTree, ExistentialRuleList, ObservableSpec,
    TPRFPRRuleScore, gini_gain,
)


def main():
    observables = (ObservableSpec("pt", "continuous"),)
    event_observables = (ObservableSpec("HT", "continuous"),)
    signal = np.array([[[80.], [0.]], [[90.], [0.]]])
    background = np.array([[[80.], [0.]], [[10.], [0.]]])
    signal_events = np.array([[800.], [900.]])
    background_events = np.array([[100.], [800.]])
    mask = np.array([[True, False], [True, False]])
    fit_args = dict(
        signal_event_features=signal_events,
        background_event_features=background_events,
        signal_slot_mask=mask,
        background_slot_mask=mask,
    )

    tree = ExistentialDecisionTree(
        observables, event_observables=event_observables,
        max_depth=2, gain_function=gini_gain,
    ).fit(signal, background, **fit_args)
    leaf = int(tree.apply(signal, event_features=signal_events, slot_mask=mask)[0])
    print("Tree path for signal:")
    print(tree.cut_chain_text(leaf))
    performance = tree.evaluate(signal, background, **fit_args)
    print(f"Tree TPR={performance.true_positive_rate:g}, FPR={performance.false_positive_rate:g}")

    model = ExistentialRuleList(
        observables, event_observables=event_observables,
        max_rule_depth=2, beam_width=20, scoring_function=TPRFPRRuleScore(),
    ).fit(signal, background, **fit_args)
    rule = model.get_rule(0)
    print("Best rule:", rule.to_text())
    union = model.build_union([0])
    print("Signal:", union.predict(signal, event_features=signal_events, slot_mask=mask))
    print("Background:", union.predict(background, event_features=background_events, slot_mask=mask))


if __name__ == "__main__":
    main()
