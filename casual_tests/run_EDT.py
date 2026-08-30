"""Minimal example: fit an EDT and inspect one leaf's learned cut chain."""

from pathlib import Path
import sys
import torch
import numpy as np

# Allow ``python run_EDT.py`` before the package has been installed.
sys.path.insert(0, str(Path(__file__).resolve().parent / "../src"))

from set_based_dt import ExistentialDecisionTree, ObservableSpec, gini_gain


def main() -> None:
    # Tensor axes are (sample, slot, observable). These small arrays only make
    # the example runnable; replace them with the real signal/background data.

    raw = torch.load("data/test.pt", map_location="cpu")
    obj = torch.nan_to_num(raw["obj"], nan=0.0, posinf=0.0, neginf=0.0)
    label = torch.nan_to_num(raw["label"], nan=0.0, posinf=0.0, neginf=0.0)
    print(torch.unique(raw["obj"][:,:,3]))
    mask_QCD = label == 0
    QCD_subset = obj[mask_QCD]
    QCD_subset = QCD_subset.detach().cpu().numpy()

    mask_TT = label == 2
    TT_subset = obj[mask_TT]
    TT_subset = TT_subset.detach().cpu().numpy()

    observables = (
        ObservableSpec("pt", "continuous"),
        ObservableSpec("eta", "continuous"),
        ObservableSpec("phi", "continuous"),
        ObservableSpec("type_id", "categorical", {0:"jet", 1:"muon", 2:"electron", 3:"photon", 4:"met"}),
    )

    signal = TT_subset
    background = QCD_subset

    tree = ExistentialDecisionTree(
        observables,
        max_depth=6,
        gain_function=gini_gain,
        min_samples_leaf=20,
    )
    tree.fit(signal, background)

    leaves = [node for node in tree.nodes_ if node.is_leaf]
    #selected_leaf = max(leaves, key=lambda node: node.score or 0.0)
    nodes = []
    scores = []
    cuts = []
    for selected_leaf in leaves:

        print(f"Selected leaf: {selected_leaf.node_id}")
        print(f"Signal score: {selected_leaf.score:.3f}")
        print("\nExact root-to-leaf event path:")
        print(tree.cut_chain_text(selected_leaf.node_id))
        nodes.append(selected_leaf.node_id)
        scores.append(selected_leaf.score)
        cuts.append(tree.cut_chain_text(selected_leaf.node_id))
        print("\nPositive-branch slot cut chain:")
        slot_chain = tree.slot_cut_chain(selected_leaf.node_id)
        if slot_chain:
            for cut in slot_chain:
                print(f"  {cut.to_text(tree.observables)}")
        else:
            print("  (no positive slot cuts)")
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    ranked_scores = [scores[i] for i in order]
    ranked_cuts = [cuts[i] for i in order]
    for i in range(len(ranked_scores)): print("\n", ranked_cuts[i], ranked_scores[i])

if __name__ == "__main__":
    main()
