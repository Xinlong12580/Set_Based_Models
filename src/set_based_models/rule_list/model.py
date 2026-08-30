"""Public first-stage existential rule-list model."""
from __future__ import annotations
from collections.abc import Callable, Iterable, Sequence
from numbers import Integral
import numpy as np
from ..core.data import ObservableSpec, SetInput
from ..core.metrics import BinaryPerformance
from .generation import BeamRuleGenerator, RuleScorer, placeholder_rule_score
from .rules import ExistentialRule
from .statistics import RuleStatistics
from .unions import RuleUnion

class ExistentialRuleList:
    def __init__(self, observables: Sequence[ObservableSpec], *, max_rule_depth=4,
                 beam_width=100, max_candidate_rules=1000, max_thresholds=64,
                 minimum_count=1, min_signal_efficiency=0.0,
                 beam_signal_similarity_threshold=0.95,
                 beam_background_similarity_threshold=0.95,
                 beam_similarity_filter=True,
                 scoring_function: RuleScorer = placeholder_rule_score):
        self.observables = tuple(observables)
        if not self.observables or not all(isinstance(s, ObservableSpec) for s in self.observables):
            raise TypeError("observables must be a non-empty sequence of ObservableSpec")
        if len({s.name for s in self.observables}) != len(self.observables):
            raise ValueError("Observable names must be unique")
        if max_rule_depth < 1 or beam_width < 1 or minimum_count < 1:
            raise ValueError("depth, beam width, and minimum count must be positive")
        if max_candidate_rules is not None and max_candidate_rules < 1:
            raise ValueError("max_candidate_rules must be positive or None")
        if max_thresholds is not None and max_thresholds < 1:
            raise ValueError("max_thresholds must be positive or None")
        if not 0 <= min_signal_efficiency <= 1:
            raise ValueError("min_signal_efficiency must be in [0, 1]")
        for name, threshold in (
            ("beam_signal_similarity_threshold", beam_signal_similarity_threshold),
            ("beam_background_similarity_threshold", beam_background_similarity_threshold),
        ):
            if not 0 <= threshold <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if not isinstance(beam_similarity_filter, bool):
            raise TypeError("beam_similarity_filter must be a boolean")
        if not callable(scoring_function): raise TypeError("scoring_function must be callable")
        self.max_rule_depth, self.beam_width = max_rule_depth, beam_width
        self.max_candidate_rules, self.max_thresholds = max_candidate_rules, max_thresholds
        self.minimum_count, self.min_signal_efficiency = minimum_count, min_signal_efficiency
        self.beam_signal_similarity_threshold = beam_signal_similarity_threshold
        self.beam_background_similarity_threshold = beam_background_similarity_threshold
        self.beam_similarity_filter = beam_similarity_filter
        self.scoring_function = scoring_function
        self.candidate_rules_: list[ExistentialRule] = []
        self.selected_rules_: list[ExistentialRule] = []

    def fit(self, signal, background, *, signal_slot_mask=None,
            background_slot_mask=None, signal_weight=None, background_weight=None,
            scoring_function: Callable[[RuleStatistics], float] | None = None):
        signal_data = SetInput.from_arrays(signal, self.observables,
                                           slot_mask=signal_slot_mask,
                                           sample_weight=signal_weight)
        background_data = SetInput.from_arrays(background, self.observables,
                                               slot_mask=background_slot_mask,
                                               sample_weight=background_weight)
        if signal_data.X.shape[1:] != background_data.X.shape[1:]:
            raise ValueError("Signal and background must have identical slot/feature shape")
        generator = BeamRuleGenerator(
            max_rule_depth=self.max_rule_depth, beam_width=self.beam_width,
            max_candidate_rules=self.max_candidate_rules,
            max_thresholds=self.max_thresholds, minimum_count=self.minimum_count,
            min_signal_efficiency=self.min_signal_efficiency,
            scoring_function=scoring_function or self.scoring_function,
            beam_signal_similarity_threshold=self.beam_signal_similarity_threshold,
            beam_background_similarity_threshold=self.beam_background_similarity_threshold,
            beam_similarity_filter=self.beam_similarity_filter,
        )
        self.candidate_rules_ = generator.generate(signal_data, background_data)
        self.selected_rules_ = []
        return self

    def get_rule(self, rule_id):
        if not isinstance(rule_id, Integral) or not 0 <= rule_id < len(self.candidate_rules_):
            raise KeyError(f"Unknown rule {rule_id}")
        return self.candidate_rules_[int(rule_id)]

    def build_union(self, rule_indices: Iterable[int]) -> RuleUnion:
        """Build an executable OR of candidate rules selected by index."""
        indices = tuple(rule_indices)
        if len(set(indices)) != len(indices):
            raise ValueError("rule_indices must not contain duplicates")
        return RuleUnion(
            self.observables,
            tuple(self.get_rule(rule_index) for rule_index in indices),
        )

    def prune_candidates(
        self,
        *,
        signal_similarity_threshold=0.95,
        background_similarity_threshold=0.95,
    ) -> list[ExistentialRule]:
        """Return a score-ordered, greedily similarity-pruned rule list."""
        for name, threshold in (
            ("signal_similarity_threshold", signal_similarity_threshold),
            ("background_similarity_threshold", background_similarity_threshold),
        ):
            if not 0 <= threshold <= 1:
                raise ValueError(f"{name} must be in [0, 1]")

        retained: list[ExistentialRule] = []
        for rule in self.candidate_rules_:
            if rule.signal_selection is None or rule.background_selection is None:
                raise RuntimeError(
                    "Candidate rules do not contain training masks; fit the model "
                    "before pruning"
                )
            redundant = any(
                _jaccard(rule.signal_selection, previous.signal_selection)
                >= signal_similarity_threshold
                and _jaccard(
                    rule.background_selection,
                    previous.background_selection,
                ) >= background_similarity_threshold
                for previous in retained
            )
            if not redundant:
                retained.append(rule)
        return retained

    def evaluate_rule(self, rule_id, signal, background, *, signal_slot_mask=None,
                      background_slot_mask=None, signal_weight=None, background_weight=None):
        rule = self.get_rule(rule_id)
        return BinaryPerformance.from_selections(
            rule.predict(signal, slot_mask=signal_slot_mask),
            rule.predict(background, slot_mask=background_slot_mask),
            signal_weight=signal_weight, background_weight=background_weight,
        )


def _jaccard(left, right):
    intersection = np.count_nonzero(left & right)
    union = np.count_nonzero(left | right)
    return intersection / union if union else 1.0
