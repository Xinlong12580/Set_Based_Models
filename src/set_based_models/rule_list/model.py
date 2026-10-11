"""Public first-stage existential rule-list model."""
from __future__ import annotations
from collections.abc import Callable, Iterable, Sequence
from numbers import Integral
from ..core.data import ObservableSpec, SetInput, _validate_event_observables, _validate_validity_columns
from ..core.metrics import BinaryPerformance
from .generation import (
    BeamRuleGenerator, RuleScorer, _validate_object_cut_priority, placeholder_rule_score,
)
from .pruning import RulePruningResult, prune_rule_candidates
from .rules import ExistentialRule
from .statistics import RuleStatistics
from .unions import RuleUnion

class ExistentialRuleList:
    """Discover conjunctions of same-slot predicates and event predicates.

    Event features use ``event_observables`` and aligned 2D class arrays.
    Object tensors are optional: omit them for native event-only discovery.
    Every added cut counts toward ``max_rule_depth``, regardless of scope.
    Schema-linked validity flags are guard-only during discovery by default.
    ``allow_validity_cuts=True`` permits explicit flag cuts without disabling
    the automatic guards on their associated measurements.
    ``object_cut_priority`` orders object cuts: only the feature at the next
    list position is eligible until the list is exhausted. Repeated names
    require additional cuts on that feature, one per position. A direction
    can be supplied as 'feature:>' or (feature, '>'), using >, >=, < or <=.
    Plain names leave the direction unrestricted. Thresholds are optimized.
    Event cuts stay unrestricted, and incomplete prefixes remain candidates.
    Search states use packed masks; only the current parent's masks are dense.
    Completed-layer results retain event selections, not unused slot masks.
    This storage policy preserves candidate ordering, scores and similarity.
    """

    def __init__(self, observables: Sequence[ObservableSpec] = (), *, event_observables=(), max_rule_depth=4,
                 beam_width=100, max_candidate_rules=1000, max_thresholds=64,
                 minimum_count=1, min_signal_efficiency=0.0,
                 beam_signal_similarity_threshold=0.95,
                 beam_background_similarity_threshold=0.95,
                 beam_similarity_filter=True, allow_validity_cuts=False,
                 object_cut_priority: Sequence[str | tuple[str, str | None]] | None = (),
                 scoring_function: RuleScorer = placeholder_rule_score):
        self.observables = tuple(observables)
        self.event_observables = _validate_event_observables(event_observables)
        if not all(isinstance(s, ObservableSpec) for s in self.observables):
            raise TypeError("observables must be a sequence of ObservableSpec")
        if not self.observables and not self.event_observables:
            raise ValueError("Configure object observables, event observables, or both")
        if len({s.name for s in self.observables}) != len(self.observables):
            raise ValueError("Observable names must be unique")
        _validate_validity_columns(self.observables)
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
        if not isinstance(allow_validity_cuts, bool):
            raise TypeError("allow_validity_cuts must be a boolean")
        if not callable(scoring_function): raise TypeError("scoring_function must be callable")
        self.max_rule_depth, self.beam_width = max_rule_depth, beam_width
        self.max_candidate_rules, self.max_thresholds = max_candidate_rules, max_thresholds
        self.minimum_count, self.min_signal_efficiency = minimum_count, min_signal_efficiency
        self.beam_signal_similarity_threshold = beam_signal_similarity_threshold
        self.beam_background_similarity_threshold = beam_background_similarity_threshold
        self.beam_similarity_filter = beam_similarity_filter
        self.allow_validity_cuts = allow_validity_cuts
        self.object_cut_priority = _validate_object_cut_priority(
            object_cut_priority, self.observables, allow_validity_cuts=allow_validity_cuts,
        )
        self.scoring_function = scoring_function
        self.candidate_rules_: list[ExistentialRule] = []
        self.selected_rules_: list[ExistentialRule] = []
        self.pruning_result_: RulePruningResult | None = None
        self.input_mode_: str | None = None

    def fit(self, signal=None, background=None, *, signal_slot_mask=None,
            background_slot_mask=None, signal_weight=None, background_weight=None,
            signal_event_features=None, background_event_features=None,
            scoring_function: Callable[[RuleStatistics], float] | None = None,
            progress_callback: Callable[[dict], None] | None = None):
        """Fit object-only, event-only, or combined inputs.

        signal/background are optional 3D object arrays. Supply separate 2D
        signal_event_features/background_event_features for event observables.
        Both classes must use the same mode. Omitted objects create zero slots,
        not a dummy observable, and prevent all object-cut generation.
        An optional progress_callback receives plain search-status dictionaries
        (stage, depth, max_depth, parent/cut counts). No output is printed by
        the core model; callbacks must not mutate model inputs or configuration.
        """
        if progress_callback is not None and not callable(progress_callback):
            raise TypeError("progress_callback must be callable or None")
        signal_data = SetInput.from_arrays(signal, self.observables,
                                           slot_mask=signal_slot_mask,
                                           sample_weight=signal_weight,
                                           event_features=signal_event_features,
                                           event_observables=self.event_observables)
        background_data = SetInput.from_arrays(background, self.observables,
                                               slot_mask=background_slot_mask,
                                               sample_weight=background_weight,
                                               event_features=background_event_features,
                                               event_observables=self.event_observables)
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
            allow_validity_cuts=self.allow_validity_cuts,
            object_cut_priority=self.object_cut_priority,
        )
        self.candidate_rules_ = generator.generate(
            signal_data, background_data, progress_callback=progress_callback,
        )
        self.selected_rules_ = []
        self.pruning_result_ = None
        self.input_mode_ = ("event_only" if signal_data.n_slots == 0 else
                            "object_only" if not signal_data.event_observables else "combined")
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
            event_observables=self.event_observables,
        )

    def prune_candidates(
        self,
        *,
        signal_similarity_threshold=0.95,
        background_similarity_threshold=0.95,
        similarity_filter=True,
        stricter_score_improvement=None,
        max_reference_acceptance=None,
        require_complete_priority=False,
    ) -> list[ExistentialRule]:
        """Select final rules, preserving candidates and their original IDs.

        Defaults retain the existing Jaccard-only policy. Set
        ``stricter_score_improvement`` to a finite value to merge equivalent
        rules and require score(stricter) - score(looser) to exceed that
        threshold. Negative values allow a bounded score loss.
        Both tighter bounds and extra AND conditions are covered, with 1e-12
        roundoff tolerance. Event, object, and mixed rules are supported;
        object predicates share a slot scope and minimum counts are respected.

        ``require_complete_priority=True`` excludes candidates that have not
        consumed every object-priority entry, before all other pruning stages.
        Incomplete prefixes remain available during search and in candidate_rules_.
        An empty priority list imposes no completion restriction.
        Optional ``max_reference_acceptance`` is a weighted training fraction
        in [0, 1], applied next. Optional similarity pruning runs last and can
        remove looser rules; disable it to preserve the logical-only policy.
        Beam search is unchanged. Results and per-candidate audits are stored
        in ``selected_rules_`` and ``pruning_result_`` and reset on refitting.
        """
        result = prune_rule_candidates(
            self.candidate_rules_, similarity_filter=similarity_filter,
            signal_similarity_threshold=signal_similarity_threshold,
            background_similarity_threshold=background_similarity_threshold,
            stricter_score_improvement=stricter_score_improvement,
            max_reference_acceptance=max_reference_acceptance,
            required_object_cut_priority=self.object_cut_priority if require_complete_priority else (),
        )
        self.pruning_result_ = result
        self.selected_rules_ = list(result.retained)
        return list(result.retained)

    def evaluate_rule(self, rule_id, signal=None, background=None, *, signal_slot_mask=None,
                      background_slot_mask=None, signal_weight=None, background_weight=None,
                      signal_event_features=None, background_event_features=None):
        rule = self.get_rule(rule_id)
        return BinaryPerformance.from_selections(
            rule.predict(signal, slot_mask=signal_slot_mask, event_features=signal_event_features),
            rule.predict(background, slot_mask=background_slot_mask,
                         event_features=background_event_features),
            signal_weight=signal_weight, background_weight=background_weight,
        )
