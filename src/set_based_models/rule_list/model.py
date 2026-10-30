"""Public first-stage existential rule-list model."""
from __future__ import annotations
from collections.abc import Callable, Iterable, Sequence
from numbers import Integral
import numpy as np
from ..core.data import ObservableSpec, SetInput, _validate_event_observables, _validate_validity_columns
from ..core.metrics import BinaryPerformance
from ..core.predicates import EventCut
from .generation import (
    BeamRuleGenerator, RuleScorer, _validate_object_cut_priority, placeholder_rule_score,
)
from .pruning import RulePruningResult, prune_rule_candidates
from .rules import ExistentialRule
from .similarity import rule_jaccard_matrices
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
    Optional ``beam_novelty_weight`` rewards distinct signal-event selections
    during beam retention and candidate-pool capping. Zero preserves the
    score-only policy; intrinsic scores and final pruning are unchanged.
    Fit and recalculation cache event Jaccard matrices for every candidate.
    Saving retains these matrices, while event/slot masks remain transient.
    """

    def __init__(self, observables: Sequence[ObservableSpec] = (), *, event_observables=(), max_rule_depth=4,
                 beam_width=100, max_candidate_rules=1000, max_thresholds=64,
                 minimum_count=1, min_signal_efficiency=0.0,
                 beam_signal_similarity_threshold=0.95,
                 beam_background_similarity_threshold=0.95,
                 beam_similarity_filter=True, beam_novelty_weight=0.0, allow_validity_cuts=False,
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
        if not np.isfinite(beam_novelty_weight):
            raise ValueError("beam_novelty_weight must be finite")
        if not isinstance(allow_validity_cuts, bool):
            raise TypeError("allow_validity_cuts must be a boolean")
        if not callable(scoring_function): raise TypeError("scoring_function must be callable")
        self.max_rule_depth, self.beam_width = max_rule_depth, beam_width
        self.max_candidate_rules, self.max_thresholds = max_candidate_rules, max_thresholds
        self.minimum_count, self.min_signal_efficiency = minimum_count, min_signal_efficiency
        self.beam_signal_similarity_threshold = beam_signal_similarity_threshold
        self.beam_background_similarity_threshold = beam_background_similarity_threshold
        self.beam_similarity_filter = beam_similarity_filter
        self.beam_novelty_weight = beam_novelty_weight
        self.allow_validity_cuts = allow_validity_cuts
        self.object_cut_priority = _validate_object_cut_priority(
            object_cut_priority, self.observables, allow_validity_cuts=allow_validity_cuts,
        )
        self.scoring_function = scoring_function
        self.candidate_rules_: list[ExistentialRule] = []
        self.selected_rules_: list[ExistentialRule] = []
        self.pruning_result_: RulePruningResult | None = None
        self.pruning_configuration_: dict | None = None
        self.input_mode_: str | None = None
        self.statistics_source_: str | None = None
        self.merge_provenance_: dict | None = None
        self.self_pruning_provenance_: dict | None = None
        self.signal_jaccard_: np.ndarray | None = None
        self.background_jaccard_: np.ndarray | None = None
        self.jaccard_source_: str | None = None
        self.jaccard_blocks_: tuple[int, ...] = ()

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
            beam_novelty_weight=self.beam_novelty_weight,
            allow_validity_cuts=self.allow_validity_cuts,
            object_cut_priority=self.object_cut_priority,
        )
        candidates = generator.generate(
            signal_data, background_data, progress_callback=progress_callback,
        )
        signal_jaccard, background_jaccard = rule_jaccard_matrices(candidates)
        self.candidate_rules_ = candidates
        self.signal_jaccard_, self.background_jaccard_ = signal_jaccard, background_jaccard
        self.jaccard_source_, self.jaccard_blocks_ = "event_masks", (len(candidates),)
        self.selected_rules_ = []
        self.pruning_result_ = None
        self.pruning_configuration_ = None
        self.input_mode_ = ("event_only" if signal_data.n_slots == 0 else
                            "object_only" if not signal_data.event_observables else "combined")
        self.statistics_source_ = "training"
        self.merge_provenance_ = None
        self.self_pruning_provenance_ = None
        if scoring_function is not None:
            self.scoring_function = scoring_function
        return self

    def get_rule(self, rule_id):
        if not isinstance(rule_id, Integral) or not 0 <= rule_id < len(self.candidate_rules_):
            raise KeyError(f"Unknown rule {rule_id}")
        return self.candidate_rules_[int(rule_id)]

    def compute_jaccard_matrices(self):
        """Refresh the two candidate event-similarity matrices from current masks.

        Fit and recalculation do this automatically. Loaded models can use their
        saved matrices directly; rebuilding requires event masks. Returns the
        signal and background arrays, indexed by candidate rule ID.
        """
        signal, background = rule_jaccard_matrices(self.candidate_rules_)
        self.signal_jaccard_, self.background_jaccard_ = signal, background
        self.jaccard_source_, self.jaccard_blocks_ = "event_masks", (len(self.candidate_rules_),)
        return signal, background

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

    def predict(self, X=None, *, slot_mask=None, event_features=None, rule_ids=None):
        """Return event acceptance from an OR of the selected or specified rules.

        By default, final pruning must have been performed. Explicit rule_ids
        may select any candidates, including before pruning. An empty selection
        accepts no events. Object cuts in each rule share a slot, and counts
        are applied independently before combining rules with OR.
        """
        if rule_ids is None:
            if self.pruning_result_ is None:
                raise RuntimeError("Call prune_candidates() first, or specify rule_ids for prediction")
            rule_ids = [rule.rule_id for rule in self.selected_rules_]
        return self.build_union(rule_ids).predict(X, slot_mask=slot_mask, event_features=event_features)

    def evaluate(self, signal=None, background=None, *, rule_ids=None,
                 signal_slot_mask=None, background_slot_mask=None,
                 signal_weight=None, background_weight=None,
                 signal_event_features=None, background_event_features=None):
        """Measure the selected/specified OR on new signal and reference arrays."""
        rule_ids = None if rule_ids is None else tuple(rule_ids)
        return BinaryPerformance.from_selections(
            self.predict(signal, slot_mask=signal_slot_mask,
                         event_features=signal_event_features, rule_ids=rule_ids),
            self.predict(background, slot_mask=background_slot_mask,
                         event_features=background_event_features, rule_ids=rule_ids),
            signal_weight=signal_weight, background_weight=background_weight,
        )

    def recalculate_candidates(self, signal=None, background=None, *, signal_slot_mask=None,
                               background_slot_mask=None, signal_weight=None, background_weight=None,
                               signal_event_features=None, background_event_features=None, scoring_function=None):
        """Refresh every candidate's prefix statistics, scores and final masks.

        Uses fixed cuts and IDs, without search or candidate filtering. Inputs
        follow fit(), including optional objects, events, masks and weights.
        The scorer defaults to the model's current callable; loaded models need
        one supplied here or at load(). All updates commit together on success.
        Previous selected rules and pruning records are cleared. Returns self.

        This replaces active statistics, including generation_score; save the
        training model first if its original measurements should be retained.
        evaluate() and evaluate_rule() remain read-only.
        """
        scorer = self.scoring_function if scoring_function is None else scoring_function
        if not callable(scorer):
            raise TypeError("scoring_function must be callable")
        if (signal is None or background is None) and any(
                not isinstance(cut, EventCut) for rule in self.candidate_rules_ for cut in rule.cuts):
            raise ValueError("Object tensors are required to recalculate object-level candidates")
        signal_data = SetInput.from_arrays(signal, self.observables, slot_mask=signal_slot_mask,
                                          sample_weight=signal_weight, event_features=signal_event_features,
                                          event_observables=self.event_observables)
        background_data = SetInput.from_arrays(background, self.observables, slot_mask=background_slot_mask,
                                              sample_weight=background_weight, event_features=background_event_features,
                                              event_observables=self.event_observables)
        from .refinement import recalculate_rules
        recalculated = recalculate_rules(self.candidate_rules_, signal_data, background_data, scorer)
        signal_jaccard, background_jaccard = rule_jaccard_matrices(recalculated)
        self.candidate_rules_ = recalculated
        self.signal_jaccard_, self.background_jaccard_ = signal_jaccard, background_jaccard
        self.jaccard_source_, self.jaccard_blocks_ = "event_masks", (len(recalculated),)
        self.scoring_function = scorer
        self.statistics_source_ = "recalculated"
        self.selected_rules_ = []
        self.pruning_result_ = None
        self.pruning_configuration_ = None
        return self

    def self_prune_candidates(self, signal=None, background=None, *, signal_slot_mask=None,
                              background_slot_mask=None, signal_weight=None, background_weight=None,
                              signal_event_features=None, background_event_features=None, scoring_function=None,
                              stricter_score_improvement=0.0, max_reference_acceptance=None,
                              require_complete_priority=False, rule_ids=None, compute_jaccard=True):
        """Return a new validation-measured candidate pool before final pruning.

        For each source rule, evaluate every nonempty cut subset, including
        replacing guarded measurements with their True validity flags. Cuts
        retain their order, same-slot semantics and minimum object counts.
        Locally merge equivalents and remove stricter variants whose score
        improvement does not exceed stricter_score_improvement. Signed gaps
        work as in prune_candidates(); None disables local score pruning.
        Rate and optional source-priority
        requirements are applied before local score comparisons.

        All source candidates are used by default, regardless of any previous
        final selection; rule_ids selects a subset. Surviving variants are
        pooled, logically equivalent copies deduplicated, and IDs reassigned.
        Different source priorities/progress remain distinct. The returned
        model has fresh prefix statistics, scores, masks and (by default)
        full Jaccard matrices, but no final selection. Call prune_candidates()
        on it separately. compute_jaccard=False avoids dense matrix storage;
        retained masks still allow in-memory similarity pruning.

        No cuts or thresholds are optimized, TRUE is excluded, and no search
        budgets/efficiency thresholds are applied. Expansion is exponential
        in rule depth (two choices per cut, three per guarded measurement).
        Inputs and scorer follow recalculate_candidates(). The source model
        stays unchanged, including its previous selection and statistics.
        """
        from .refinement import self_prune_candidates
        return self_prune_candidates(
            self, signal, background, signal_slot_mask=signal_slot_mask,
            background_slot_mask=background_slot_mask, signal_weight=signal_weight,
            background_weight=background_weight, signal_event_features=signal_event_features,
            background_event_features=background_event_features, scoring_function=scoring_function,
            stricter_score_improvement=stricter_score_improvement,
            max_reference_acceptance=max_reference_acceptance,
            require_complete_priority=require_complete_priority, rule_ids=rule_ids,
            compute_jaccard=compute_jaccard,
        )

    def to_dict(self):
        """Export a versioned JSON-compatible artifact without event masks."""
        from .serialization import model_to_dict
        return model_to_dict(self)

    @classmethod
    def from_dict(cls, value, *, scoring_function=placeholder_rule_score):
        """Restore rules and fitted statistics; no scorer is executed or imported.

        Prediction requires no scorer. Supply scoring_function here or on a
        later fit() or recalculate_candidates() call. Event masks are not
        restored. Saved Jaccard matrices support repruning the recorded sample;
        recalculate to measure similarities and statistics on new arrays.
        """
        from .serialization import model_from_dict
        return model_from_dict(value, cls, scoring_function=scoring_function)

    def save(self, filename, *, overwrite=True):
        """Atomically save JSON; warn before replacing an existing model file."""
        from .serialization import save_model
        return save_model(self, filename, overwrite=overwrite)

    @classmethod
    def load(cls, filename, *, scoring_function=placeholder_rule_score):
        """Load a core ERL JSON artifact (not a runner's quantitative report)."""
        from .serialization import load_model
        return load_model(filename, cls, scoring_function=scoring_function)

    @classmethod
    def merge(cls, models, *, scoring_function=placeholder_rule_score, source_names=None):
        """Pool all candidates from compatible models without changing the sources.

        Ordered object/event schemas must match, including categories and guards.
        Candidates receive sequential IDs; cuts, minimum counts and source
        provenance are preserved. There is no candidate cap or deduplication.
        Source scores remain available but cannot be compared until refreshed
        together with recalculate_candidates(). Selection and masks are cleared.
        Source search settings are provenance, not combined search restrictions.
        Available Jaccard matrices form diagonal blocks; cross-model entries
        are assumed zero until recalculation measures the complete pool.
        Optional source_names labels the inputs; load_many() uses filenames.
        """
        from .merging import merge_models
        return merge_models(models, cls, scoring_function=scoring_function, source_names=source_names)

    @classmethod
    def load_many(cls, filenames, *, scoring_function=placeholder_rule_score):
        """Load model files and merge their full candidate pools for recalculation."""
        filenames = list(filenames)
        models = [cls.load(filename) for filename in filenames]
        return cls.merge(models, scoring_function=scoring_function,
                         source_names=[str(filename) for filename in filenames])

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
        Merged candidates use their individual source priorities for completion.
        Merged pools require recalculation before any final pruning.
        Optional ``max_reference_acceptance`` is a weighted reference fraction
        in [0, 1], applied next. Optional similarity pruning runs last and can
        remove looser rules; disable it to preserve the logical-only policy.
        Beam search is unchanged. Results and per-candidate audits are stored
        in ``selected_rules_`` and ``pruning_result_`` and reset on refitting
        or recalculation. Pruning uses the current statistics/scores/masks,
        which can come from training or recalculate_candidates(). Saved event
        Jaccard matrices allow repruning after loading without restoring masks.
        """
        if self.statistics_source_ == "merged":
            raise RuntimeError("Call recalculate_candidates() before pruning merged candidates")
        priorities = None
        if require_complete_priority and self.merge_provenance_ is not None:
            sources = self.merge_provenance_["sources"]
            priorities = {
                rule_id: sources[origin["source_index"]]["configuration"]["object_cut_priority"]
                for rule_id, origin in enumerate(self.merge_provenance_["rules"])
            }
        result = prune_rule_candidates(
            sorted(self.candidate_rules_, key=lambda rule: (-rule.generation_score, rule.rule_id)),
            similarity_filter=similarity_filter,
            signal_similarity_threshold=signal_similarity_threshold,
            background_similarity_threshold=background_similarity_threshold,
            stricter_score_improvement=stricter_score_improvement,
            max_reference_acceptance=max_reference_acceptance,
            required_object_cut_priority=self.object_cut_priority if require_complete_priority else (),
            required_object_cut_priority_by_rule=priorities,
            signal_jaccard=self.signal_jaccard_, background_jaccard=self.background_jaccard_,
        )
        self.pruning_result_ = result
        self.selected_rules_ = list(result.retained)
        self.pruning_configuration_ = {
            "signal_similarity_threshold": signal_similarity_threshold,
            "background_similarity_threshold": background_similarity_threshold,
            "similarity_filter": similarity_filter,
            "stricter_score_improvement": stricter_score_improvement,
            "max_reference_acceptance": max_reference_acceptance,
            "require_complete_priority": require_complete_priority,
        }
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
