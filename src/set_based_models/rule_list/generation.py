"""Beam-search generation of existential conjunction rules."""
from __future__ import annotations
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import numpy as np
from ..core.candidates import event_candidate_cuts, observable_cuts
from ..core.data import SetInput
from ..core.predicates import Cut, EventCut, ThresholdCut
from .rules import ExistentialRule, RuleStep
from .statistics import RuleStatistics

RuleScorer = Callable[[RuleStatistics], float]

def placeholder_rule_score(statistics: RuleStatistics) -> float:
    raise NotImplementedError("Provide a rule scoring function before training")

@dataclass(slots=True)
class _Candidate:
    steps: tuple[RuleStep, ...]
    signal_mask: np.ndarray
    background_mask: np.ndarray
    signal_selection: np.ndarray
    background_selection: np.ndarray
    object_priority_progress: int = 0
    @property
    def score(self): return self.steps[-1].generation_score

class BeamRuleGenerator:
    def __init__(self, *, max_rule_depth, beam_width, max_candidate_rules,
                 max_thresholds, minimum_count, min_signal_efficiency,
                 scoring_function, beam_signal_similarity_threshold=0.95,
                 beam_background_similarity_threshold=0.95,
                 beam_similarity_filter=True, allow_validity_cuts=False,
                 object_cut_priority=()):
        if not isinstance(allow_validity_cuts, bool):
            raise TypeError("allow_validity_cuts must be a boolean")
        self.max_rule_depth = max_rule_depth
        self.beam_width = beam_width
        self.max_candidate_rules = max_candidate_rules
        self.max_thresholds = max_thresholds
        self.minimum_count = minimum_count
        self.min_signal_efficiency = min_signal_efficiency
        self.scoring_function = scoring_function
        self.beam_signal_similarity_threshold = beam_signal_similarity_threshold
        self.beam_background_similarity_threshold = beam_background_similarity_threshold
        self.beam_similarity_filter = beam_similarity_filter
        self.allow_validity_cuts = allow_validity_cuts
        self.object_cut_priority = _validate_object_cut_priority(object_cut_priority)

    def generate(self, signal: SetInput, background: SetInput):
        if (signal.observables != background.observables
                or signal.event_observables != background.event_observables):
            raise ValueError("Signal and background must use the same observable schemas")
        _validate_object_cut_priority(self.object_cut_priority, signal.observables,
                                      allow_validity_cuts=self.allow_validity_cuts)
        if self.object_cut_priority and not signal.n_slots:
            raise ValueError("object_cut_priority requires object tensor inputs with slots")
        indices = {spec.name: i for i, spec in enumerate(signal.observables)}
        priority_indices = tuple(indices[name] for name in self.object_cut_priority)
        sw, bw = _weights(signal), _weights(background)
        root = _Candidate((), signal.slot_mask.copy(), background.slot_mask.copy(),
                          np.ones(signal.n_samples, bool), np.ones(background.n_samples, bool))
        beam, pool = [root], []
        for depth in range(1, self.max_rule_depth + 1):
            expanded, by_rule, by_state = [], set(), set()
            for parent in beam:
                for cut in self._candidate_cuts(parent, signal, background):
                    key = _canonical_key((*[s.cut for s in parent.steps], cut))
                    if key in by_rule: continue
                    by_rule.add(key)
                    if isinstance(cut, EventCut):
                        sm, bm = parent.signal_mask, parent.background_mask
                        ss = parent.signal_selection & cut.evaluate(signal)
                        bs = parent.background_selection & cut.evaluate(background)
                    else:
                        sm = parent.signal_mask & cut.evaluate(signal)
                        bm = parent.background_mask & cut.evaluate(background)
                        ss = parent.signal_selection & (
                            np.count_nonzero(sm, axis=1) >= self.minimum_count
                        )
                        bs = parent.background_selection & (
                            np.count_nonzero(bm, axis=1) >= self.minimum_count
                        )
                    stats = _statistics(ss, bs, sw, bw, depth, self.minimum_count)
                    if stats.true_positive_rate < self.min_signal_efficiency: continue
                    score = float(self.scoring_function(stats))
                    if not np.isfinite(score): continue
                    # Event selections alone are not a sufficient beam-search
                    # state. Two rules can accept the same events through
                    # different slots, and those slots can behave differently
                    # when the rules are extended by another cut.
                    # Event cuts constrain selections without changing slots,
                    # so both parts of the state are needed for mixed rules.
                    progress = parent.object_priority_progress
                    if (not isinstance(cut, EventCut) and progress < len(priority_indices)
                            and cut.observable_index == priority_indices[progress]):
                        progress += 1
                    # Identical masks can still allow different next object
                    # features when one prefix has advanced further in priority.
                    signature = (progress, *(_mask_signature(mask) for mask in (ss, bs, sm, bm)))
                    if signature in by_state: continue
                    by_state.add(signature)
                    expanded.append(_Candidate(parent.steps + (RuleStep(depth, cut, stats, score),),
                                               sm, bm, ss, bs, progress))
            expanded.sort(key=lambda c: c.score, reverse=True)
            beam = self._retain_diverse(expanded)
            pool.extend(beam)
            if not beam: break
        pool.sort(key=lambda c: c.score, reverse=True)
        if self.max_candidate_rules is not None: pool = pool[:self.max_candidate_rules]
        return [ExistentialRule(
                    i, signal.observables, c.steps, self.minimum_count,
                    c.signal_selection, c.background_selection,
                    event_observables=signal.event_observables,
                )
                for i, c in enumerate(pool)]

    def _retain_diverse(self, candidates):
        """Greedily keep strong candidates with distinct event/slot states."""
        if not self.beam_similarity_filter:
            return candidates[:self.beam_width]
        retained = []
        for candidate in candidates:
            redundant = any(
                candidate.object_priority_progress == previous.object_priority_progress
                and _jaccard(candidate.signal_selection, previous.signal_selection)
                >= self.beam_signal_similarity_threshold
                and _jaccard(
                    candidate.background_selection,
                    previous.background_selection,
                ) >= self.beam_background_similarity_threshold
                and _jaccard(candidate.signal_mask, previous.signal_mask)
                >= self.beam_signal_similarity_threshold
                and _jaccard(candidate.background_mask, previous.background_mask)
                >= self.beam_background_similarity_threshold
                for previous in retained
            )
            if redundant:
                continue
            retained.append(candidate)
            if len(retained) == self.beam_width:
                break
        return retained

    def _candidate_cuts(self, parent, signal, background):
        result: list[Cut] = []
        object_specs = signal.observables if signal.n_slots else ()
        object_guards = {spec.validity_column for spec in object_specs
                         if spec.validity_column is not None}
        next_feature = (self.object_cut_priority[parent.object_priority_progress]
                        if parent.object_priority_progress < len(self.object_cut_priority) else None)
        for index, spec in enumerate(object_specs):
            if next_feature is not None and spec.name != next_feature:
                continue
            if not self.allow_validity_cuts and spec.name in object_guards:
                continue
            values = np.concatenate((
                signal.X[..., index][parent.signal_mask & signal.value_mask[..., index]
                                     & parent.signal_selection[:, None]],
                background.X[..., index][parent.background_mask & background.value_mask[..., index]
                                         & parent.background_selection[:, None]],
            ))
            result.extend(observable_cuts(index, spec, values, self.max_thresholds))
        result.extend(event_candidate_cuts(
            signal, background, parent.signal_selection, parent.background_selection,
            self.max_thresholds,
            excluded_columns=(() if self.allow_validity_cuts else
                              {spec.validity_column for spec in signal.event_observables
                               if spec.validity_column is not None}),
        ))
        return tuple(result)


def _validate_object_cut_priority(priority, observables=None, *, allow_validity_cuts=False):
    """Normalize an ordered list of names and check its object-scope schema."""
    if priority is None:
        priority = ()
    if isinstance(priority, (str, bytes)) or not isinstance(priority, Sequence):
        raise TypeError("object_cut_priority must be an ordered sequence of feature names")
    if not all(isinstance(name, str) and name.strip() for name in priority):
        raise ValueError("object_cut_priority must contain nonempty feature names")
    priority = tuple(name.strip() for name in priority)
    if len(set(priority)) != len(priority):
        raise ValueError("object_cut_priority must not contain duplicate feature names")
    if observables is not None:
        names = {spec.name for spec in observables}
        unknown = [name for name in priority if name not in names]
        if unknown:
            raise ValueError(f"Unknown object_cut_priority features: {unknown}")
        guards = {spec.validity_column for spec in observables if spec.validity_column is not None}
        if not allow_validity_cuts and any(name in guards for name in priority):
            raise ValueError("object_cut_priority cannot contain guard-only validity flags; "
                             "enable allow_validity_cuts to prioritize them")
    return priority

def _statistics(ss, bs, sw, bw, depth, minimum_count):
    return RuleStatistics(int(ss.sum()), ss.size, int(bs.sum()), bs.size,
                          float(sw[ss].sum()), float(sw.sum()),
                          float(bw[bs].sum()), float(bw.sum()), depth, minimum_count)

def _weights(data):
    return data.sample_weight if data.sample_weight is not None else np.ones(data.n_samples)


def _jaccard(left, right):
    intersection = np.count_nonzero(left & right)
    union = np.count_nonzero(left | right)
    return intersection / union if union else 1.0


def _mask_signature(mask):
    return np.packbits(mask, axis=None).tobytes()


def _canonical_key(cuts):
    def key(cut):
        scope = "event" if isinstance(cut, EventCut) else "slot"
        if isinstance(cut, EventCut):
            cut = cut.cut
        if isinstance(cut, ThresholdCut):
            return (scope, "t", cut.observable_index, cut.comparison.value, cut.threshold)
        return (scope, "c", cut.observable_index, tuple(sorted(cut.accepted_codes)))
    return tuple(sorted(key(cut) for cut in cuts))
