"""Beam-search generation of existential conjunction rules."""
from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
import numpy as np
from ..core.data import ObservableKind, SetInput
from ..core.predicates import CategoryCut, Comparison, SlotCut, ThresholdCut
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
    @property
    def score(self): return self.steps[-1].generation_score

class BeamRuleGenerator:
    def __init__(self, *, max_rule_depth, beam_width, max_candidate_rules,
                 max_thresholds, minimum_count, min_signal_efficiency,
                 scoring_function, beam_signal_similarity_threshold=0.95,
                 beam_background_similarity_threshold=0.95,
                 beam_similarity_filter=True):
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

    def generate(self, signal: SetInput, background: SetInput):
        sw, bw = _weights(signal), _weights(background)
        root = _Candidate((), signal.slot_mask.copy(), background.slot_mask.copy(),
                          np.ones(signal.n_samples, bool), np.ones(background.n_samples, bool))
        beam, pool = [root], []
        for depth in range(1, self.max_rule_depth + 1):
            expanded, by_rule, by_selection = [], set(), set()
            for parent in beam:
                for cut in self._candidate_cuts(parent, signal, background):
                    key = _canonical_key((*[s.cut for s in parent.steps], cut))
                    if key in by_rule: continue
                    by_rule.add(key)
                    sm = parent.signal_mask & cut.evaluate(signal)
                    bm = parent.background_mask & cut.evaluate(background)
                    ss = np.count_nonzero(sm, axis=1) >= self.minimum_count
                    bs = np.count_nonzero(bm, axis=1) >= self.minimum_count
                    stats = _statistics(ss, bs, sw, bw, depth, self.minimum_count)
                    if stats.true_positive_rate < self.min_signal_efficiency: continue
                    score = float(self.scoring_function(stats))
                    if not np.isfinite(score): continue
                    signature = (np.packbits(ss).tobytes(), np.packbits(bs).tobytes())
                    if signature in by_selection: continue
                    by_selection.add(signature)
                    expanded.append(_Candidate(parent.steps + (RuleStep(depth, cut, stats, score),),
                                               sm, bm, ss, bs))
            expanded.sort(key=lambda c: c.score, reverse=True)
            beam = self._retain_diverse(expanded)
            pool.extend(beam)
            if not beam: break
        pool.sort(key=lambda c: c.score, reverse=True)
        if self.max_candidate_rules is not None: pool = pool[:self.max_candidate_rules]
        return [ExistentialRule(
                    i, signal.observables, c.steps, self.minimum_count,
                    c.signal_selection, c.background_selection,
                )
                for i, c in enumerate(pool)]

    def _retain_diverse(self, candidates):
        """Greedily keep strong candidates with distinct event selections."""
        if not self.beam_similarity_filter:
            return candidates[:self.beam_width]
        retained = []
        for candidate in candidates:
            redundant = any(
                _jaccard(candidate.signal_selection, previous.signal_selection)
                >= self.beam_signal_similarity_threshold
                and _jaccard(
                    candidate.background_selection,
                    previous.background_selection,
                ) >= self.beam_background_similarity_threshold
                for previous in retained
            )
            if redundant:
                continue
            retained.append(candidate)
            if len(retained) == self.beam_width:
                break
        return retained

    def _candidate_cuts(self, parent, signal, background):
        result: list[SlotCut] = []
        for index, spec in enumerate(signal.observables):
            if spec.kind in (ObservableKind.CONTINUOUS, ObservableKind.ORDINAL):
                values = np.concatenate((
                    signal.X[..., index][parent.signal_mask & signal.value_mask[..., index]],
                    background.X[..., index][parent.background_mask & background.value_mask[..., index]],
                ))
                unique = np.unique(values)
                if unique.size < 2: continue
                thresholds = unique[:-1] + np.diff(unique) / 2
                if self.max_thresholds is not None and thresholds.size > self.max_thresholds:
                    idx = np.linspace(0, thresholds.size - 1, self.max_thresholds, dtype=int)
                    thresholds = thresholds[np.unique(idx)]
                for threshold in thresholds:
                    result.extend((ThresholdCut(index, Comparison.GREATER, float(threshold)),
                                   ThresholdCut(index, Comparison.LESS_EQUAL, float(threshold))))
            else:
                assert spec.allowed_codes is not None
                result.extend(CategoryCut(index, {code}) for code in sorted(spec.allowed_codes))
        return tuple(result)

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

def _canonical_key(cuts):
    def key(cut):
        if isinstance(cut, ThresholdCut):
            return ("t", cut.observable_index, cut.comparison.value, cut.threshold)
        return ("c", cut.observable_index, tuple(sorted(cut.accepted_codes)))
    return tuple(sorted(key(cut) for cut in cuts))
