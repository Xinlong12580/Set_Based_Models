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
    """Dense working state for the single parent currently being expanded."""
    steps: tuple[RuleStep, ...]
    signal_mask: np.ndarray
    background_mask: np.ndarray
    signal_selection: np.ndarray
    background_selection: np.ndarray
    object_priority_progress: int = 0
    @property
    def score(self): return self.steps[-1].generation_score


@dataclass(slots=True)
class _PackedCandidate:
    """Expansion state shared with the duplicate-detection key, not copied."""
    steps: tuple[RuleStep, ...]
    state: tuple[int, bytes, bytes, bytes, bytes]

    @property
    def score(self): return self.steps[-1].generation_score

    @property
    def object_priority_progress(self): return self.state[0]

    def unpack(self, signal_shape, background_shape):
        _, ss, bs, sm, bm = self.state
        return _Candidate(
            self.steps, _unpack_mask(sm, signal_shape), _unpack_mask(bm, background_shape),
            _unpack_mask(ss, (signal_shape[0],)), _unpack_mask(bs, (background_shape[0],)),
            self.object_priority_progress,
        )


@dataclass(slots=True)
class _RuleCandidate:
    """Completed-layer result: slot masks are unnecessary for final rules."""
    steps: tuple[RuleStep, ...]
    signal_selection: bytes
    background_selection: bytes

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

    def generate(self, signal: SetInput, background: SetInput, *, progress_callback=None):
        if progress_callback is not None and not callable(progress_callback):
            raise TypeError("progress_callback must be callable or None")
        if (signal.observables != background.observables
                or signal.event_observables != background.event_observables):
            raise ValueError("Signal and background must use the same observable schemas")
        _validate_object_cut_priority(self.object_cut_priority, signal.observables,
                                      allow_validity_cuts=self.allow_validity_cuts)
        if self.object_cut_priority and not signal.n_slots:
            raise ValueError("object_cut_priority requires object tensor inputs with slots")
        indices = {spec.name: i for i, spec in enumerate(signal.observables)}
        priority_indices = tuple(indices[_priority_entry(entry)[0]] for entry in self.object_cut_priority)
        sw, bw = _weights(signal), _weights(background)
        root = _PackedCandidate((), (
            0, _mask_signature(np.ones(signal.n_samples, bool)),
            _mask_signature(np.ones(background.n_samples, bool)),
            _mask_signature(signal.slot_mask), _mask_signature(background.slot_mask),
        ))
        beam, pool = [root], []
        del root
        evaluated = 0
        status = {"max_depth": self.max_rule_depth, "evaluated_cuts": 0}

        def emit(stage, **updates):
            if progress_callback is not None:
                status.update(updates)
                progress_callback({**status, "stage": stage})

        for depth in range(1, self.max_rule_depth + 1):
            emit("depth_started", depth=depth, parents_total=len(beam), parents_completed=0,
                 cuts_completed=0, cuts_total=0)
            expanded, by_rule, by_state = [], set(), set()
            for parent_index, packed_parent in enumerate(beam, 1):
                parent = packed_parent.unpack(signal.slot_mask.shape, background.slot_mask.shape)
                emit("generating_cuts", parents_completed=parent_index - 1,
                     cuts_completed=0, cuts_total=0)
                cuts = self._candidate_cuts(parent, signal, background)
                for cut_index, cut in enumerate(cuts, 1):
                    if progress_callback is not None:
                        emit("evaluating_cuts", cuts_completed=cut_index - 1,
                             cuts_total=len(cuts), evaluated_cuts=evaluated)
                    key = _canonical_key((*[s.cut for s in parent.steps], cut))
                    if key in by_rule: continue
                    by_rule.add(key)
                    evaluated += 1
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
                    # Event cuts leave slots unchanged: reuse immutable packed
                    # parent slots instead of repacking/storing another copy.
                    slot_bits = (packed_parent.state[3:] if isinstance(cut, EventCut) else
                                 (_mask_signature(sm), _mask_signature(bm)))
                    signature = (progress, _mask_signature(ss), _mask_signature(bs), *slot_bits)
                    if signature in by_state: continue
                    by_state.add(signature)
                    expanded.append(_PackedCandidate(
                        parent.steps + (RuleStep(depth, cut, stats, score),), signature,
                    ))
                # No dense masks survive in expansions, old beam states or pool.
                # Clear the last evaluated cut as well as its working parent.
                sm = bm = ss = bs = None
                del parent
                emit("parent_finished", parents_completed=parent_index,
                     cuts_completed=len(cuts), cuts_total=len(cuts), evaluated_cuts=evaluated)
            beam.clear()  # All old parents have been expanded; their slots can go.
            del packed_parent
            signature = slot_bits = None
            by_rule.clear()
            by_state.clear()
            emit("filtering_beam", filter_completed=0, filter_total=len(expanded), retained=0)
            expanded.sort(key=lambda c: c.score, reverse=True)
            filter_progress = (None if progress_callback is None else
                               lambda completed, total, retained: emit(
                                   "filtering_beam", filter_completed=completed,
                                   filter_total=total, retained=retained))
            beam = self._retain_diverse(expanded, progress_callback=filter_progress)
            pool.extend(_RuleCandidate(candidate.steps, candidate.state[1], candidate.state[2])
                        for candidate in beam)
            expanded.clear()  # Discard non-retained masks before the next layer.
            emit("depth_finished", retained=len(beam), candidate_rules=len(pool))
            if not beam: break
        beam.clear()  # Final-layer slot states are never needed again.
        pool.sort(key=lambda c: c.score, reverse=True)
        if self.max_candidate_rules is not None: pool = pool[:self.max_candidate_rules]
        emit("building_rules", candidate_rules=len(pool))
        rules = [ExistentialRule(
                    i, signal.observables, c.steps, self.minimum_count,
                    _unpack_mask(c.signal_selection, (signal.n_samples,)),
                    _unpack_mask(c.background_selection, (background.n_samples,)),
                    event_observables=signal.event_observables,
                )
                for i, c in enumerate(pool)]
        emit("finished", candidate_rules=len(rules))
        return rules

    def _retain_diverse(self, candidates, *, progress_callback=None):
        """Greedily keep strong candidates with distinct event/slot states."""
        if not self.beam_similarity_filter:
            return candidates[:self.beam_width]
        retained = []
        processed = 0
        for processed, candidate in enumerate(candidates, 1):
            if progress_callback is not None:
                progress_callback(processed - 1, len(candidates), len(retained))
            redundant = any(
                candidate.object_priority_progress == previous.object_priority_progress
                and _packed_jaccard(candidate.state[1], previous.state[1])
                >= self.beam_signal_similarity_threshold
                and _packed_jaccard(candidate.state[2], previous.state[2])
                >= self.beam_background_similarity_threshold
                and _packed_jaccard(candidate.state[3], previous.state[3])
                >= self.beam_signal_similarity_threshold
                and _packed_jaccard(candidate.state[4], previous.state[4])
                >= self.beam_background_similarity_threshold
                for previous in retained
            )
            if redundant:
                continue
            retained.append(candidate)
            if len(retained) == self.beam_width:
                break
        if progress_callback is not None:
            progress_callback(processed, len(candidates), len(retained))
        return retained

    def _candidate_cuts(self, parent, signal, background):
        result: list[Cut] = []
        object_specs = signal.observables if signal.n_slots else ()
        object_guards = {spec.validity_column for spec in object_specs
                         if spec.validity_column is not None}
        next_feature, direction = (_priority_entry(self.object_cut_priority[parent.object_priority_progress])
                                   if parent.object_priority_progress < len(self.object_cut_priority)
                                   else (None, None))
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
            cuts = observable_cuts(index, spec, values, self.max_thresholds)
            if direction is None:
                result.extend(cuts)
            else:
                base_direction = ">" if direction in (">", ">=") else "<="
                result.extend(ThresholdCut(index, direction, cut.threshold) for cut in cuts
                              if isinstance(cut, ThresholdCut) and cut.comparison.value == base_direction)
        result.extend(event_candidate_cuts(
            signal, background, parent.signal_selection, parent.background_selection,
            self.max_thresholds,
            excluded_columns=(() if self.allow_validity_cuts else
                              {spec.validity_column for spec in signal.event_observables
                               if spec.validity_column is not None}),
        ))
        return tuple(result)


def _priority_entry(entry):
    """Read a feature name, 'feature:operator', or (feature, operator) pair."""
    if isinstance(entry, str):
        name, separator, direction = entry.rpartition(":")
        if not separator:
            name, direction = entry, None
    elif isinstance(entry, Sequence) and len(entry) == 2:
        name, direction = entry
    else:
        raise ValueError("Priority entries must be feature names or (feature, direction) pairs")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("object_cut_priority must contain nonempty feature names")
    if isinstance(direction, str):
        direction = direction.strip()
    if direction is not None and direction not in (">", ">=", "<", "<="):
        raise ValueError("Priority directions must be >, >=, <, or <=")
    return name.strip(), direction


def _priority_matches(cut, entry, observables):
    name, direction = _priority_entry(entry)
    return (not isinstance(cut, EventCut) and observables[cut.observable_index].name == name
            and (direction is None or isinstance(cut, ThresholdCut) and cut.comparison.value == direction))


def _object_priority_progress(cuts, priority, observables):
    """Count consumed priority entries from the original ordered cut history."""
    progress = 0
    for cut in cuts:
        if progress < len(priority) and _priority_matches(cut, priority[progress], observables):
            progress += 1
    return progress


def _validate_object_cut_priority(priority, observables=None, *, allow_validity_cuts=False):
    """Normalize an ordered cut sequence, including repeats and directions."""
    if priority is None:
        priority = ()
    if isinstance(priority, (str, bytes)) or not isinstance(priority, Sequence):
        raise TypeError("object_cut_priority must be an ordered sequence of feature entries")
    entries = tuple(_priority_entry(entry) for entry in priority)
    priority = tuple(name if direction is None else f"{name}:{direction}" for name, direction in entries)
    if observables is not None:
        names = {spec.name for spec in observables}
        unknown = [name for name, _ in entries if name not in names]
        if unknown:
            raise ValueError(f"Unknown object_cut_priority features: {unknown}")
        guards = {spec.validity_column for spec in observables if spec.validity_column is not None}
        if not allow_validity_cuts and any(name in guards for name, _ in entries):
            raise ValueError("object_cut_priority cannot contain guard-only validity flags; "
                             "enable allow_validity_cuts to prioritize them")
        specs = {spec.name: spec for spec in observables}
        if any(direction is not None and not specs[name].allows_threshold for name, direction in entries):
            raise ValueError("Priority cut directions require a numerical object feature")
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


def _unpack_mask(packed, shape):
    """Restore exact dimensions, excluding packbits' zero-padded trailing bits."""
    bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), count=int(np.prod(shape)))
    return bits.reshape(shape).view(np.bool_)


_BYTE_POPCOUNT = np.array([value.bit_count() for value in range(256)], dtype=np.uint8)


def _packed_jaccard(left, right):
    """Same Jaccard as dense masks, using byte operations and exact bit counts."""
    left, right = np.frombuffer(left, dtype=np.uint8), np.frombuffer(right, dtype=np.uint8)
    intersection = int(_BYTE_POPCOUNT[left & right].sum(dtype=np.int64))
    union = int(_BYTE_POPCOUNT[left | right].sum(dtype=np.int64))
    return intersection / union if union else 1.0


def _canonical_key(cuts):
    def key(cut):
        scope = "event" if isinstance(cut, EventCut) else "slot"
        if isinstance(cut, EventCut):
            cut = cut.cut
        if isinstance(cut, ThresholdCut):
            return (scope, "t", cut.observable_index, cut.comparison.value, cut.threshold)
        return (scope, "c", cut.observable_index, tuple(sorted(cut.accepted_codes)))
    return tuple(sorted(key(cut) for cut in cuts))
