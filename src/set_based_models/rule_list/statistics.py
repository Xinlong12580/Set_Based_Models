from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class RuleStatistics:
    signal_accepted: int
    signal_total: int
    background_accepted: int
    background_total: int
    signal_weight_accepted: float
    signal_weight_total: float
    background_weight_accepted: float
    background_weight_total: float
    rule_depth: int
    minimum_count: int

    @property
    def true_positive_rate(self):
        return self.signal_weight_accepted / self.signal_weight_total if self.signal_weight_total else 0.0
    @property
    def false_positive_rate(self):
        return self.background_weight_accepted / self.background_weight_total if self.background_weight_total else 0.0
    @property
    def precision(self):
        total = self.signal_weight_accepted + self.background_weight_accepted
        return self.signal_weight_accepted / total if total else 0.0
