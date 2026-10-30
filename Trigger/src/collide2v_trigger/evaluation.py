"""Unweighted trigger acceptances and equivalent rates."""

import json
from pathlib import Path

import numpy as np

from .data import input_files, iter_batches


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def evaluate_rates(inputs, menu=None, triggers=None, *, collision_rate_khz=40000.0,
                   batch_size=10000, max_events=None, bits_output=None):
    if menu is None:
        raise ValueError("Supply a trigger menu.")
    names = menu.implementable_triggers if triggers is None else list(triggers)
    counts = dict.fromkeys([*names, "passes_any"], 0)
    saved = {name: [] for name in counts} if bits_output is not None else None
    total = 0
    for batch in iter_batches(inputs, batch_size, max_events, with_pairs=menu.requires_pairs(names)):
        bits = menu.trigger_bits(batch, names)
        total += batch.size
        for name, flags in bits.items():
            counts[name] += int(flags.sum())
            if saved is not None:
                saved[name].append(flags)
    if total == 0:
        raise ValueError("The inputs contain no events.")
    result = {"event_count": total, "collision_rate_khz": collision_rate_khz,
              "inputs": [str(path) for path in input_files(inputs)], "max_events": max_events,
              "object_selection": menu.object_selection(),
              "menu": menu.to_dict(),
              "thresholds": {name: menu.thresholds_for(name) for name in names},
              "triggers": {name: {"passed": count, "efficiency": count / total,
                                   "rate_khz": collision_rate_khz * count / total}
                           for name, count in counts.items()}}
    if saved is not None:
        path = Path(bits_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{name: np.concatenate(parts) for name, parts in saved.items()})
    return result


def print_rates(result):
    print(f"Events: {result['event_count']}    Collision rate: {result['collision_rate_khz']:g} kHz")
    print(f"{'Trigger':52} {'Passed':>9} {'Efficiency':>12} {'Rate [kHz]':>12}")
    for name, values in result["triggers"].items():
        print(f"{name:52} {values['passed']:9d} {values['efficiency']:12.6g} {values['rate_khz']:12.6g}")
