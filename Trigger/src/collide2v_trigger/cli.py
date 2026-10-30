"""Menu inspection, rate evaluation, and threshold calibration."""

import argparse
import json
from pathlib import Path

from .calibration import calibrate_menu
from .evaluation import evaluate_rates, print_rates, write_json
from .menu import TriggerMenu
from .data import iter_batches
from .erl import ERLTrigger, parse_erl_rule
from .presets import PHASE2_MENU


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("menu", help="List the triggers in a saved menu.")
    inspect.add_argument("menu", nargs="?", default="phase2", help="Menu JSON or the bundled phase2 preset.")
    convert = commands.add_parser("parse-erl", help="Export one ERL rule as an independent tunable trigger.")
    convert.add_argument("model", type=Path)
    convert.add_argument("--rule-id", type=int)
    convert.add_argument("--data", required=True, help="Reference PT file or chunk manifest from the training conversion.")
    convert.add_argument("--output", type=Path, required=True)
    convert.add_argument("--name")
    convert.add_argument("--target-rate-khz", type=float)
    convert.add_argument("--tune", type=int, nargs="+", help="Zero-based cut indices; other cuts remain fixed.")
    convert.add_argument("--slot-source", choices=("obj", "pair_obj"))
    convert.add_argument("--allow-fixed", action="store_true", help="Keep faithfully evaluable rules without tuning parameters.")
    for name in ("evaluate", "calibrate"):
        command = commands.add_parser(name)
        command.add_argument("inputs", nargs="*", help="PT files, quoted globs, or .chunks.json manifests.")
        command.add_argument("--triggers", nargs="+", help="Defaults to all supported triggers in the supplied menu.")
        command.add_argument("--menu", "--threshold-config", dest="menu", help="Menu JSON, calibrated menu JSON, or phase2.")
        command.add_argument("--erl-trigger", type=Path, action="append", default=[], help="Parsed trigger JSON; repeat for several.")
        command.add_argument("--no-isolation", dest="isolation", action="store_false", default=None)
        command.add_argument("--electron-iso-max", type=float)
        command.add_argument("--photon-iso-max", type=float)
        command.add_argument("--collision-rate-khz", type=float, default=40000.0)
        command.add_argument("--batch-size", type=int, default=10000)
        command.add_argument("--max-events", type=int)
        if name == "evaluate":
            command.add_argument("--output", type=Path, default=Path("results/rates.json"))
            command.add_argument("--bits", type=Path, help="Optional Boolean trigger arrays in an NPZ file.")
        else:
            command.add_argument("--output-dir", type=Path, default=Path("results/calibration"))
            command.add_argument("--scores", type=Path, help="Reuse saved critical_thresholds.npz and its menu settings.")
            command.add_argument("--target-rate-khz", type=float, help="Override the target for each selected seed.")
            command.add_argument("--method", choices=("bisection", "monotonic"), default="bisection")
            command.add_argument("--lower", type=float, default=0.0)
            command.add_argument("--upper", type=float, default=1000.0)
            command.add_argument("--score-tolerance", type=float, default=0.01, help="Per-event threshold precision, in GeV.")
            command.add_argument("--tolerance", type=float, default=0.01, help="Rate-fit threshold precision, in GeV.")
            command.add_argument("--spacing", type=float, default=0.25, help="Monotonic rate-scan spacing, in GeV.")
            command.add_argument("--reference-field", help="Use with a single selected seed.")
            command.add_argument("--scale-fields", nargs="+", help="Fields to scale together; others stay fixed.")
            command.add_argument("--export-unreachable-boundary", action="store_true")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "menu":
        source = PHASE2_MENU if args.menu == "phase2" else args.menu
        print(json.dumps(TriggerMenu.load(source).menu_summary(), indent=2))
        return
    if args.command == "parse-erl":
        batch = next(iter_batches(args.data, with_pairs=True))
        trigger = parse_erl_rule(args.model, batch, rule_id=args.rule_id, name=args.name,
                                 target_rate_khz=args.target_rate_khz, tune=args.tune, slot_source=args.slot_source,
                                 allow_fixed=args.allow_fixed)
        trigger.save(args.output)
        for cut in trigger.cut_actions:
            print(f"Cut {cut['index']}: {cut['observable']} -> {cut['action']} {cut.get('parameter', '')}")
        print(f"Saved {args.output}")
        return
    if not args.menu and not args.erl_trigger and not getattr(args, "scores", None):
        parser().error("Supply --menu MENU.json, --menu phase2, or --erl-trigger TRIGGER.json.")
    source = PHASE2_MENU if args.menu == "phase2" else args.menu
    menu = TriggerMenu(source, isolation=args.isolation,
                       electron_iso_max=args.electron_iso_max, photon_iso_max=args.photon_iso_max,
                       triggers=[ERLTrigger.load(path) for path in args.erl_trigger])
    common = {"triggers": args.triggers, "collision_rate_khz": args.collision_rate_khz,
              "batch_size": args.batch_size, "max_events": args.max_events}
    if args.command == "evaluate":
        result = evaluate_rates(args.inputs, menu, bits_output=args.bits, **common)
        write_json(args.output, result)
        print_rates(result)
        print(f"Saved {args.output}")
    else:
        result = calibrate_menu(args.inputs, None if args.scores else menu, scores_file=args.scores,
                                 output_dir=args.output_dir, target_rate_khz=args.target_rate_khz,
                                 method=args.method, lower=args.lower, upper=args.upper,
                                 score_tolerance=args.score_tolerance, tolerance=args.tolerance, spacing=args.spacing,
                                 reference_field=args.reference_field, scale_fields=args.scale_fields,
                                 export_unreachable_boundary=args.export_unreachable_boundary, **common)
        print_rates(result["evaluation"])
        print(f"Saved {args.output_dir}")


if __name__ == "__main__":
    main()
