"""Convert raw Parquet to event features and PT-ranked objects (no MET slot)."""

import argparse

from .convert import convert_raw_parquet_to_pt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, help="Raw Parquet filenames or remote file URLs.")
    parser.add_argument("--output", required=True, help="Local .pt file; existing outputs are overwritten with a warning.")
    parser.add_argument("--no-overwrite", action="store_true", help="Fail if the output or source sidecar already exists.")
    parser.add_argument("--label", type=int, required=True, help="Stored process label, for diagnostics only.")
    parser.add_argument("--max-events", type=int, help="Total row cap across inputs, after reading selected columns.")
    parser.add_argument("--without-native-event-info", action="store_true")
    parser.add_argument("--without-met-features", action="store_true",
                        help="Omit event-level MET, PT/MET and object-MET angles with validity flags; keep total object PT.")
    for singular, plural in (("electron", "electrons"), ("muon", "muons"),
                             ("photon", "photons"), ("jet", "jets")):
        parser.add_argument(f"--{singular}-thresholds", dest=plural, type=float, nargs="+",
                            help="Override PT thresholds in GeV; PT > 0 is always included.")
    args = parser.parse_args(argv)
    thresholds = {kind: getattr(args, kind) for kind in ("electrons", "muons", "photons", "jets")
                  if getattr(args, kind) is not None}
    try:
        target = convert_raw_parquet_to_pt(
            args.input, args.output, label=args.label, max_events=args.max_events, thresholds=thresholds,
            include_native_event_info=not args.without_native_event_info,
            include_met_features=not args.without_met_features,
            overwrite=not args.no_overwrite,
        )
    except (OSError, TypeError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(f"Saved {target}")


if __name__ == "__main__":
    main()
