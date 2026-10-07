"""Run Gini EDT on event_features only, using fully masked dummy object slots.

Example (from the repository root)::

    python casual_tests/run_EDT_event_only.py \
        --signal data/ZZ_event_features/anomalous.pt \
        --background data/ZZ_event_features/reference.pt \
        --report reports/EDT_ZZ_event_only.txt

The signal input may be a background-contaminated anomaly region. Stored
process labels are used only for reporting, never to choose the training class.
The separate native_event_info block is not used. Training settings match the
existing quantitative EDT runner; output is a plot-compatible JSON text report.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from casual_tests._event_only_reporting import run_edt_cli  # noqa: E402


def main(argv=None):
    run_edt_cli(argv, description=__doc__)


if __name__ == "__main__":
    main()
