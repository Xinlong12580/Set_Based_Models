"""Run Gini ERL on event_features only, using fully masked dummy object slots.

Example (from the repository root)::

    python casual_tests/run_ERL_Gini_event_only.py \
        --signal data/ZZ_event_features/anomalous.pt \
        --background data/ZZ_event_features/reference.pt \
        --report reports/ERL_ZZ_event_only.txt

The signal input may be a background-contaminated anomaly region. Stored
process labels are used only for reporting, never to choose the training class.
The separate native_event_info block is not used. Scoring, beam settings, and
pruning match the existing quantitative Gini ERL runner. Reports remain
compatible with scripts/plot_acceptance_report.py.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from casual_tests._event_only_reporting import run_erl_cli  # noqa: E402


def main(argv=None):
    run_erl_cli(argv, description=__doc__)


if __name__ == "__main__":
    main()
