"""Run the object-only Gini ERL and write a human-readable JSON text report."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from casual_tests._quantitative_reporting import run_erl_cli  # noqa: E402


def main(argv=None) -> None:
    run_erl_cli(argv, require_event_features=False, description=__doc__)


if __name__ == "__main__":
    main()
