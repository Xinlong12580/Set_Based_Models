"""The maintained runners are separate from the recoverable legacy archive."""

import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNERS = ROOT / "casual_tests"


class CasualTestsLayoutTests(unittest.TestCase):
    def test_only_current_entry_points_remain_at_top_level(self):
        self.assertEqual({path.name for path in RUNNERS.glob("run_*.py")}, {
            "run_EDT_event_only.py", "run_mixed_observables.py",
        })
        self.assertTrue((RUNNERS / "train_ERL.py").is_file())
        self.assertFalse((RUNNERS / "run_ERL_Gini_nested.py").exists())
        for name in ("run_ERL.py", "run_ERL_Gini.py", "run_ERL_Gini_event_only.py",
                     "run_ERL_Gini_nested_event_only.py", "run_ERL_Gini_penalized_event_only.py",
                     "run_EDT.py", "run_EDT_with_event_features.py", "_nested_rule_pruning.py",
                     "make_dataset.py"):
            with self.subTest(name=name):
                self.assertTrue((RUNNERS / "stale" / name).is_file())
                self.assertFalse((RUNNERS / name).exists())

    def test_current_modules_have_no_archive_imports(self):
        for path in RUNNERS.glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                    if node.module == "casual_tests":
                        names.extend("casual_tests." + alias.name for alias in node.names)
                else:
                    continue
                for name in names:
                    self.assertFalse(name == "casual_tests.stale" or name.startswith("casual_tests.stale."),
                                     f"{path.name} imports the legacy archive")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Study dependencies are optional")
    def test_current_and_relocated_cli_help_from_outside_repository(self):
        scripts = (
            "train_ERL.py", "run_EDT_event_only.py",
            "stale/run_ERL_Gini.py", "stale/run_ERL_Gini_nested_event_only.py",
            "stale/run_ERL_Gini_quantitative.py", "stale/run_EDT_with_event_features.py",
        )
        with tempfile.TemporaryDirectory() as temporary:
            env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
            env.pop("PYTHONPATH", None)
            for script in scripts:
                with self.subTest(script=script):
                    result = subprocess.run(
                        [sys.executable, "-B", str(RUNNERS / script), "--help"],
                        cwd=temporary, env=env, capture_output=True, text=True, timeout=30,
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn("--signal", result.stdout)
                    self.assertIn("--background", result.stdout)


if __name__ == "__main__":
    unittest.main()
