"""Exercise the Bash workflow with fake EOS metadata and mocked Python jobs.

No EOS access, conversion, tensor loading, or model fitting is performed.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNERS = ROOT / "quick_runners"


class SIGQuickRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="sig_quick_runners_")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.work = self.directory / "data with spaces"
        self.calls = self.directory / "calls.jsonl"
        self.metadata_reads = self.directory / "metadata_reads.txt"
        stubs = self.directory / "stubs"
        stubs.mkdir()
        # Only the source-listing subprocess imports these fake packages.
        (stubs / "sitecustomize.py").write_text(
            '''import io, os, sys, types
from pathlib import Path
from urllib.parse import urlsplit
class FileSystem:
    def ls(self, directory, detail=False):
        names = ["z.parquet", "a-old.parquet", ".sys.v#hidden.parquet",
                 "notes.txt", "c.parquet", "b.parquet"]
        return [directory + "/" + name for name in names] if "WJets" in directory else names
    def open(self, path, mode):
        with Path(os.environ["METADATA_READS"]).open("a") as output:
            output.write(path + "\\n")
        return io.BytesIO(path.encode())
def metadata(handle):
    path = handle.read().decode()
    version = b"wrong" if "a-old" in path or os.environ.get("NO_VERSION_MATCH") else b"collide2v_v1.0"
    return types.SimpleNamespace(metadata={b"dataset_version": version})
fsspec = types.ModuleType("fsspec")
fsspec.core = types.SimpleNamespace(url_to_fs=lambda url: (FileSystem(), urlsplit(url).path))
sys.modules["fsspec"] = fsspec
pyarrow = types.ModuleType("pyarrow")
pyarrow.__path__ = []
parquet = types.ModuleType("pyarrow.parquet")
parquet.read_metadata = metadata
pyarrow.parquet = parquet
sys.modules["pyarrow"] = pyarrow
sys.modules["pyarrow.parquet"] = parquet
''', encoding="utf-8")
        self.python = self.directory / "mock-python"
        self.python.write_text(
            f"#!{sys.executable}\n" + '''import json, os, sys
from pathlib import Path
args = sys.argv[1:]
if args[0] == "-":
    os.execv(sys.executable, [sys.executable, "-B", *args])
with Path(os.environ["CALL_LOG"]).open("a") as output:
    output.write(json.dumps({"argv": args, "cwd": os.getcwd()}) + "\\n")
print("Mock job: no conversion or fitting performed.")
if os.environ.get("FAIL_JOB") == args[0]:
    raise SystemExit(7)
''', encoding="utf-8")
        self.python.chmod(0o755)
        self.env = {
            **os.environ,
            "PYTHONPATH": str(stubs), "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHON_BIN": str(self.python), "CALL_LOG": str(self.calls),
            "METADATA_READS": str(self.metadata_reads),
            "SIG_WORK_DIR": str(self.work),
            "SIG_REPORT": str(self.directory / "reports" / "SIG.txt"),
            "SIG_MODEL": str(self.directory / "training_outputs" / "SIG.model.json"),
            "SIG_LOG": str(self.directory / "logs" / "SIG.log"),
            "EOS_REDIRECTOR": "root://fake.example",
            "EOS_SAMPLE_DIR": "/eos/fake/production_final",
            "SIG_PROCESS": "WJetsToLNu_13TeV-madgraphMLM-pythia8",
            "SIG_DATASET_VERSION": "collide2v_v1.0", "MAX_SOURCE_FILES": "2",
            "MAX_EVENTS": "10000", "REFERENCE_COUNT": "5000",
            "BACKGROUND_COUNT": "4500", "SIGNAL_COUNT": "500",
            "SAMPLING_SEED": "42", "REFERENCE_PENALTY": "20",
            "STRICTER_SCORE_IMPROVEMENT": "-0.01",
            "SIMILARITY_THRESHOLD": "0.9", "MAX_TRIGGER_RATE_KHZ": "40",
            "INPUT_RATE_MHZ": "40", "FAIL_JOB": "", "NO_VERSION_MATCH": "",
        }

    def run_script(self, name, *args, **overrides):
        return subprocess.run(
            [str(RUNNERS / name), *args], cwd=self.directory,
            env={**self.env, **overrides}, capture_output=True, text=True)

    def recorded_calls(self):
        return [json.loads(line) for line in self.calls.read_text().splitlines()]

    def test_scripts_are_executable_and_have_valid_bash_syntax(self):
        for script in RUNNERS.glob("*.sh"):
            with self.subTest(script=script.name):
                self.assertTrue(os.access(script, os.X_OK))
                result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                text = script.read_text()
                self.assertNotRegex(text, r"(?m)^\s*(?:source|\.)\s+")
                self.assertNotIn("DY", text)
        self.assertFalse((RUNNERS / "config.sh").exists())
        self.assertFalse(list(RUNNERS.glob("*_config.sh")))

    def test_full_workflow_wiring_and_source_order(self):
        result = self.run_script("pro_SIG.sh")
        self.assertEqual(result.returncode, 0, result.stderr)
        for sample in ("minbias", "SIG"):
            sources = (self.work / "sources" / f"{sample}.txt").read_text().splitlines()
            self.assertEqual([url.rsplit("/", 1)[1] for url in sources], ["b.parquet", "c.parquet"])
            self.assertTrue(all(url.startswith("root://fake.example//eos/fake/") for url in sources))
            expected_process = "minbias" if sample == "minbias" else self.env["SIG_PROCESS"]
            self.assertTrue(all(f"/{expected_process}/" in url for url in sources))
        reads = self.metadata_reads.read_text()
        self.assertNotIn("hidden", reads)
        self.assertNotIn("notes", reads)
        self.assertNotIn("z.parquet", reads)
        jobs = self.recorded_calls()
        self.assertEqual(len(jobs), 4)
        self.assertTrue(all(job["cwd"] == str(ROOT) for job in jobs))
        for job, label in zip(jobs[:2], ("1", "2")):
            args = job["argv"]
            self.assertEqual(args[:2], ["-m", "raw_event_conversion"])
            self.assertIn("--with-pair-tensor", args)
            self.assertEqual(args[args.index("--label") + 1], label)
            self.assertEqual(args[args.index("--max-events") + 1], "10000")
            sample = "minbias" if label == "1" else "SIG"
            self.assertEqual(args[args.index("--output") + 1], str(self.work / f"{sample}.pt"))
        mix = jobs[2]["argv"]
        self.assertEqual(mix[0], "scripts/make_reference_anomalous.py")
        self.assertEqual(mix[mix.index("--signal") + 1], str(self.work / "SIG.pt"))
        self.assertEqual(mix[mix.index("--background") + 1], str(self.work / "minbias.pt"))
        self.assertEqual(mix[mix.index("--output-dir") + 1], str(self.work / "mixture"))
        for name, value in (("reference-count", "5000"), ("background-count", "4500"),
                            ("signal-count", "500"), ("seed", "42")):
            self.assertEqual(mix[mix.index("--" + name) + 1], value)
        erl = jobs[3]["argv"]
        self.assertEqual(erl[0], "casual_tests/train_ERL.py")
        self.assertEqual(erl[erl.index("--model") + 1], self.env["SIG_MODEL"])
        self.assertIn("--overwrite-output", erl)
        for flag in ("--pair-tensor", "--object-only", "--require-complete-priority",
                     "--beam-similarity-filter", "--final-similarity-filter", "--progress"):
            self.assertIn(flag, erl)
        index = erl.index("--object-cut-priority")
        self.assertEqual(erl[index + 1:index + 5],
                         ["type_id_1", "type_id_2", "pair_mass:>", "pair_mass:<="])
        self.assertEqual(erl[erl.index("--signal") + 1], str(self.work / "mixture" / "anomalous.pt"))
        self.assertEqual(erl[erl.index("--background") + 1], str(self.work / "mixture" / "reference.pt"))
        self.assertIn("Mock job", Path(self.env["SIG_LOG"]).read_text())

    def test_source_limit_zero_selects_all_matching_files(self):
        result = self.run_script("01_select_SIG_sources.sh", MAX_SOURCE_FILES="0")
        self.assertEqual(result.returncode, 0, result.stderr)
        sources = (self.work / "sources" / "minbias.txt").read_text().splitlines()
        self.assertEqual([url.rsplit("/", 1)[1] for url in sources],
                         ["b.parquet", "c.parquet", "z.parquet"])

    def test_default_signal_and_event_count(self):
        result = self.run_script("pro_SIG.sh", SIG_PROCESS="", MAX_EVENTS="", MAX_SOURCE_FILES="")
        self.assertEqual(result.returncode, 0, result.stderr)
        sources = (self.work / "sources" / "SIG.txt").read_text().splitlines()
        self.assertEqual(len(sources), 3)
        self.assertTrue(all("/WJetsToLNu_13TeV-madgraphMLM-pythia8/" in url for url in sources))
        for job in self.recorded_calls()[:2]:
            args = job["argv"]
            self.assertEqual(args[args.index("--max-events") + 1], "100000")

    def test_process_specific_default_paths_and_standalone_steps(self):
        # Copy only the executables: no central config or existing dataset is available.
        isolated_root = self.directory / "standalone repo"
        isolated_runners = isolated_root / "quick_runners"
        isolated_runners.mkdir(parents=True)
        for script in RUNNERS.glob("*.sh"):
            shutil.copy2(script, isolated_runners / script.name)
        process = "ggHgammagamma"
        result = subprocess.run(
            [str(isolated_runners / "pro_SIG.sh")], cwd=self.directory,
            env={**self.env, "SIG_PROCESS": process, "SIG_WORK_DIR": "",
                 "SIG_MODEL": "", "SIG_TRAINING_PREFIX": "", "SIG_REPORT": "", "SIG_LOG": ""}, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        work = isolated_root / "data" / process
        self.assertTrue((work / "sources" / "SIG.txt").is_file())
        jobs = self.recorded_calls()
        self.assertTrue(all(job["cwd"] == str(isolated_root) for job in jobs))
        args = jobs[1]["argv"]
        self.assertEqual(args[args.index("--output") + 1], str(work / "SIG.pt"))
        args = jobs[3]["argv"]
        prefix = isolated_root / "training_outputs" / f"ERL_{process}_pairs"
        self.assertEqual(args[args.index("--training-report") + 1], str(prefix) + ".report.json")
        self.assertEqual(args[args.index("--model") + 1], str(prefix) + ".model.json")
        self.assertIn("Mock job", Path(str(prefix) + ".training.log").read_text())

    def test_missing_version_match_stops_before_conversion(self):
        result = self.run_script("pro_SIG.sh", NO_VERSION_MATCH="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No collide2v_v1.0 Parquet files found", result.stderr)
        self.assertFalse(self.calls.exists())

    def test_standalone_training_names_all_outputs_without_source_selection(self):
        prefix = self.directory / "custom training" / "ERL_signal"
        result = self.run_script("05_train_SIG_ERL.sh", SIG_TRAINING_PREFIX=str(prefix), SIG_MODEL="", SIG_REPORT="", SIG_LOG="")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.recorded_calls()[0]["argv"]
        self.assertEqual(args[0], "casual_tests/train_ERL.py")
        self.assertEqual(args[args.index("--model") + 1], str(prefix) + ".model.json")
        self.assertEqual(args[args.index("--training-report") + 1], str(prefix) + ".report.json")
        self.assertIn("--overwrite-output", args)
        self.assertIn("Mock job", Path(str(prefix) + ".training.log").read_text())

    def test_conversion_failure_stops_later_steps(self):
        result = self.run_script("pro_SIG.sh", FAIL_JOB="-m")
        self.assertEqual(result.returncode, 7)
        self.assertEqual(len(self.recorded_calls()), 1)
        self.assertFalse(Path(self.env["SIG_LOG"]).exists())

    def test_erl_failure_survives_tee_and_extra_arguments_are_forwarded(self):
        result = self.run_script(
            "05_train_SIG_ERL.sh", "--print-top", "2",
            FAIL_JOB="casual_tests/train_ERL.py", REFERENCE_PENALTY="10")
        self.assertEqual(result.returncode, 7)
        args = self.recorded_calls()[0]["argv"]
        self.assertEqual(args[-2:], ["--print-top", "2"])
        self.assertEqual(args[args.index("--reference-penalty") + 1], "10")
        self.assertIn("Mock job", Path(self.env["SIG_LOG"]).read_text())


if __name__ == "__main__":
    unittest.main()
