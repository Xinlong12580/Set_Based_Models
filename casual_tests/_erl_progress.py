"""Dependency-free, throttled ERL progress on stderr, never in JSON/stdout."""

import sys
from time import perf_counter


class SearchProgress:
    """Render core search callbacks; one live TTY line or occasional log lines."""

    def __init__(self, stream=None, min_interval=None):
        self.stream = sys.stderr if stream is None else stream
        self.tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.min_interval = (0.5 if self.tty else 2.0) if min_interval is None else min_interval
        self.started = perf_counter()
        self.last_update = float("-inf")
        self.width = 0

    def __enter__(self):
        self._write("ERL preparing inputs for search...")
        return self

    def __exit__(self, *_):
        self.close()

    def __call__(self, status):
        now = perf_counter()
        force = status["stage"] in {"depth_started", "depth_finished", "finished"}
        if not force and now - self.last_update < self.min_interval:
            return
        self.last_update = now
        elapsed = int(now - self.started)
        done, total = status["parents_completed"], status["parents_total"]
        bars = int(20 * done / total) if total else 0
        line = (f"ERL depth {status['depth']}/{status['max_depth']} "
                f"[{'#' * bars}{'-' * (20 - bars)}] parents {done}/{total}")
        stage = status["stage"]
        if stage == "filtering_beam":
            detail = f"filtering beam {status['filter_completed']}/{status['filter_total']} (kept {status['retained']})"
        elif stage == "generating_cuts":
            detail = "generating thresholds"
        elif stage == "depth_finished":
            detail = f"depth complete (kept {status['retained']})"
        elif stage == "building_rules":
            detail = "building executable rules"
        elif stage == "finished":
            detail = f"search complete ({status['candidate_rules']} rules)"
        else:
            detail = f"cuts {status['cuts_completed']}/{status['cuts_total']}"
        self._write(f"{line} | {detail} | evaluated {status['evaluated_cuts']} "
                    f"| elapsed {elapsed // 60:02d}:{elapsed % 60:02d}")
        if stage == "finished":
            self.close()

    def _write(self, line):
        if self.tty:
            self.stream.write("\r" + line + " " * max(0, self.width - len(line)))
            self.width = len(line)
        else:
            self.stream.write(line + "\n")
        self.stream.flush()

    def close(self):
        if self.tty and self.width:
            self.stream.write("\n")
            self.stream.flush()
            self.width = 0
