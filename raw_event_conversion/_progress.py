"""Dependency-free conversion progress on stderr, never in saved outputs."""

import sys
from time import perf_counter


class ConversionProgress:
    """One live terminal line, or throttled lines when stderr is redirected."""

    def __init__(self, total_files, total_events=None, *, enabled=True, stream=None):
        self.enabled = enabled
        self.stream = sys.stderr if stream is None else stream
        self.tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.interval = 0.5 if self.tty else 2.0
        self.total_files, self.total_events = total_files, total_events
        self.files = self.events = self.chunks = 0
        self.started = perf_counter()
        self.last_update = float("-inf")
        self.width = 0

    def __enter__(self):
        self.update("preparing", force=True)
        return self

    def __exit__(self, exc_type, *_):
        self.update("done" if exc_type is None else "failed/interrupted", force=True)
        if self.enabled and self.tty:
            self.stream.write("\n")
            self.stream.flush()

    def update(self, stage, *, files=None, events=None, chunks=None, force=False):
        if not self.enabled:
            return
        if files is not None:
            self.files = files
        if events is not None:
            self.events = events
        if chunks is not None:
            self.chunks = chunks
        now = perf_counter()
        if not force and now - self.last_update < self.interval:
            return
        self.last_update = now
        done, total = ((self.events, self.total_events) if self.total_events is not None
                       else (self.files, self.total_files))
        filled = min(20, int(20 * done / total)) if total else 0
        events = f"{self.events:,}"
        if self.total_events is not None:
            events += f"/{self.total_events:,}"
        elapsed = int(now - self.started)
        line = (f"Convert [{'#' * filled}{'-' * (20 - filled)}] "
                f"events {events} | files {self.files}/{self.total_files} "
                f"| chunks staged {self.chunks} | {stage} "
                f"| elapsed {elapsed // 60:02d}:{elapsed % 60:02d}")
        if self.tty:
            self.stream.write("\r" + line + " " * max(0, self.width - len(line)))
            self.width = len(line)
        else:
            self.stream.write(line + "\n")
        self.stream.flush()
