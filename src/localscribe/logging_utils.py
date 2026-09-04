from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from typing import Iterator


def configure_stdout_utf8() -> None:
    """Force UTF-8 on stdout/stderr so any non-ASCII text (e.g. a foreign
    language sample in --prompt, or a filename) can't crash the process on
    a Windows console still using a legacy codepage (cp1252). Every message
    this package prints is ASCII-only by construction anyway -- this is
    belt-and-suspenders, not a license to use non-ASCII in log lines."""

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def log(message: str) -> None:
    """ASCII-safe, timestamped progress line."""

    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


@contextmanager
def timed_step(description: str) -> Iterator[None]:
    """Logs `<description>...` when a step starts and `<description> done
    (X.Xs)` when it ends (even if it raises), so every step in the pipeline
    is visible with its own duration in seconds."""

    log(f"{description}...")
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        log(f"{description} done ({elapsed:.1f}s)")
