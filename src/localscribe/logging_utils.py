from __future__ import annotations

import sys
import time


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
