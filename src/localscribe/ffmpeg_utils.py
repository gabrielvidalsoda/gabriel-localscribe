from __future__ import annotations

import shutil
import sys

_INSTALL_HINTS = {
    "win32": "winget install Gyan.FFmpeg   (or: choco install ffmpeg)",
    "darwin": "brew install ffmpeg",
    "linux": "sudo apt install ffmpeg   (Debian/Ubuntu; use your distro's package manager otherwise)",
}


class FFmpegNotFoundError(RuntimeError):
    pass


def _install_hint() -> str:
    return _INSTALL_HINTS.get(sys.platform, "https://ffmpeg.org/download.html")


def check_ffmpeg_available() -> None:
    """Raise FFmpegNotFoundError with an actionable message if ffmpeg or
    ffprobe is not on PATH. Call this once, up front, before any file is
    touched or any model is loaded."""

    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]

    if missing:
        raise FFmpegNotFoundError(
            f"{' and '.join(missing)} not found on PATH.\n"
            f"Install ffmpeg, then re-run:\n  {_install_hint()}"
        )
