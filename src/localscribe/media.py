from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Literal

from localscribe.constants import AUDIO_EXTS, EXTRACT_CHANNELS, EXTRACT_SAMPLE_RATE, VIDEO_EXTS


class UnsupportedMediaError(ValueError):
    pass


def classify_input(path: Path) -> Literal["audio", "video"]:
    ext = path.suffix.lower()

    if ext in AUDIO_EXTS:
        return "audio"
    if ext in VIDEO_EXTS:
        return "video"

    raise UnsupportedMediaError(
        f"Unrecognized extension '{ext}' for {path.name}. "
        f"Supported audio: {sorted(AUDIO_EXTS)}. Supported video: {sorted(VIDEO_EXTS)}."
    )


def probe_duration_seconds(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def extract_audio(
    video_path: Path,
    out_dir: Path,
    sample_rate: int = EXTRACT_SAMPLE_RATE,
    channels: int = EXTRACT_CHANNELS,
) -> Path:
    """Extract just the audio track from a video file, downmixed and
    downsampled to Whisper's native input format, dropped to a small VBR
    mp3. No video-quality audio is needed for speech recognition."""

    out_path = out_dir / f"{video_path.stem}.mp3"

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn",
            "-ac", str(channels),
            "-ar", str(sample_rate),
            "-c:a", "libmp3lame", "-q:a", "4",
            str(out_path),
        ],
        check=True,
        capture_output=True,
    )

    return out_path
