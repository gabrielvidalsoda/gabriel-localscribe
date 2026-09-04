from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from localscribe.models import Transcript


def _format_srt_timestamp(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def _format_vtt_timestamp(seconds: float) -> str:
    return _format_srt_timestamp(seconds).replace(",", ".")


def write_txt(transcript: Transcript, path: Path) -> None:
    path.write_text(transcript.text, encoding="utf-8")


def write_srt(transcript: Transcript, path: Path) -> None:
    lines = []

    for segment in transcript.segments:
        lines.append(str(segment.id + 1))
        lines.append(
            f"{_format_srt_timestamp(segment.start)} --> {_format_srt_timestamp(segment.end)}"
        )
        lines.append(segment.text)
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def write_vtt(transcript: Transcript, path: Path) -> None:
    lines = ["WEBVTT", ""]

    for i, segment in enumerate(transcript.segments):
        # Some players fail to render a caption starting at exactly 0s;
        # nudge only the very first cue, only in the output text.
        start = 0.001 if i == 0 and segment.start == 0 else segment.start
        lines.append(f"{_format_vtt_timestamp(start)} --> {_format_vtt_timestamp(segment.end)}")
        lines.append(segment.text)
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def write_json(transcript: Transcript, path: Path) -> None:
    data = {
        "language": transcript.language,
        "task": transcript.task,
        "text": transcript.text,
        "segments": [asdict(segment) for segment in transcript.segments],
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


WRITERS: dict[str, Callable[[Transcript, Path], None]] = {
    "txt": write_txt,
    "srt": write_srt,
    "vtt": write_vtt,
    "json": write_json,
}


def write_outputs(
    transcript: Transcript,
    output_dir: Path,
    stem: str,
    formats: list[str],
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for fmt in formats:
        writer = WRITERS[fmt]
        path = output_dir / f"{stem}.{fmt}"
        writer(transcript, path)
        written.append(path)

    return written
