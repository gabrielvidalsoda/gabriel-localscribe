from __future__ import annotations

import math
from pathlib import Path

from pydub import AudioSegment
from pydub.silence import split_on_silence

from localscribe.models import Chunk

# Tuned silence-detection parameters (ported from prior art: seek in small
# steps, require a real pause before splitting, keep_silence so segments
# aren't clipped mid-breath).
SEEK_STEP_MS = 5
MIN_SILENCE_LEN_MS = 1250
SILENCE_THRESH_DB = -25


def chunk_audio(audio_path: Path, work_dir: Path, max_chunk_minutes: float) -> list[Chunk]:
    """Split `audio_path` into chunks of at most `max_chunk_minutes`, cutting
    on silence so speech is never split mid-sentence where avoidable. Chunks
    are written into `work_dir` (caller owns cleanup, e.g. via
    tempfile.TemporaryDirectory) -- never as siblings of the source file."""

    max_chunk_ms = int(max_chunk_minutes * 60 * 1000)
    audio = AudioSegment.from_file(audio_path)
    chunks: list[Chunk] = []

    def add_chunk(segment: AudioSegment) -> None:
        index = len(chunks)
        chunk_path = work_dir / f"chunk_{index:04d}.mp3"
        segment.export(chunk_path, format="mp3")
        chunks.append(Chunk(path=chunk_path, duration_ms=len(segment), index=index))

    def raw_split(big_segment: AudioSegment) -> None:
        # No silence found to cut on and it's still too long: fall back to
        # equal-sized hard cuts so we never hand Whisper an oversized chunk.
        subchunks = math.ceil(len(big_segment) / max_chunk_ms)

        for i in range(subchunks):
            start = max_chunk_ms * i
            end = min(max_chunk_ms * (i + 1), len(big_segment))
            add_chunk(big_segment[start:end])

    non_silent_segments = split_on_silence(
        audio,
        seek_step=SEEK_STEP_MS,
        min_silence_len=MIN_SILENCE_LEN_MS,
        silence_thresh=SILENCE_THRESH_DB,
        keep_silence=True,
    )

    current = non_silent_segments[0] if non_silent_segments else audio

    for next_segment in non_silent_segments[1:]:
        if len(current) > max_chunk_ms:
            raw_split(current)
            current = next_segment
        elif len(current) + len(next_segment) <= max_chunk_ms:
            current += next_segment
        else:
            add_chunk(current)
            current = next_segment

    if len(current) > max_chunk_ms:
        raw_split(current)
    else:
        add_chunk(current)

    return chunks
