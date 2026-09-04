from __future__ import annotations

from localscribe.models import Chunk, Segment, Transcript


def merge_chunks(
    chunk_results: list[tuple[Chunk, list[Segment]]],
    language: str,
    task: str,
) -> Transcript:
    """Stitch per-chunk segments into one continuous, correctly timestamped
    transcript. The offset applied to each chunk is the cumulative sum of
    *known chunk durations* (never a chunk's last segment end time) -- that
    stays correct even when VAD trims trailing silence a chunk's segments
    don't reach all the way to the chunk boundary."""

    merged: list[Segment] = []
    offset_seconds = 0.0

    for chunk, segments in chunk_results:
        for segment in segments:
            merged.append(
                Segment(
                    id=0,  # renumbered below
                    start=segment.start + offset_seconds,
                    end=segment.end + offset_seconds,
                    text=segment.text,
                )
            )
        offset_seconds += chunk.duration_ms / 1000

    for new_id, segment in enumerate(merged):
        segment.id = new_id

    return Transcript(language=language, task=task, segments=merged)
