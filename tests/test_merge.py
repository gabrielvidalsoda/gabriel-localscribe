from __future__ import annotations

from pathlib import Path

from localscribe.merge import merge_chunks
from localscribe.models import Chunk, Segment


def test_merge_applies_cumulative_duration_offset():
    chunk_a = Chunk(path=Path("a.mp3"), duration_ms=10_000, index=0)  # 10s
    chunk_b = Chunk(path=Path("b.mp3"), duration_ms=5_000, index=1)   # 5s

    segments_a = [Segment(id=0, start=0.0, end=2.0, text="hello")]
    # Chunk B's own timestamps are relative to its own start (0-based).
    segments_b = [Segment(id=0, start=0.5, end=1.5, text="world")]

    transcript = merge_chunks(
        [(chunk_a, segments_a), (chunk_b, segments_b)], language="en", task="transcribe"
    )

    assert [s.text for s in transcript.segments] == ["hello", "world"]
    assert transcript.segments[0].start == 0.0
    assert transcript.segments[0].end == 2.0
    # chunk_b's offset is chunk_a's duration (10s), not segments_a's last end (2s).
    assert transcript.segments[1].start == 10.5
    assert transcript.segments[1].end == 11.5


def test_merge_renumbers_segment_ids_sequentially():
    chunk_a = Chunk(path=Path("a.mp3"), duration_ms=1_000, index=0)
    chunk_b = Chunk(path=Path("b.mp3"), duration_ms=1_000, index=1)

    segments_a = [Segment(id=7, start=0.0, end=0.5, text="a1"), Segment(id=8, start=0.5, end=1.0, text="a2")]
    segments_b = [Segment(id=0, start=0.0, end=0.5, text="b1")]

    transcript = merge_chunks(
        [(chunk_a, segments_a), (chunk_b, segments_b)], language="en", task="transcribe"
    )

    assert [s.id for s in transcript.segments] == [0, 1, 2]


def test_merge_empty_chunk_list_produces_empty_transcript():
    transcript = merge_chunks([], language="en", task="transcribe")

    assert transcript.segments == []
    assert transcript.text == ""


def test_transcript_text_joins_segment_text_with_newlines():
    chunk = Chunk(path=Path("a.mp3"), duration_ms=1_000, index=0)
    segments = [Segment(id=0, start=0.0, end=0.5, text="one"), Segment(id=1, start=0.5, end=1.0, text="two")]

    transcript = merge_chunks([(chunk, segments)], language="en", task="transcribe")

    assert transcript.text == "one\ntwo"
