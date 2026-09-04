from __future__ import annotations

from pathlib import Path

from pydub import AudioSegment
from pydub.generators import Sine

from localscribe.chunking import chunk_audio


def test_single_chunk_when_max_is_large(tmp_path: Path, synthetic_speech_like_audio: Path):
    chunks = chunk_audio(synthetic_speech_like_audio, tmp_path, max_chunk_minutes=10)

    assert len(chunks) == 1
    assert chunks[0].path.exists()
    original_duration_ms = len(AudioSegment.from_file(synthetic_speech_like_audio))
    assert abs(chunks[0].duration_ms - original_duration_ms) < 50


def test_splits_on_silence_when_max_is_small(tmp_path: Path, synthetic_speech_like_audio: Path):
    # Each tone is 800ms; a ~0.9s cap forces a split between the two silence-
    # delimited, non-silent segments.
    chunks = chunk_audio(synthetic_speech_like_audio, tmp_path, max_chunk_minutes=0.9 / 60)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.path.exists()
        assert chunk.path.suffix == ".mp3"

    original_duration_ms = len(AudioSegment.from_file(synthetic_speech_like_audio))
    total_chunked_ms = sum(c.duration_ms for c in chunks)
    # mp3 round-tripping can shift boundaries slightly; stay within a loose tolerance.
    assert abs(total_chunked_ms - original_duration_ms) < 200


def test_raw_split_fallback_when_no_silence_found(tmp_path: Path):
    # One continuous 5s tone, no pause anywhere -- split_on_silence can't
    # find a cut point, so chunk_audio must fall back to hard equal splits.
    continuous_tone = Sine(440).to_audio_segment(duration=5000)
    audio_path = tmp_path / "continuous.wav"
    continuous_tone.export(audio_path, format="wav")

    max_chunk_ms = 2000
    chunks = chunk_audio(audio_path, tmp_path, max_chunk_minutes=max_chunk_ms / 60_000)

    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.duration_ms <= max_chunk_ms + 50

    total_chunked_ms = sum(c.duration_ms for c in chunks)
    assert abs(total_chunked_ms - 5000) < 200


def test_chunk_indices_are_sequential(tmp_path: Path, synthetic_speech_like_audio: Path):
    chunks = chunk_audio(synthetic_speech_like_audio, tmp_path, max_chunk_minutes=0.9 / 60)

    assert [c.index for c in chunks] == list(range(len(chunks)))
