from __future__ import annotations

from pathlib import Path

import pytest

from localscribe.models import TranscribeOptions
from localscribe.pipeline import transcribe_file

# These load a real faster-whisper model (downloading it on first run) and
# run real inference -- slow, and skipped by default (pytest.ini: -m "not
# slow"). Run explicitly with: pytest -m slow

pytestmark = pytest.mark.slow


def test_transcribe_short_audio_file_end_to_end(synthetic_speech_like_audio: Path):
    options = TranscribeOptions(model_size="tiny", device="cpu", compute_type="int8")

    transcript = transcribe_file(synthetic_speech_like_audio, options)

    assert transcript.language
    # Synthetic tones aren't real speech, so we don't assert on the text --
    # this test exists to catch faster-whisper API drift and pipeline
    # wiring bugs, not transcription accuracy.
    assert isinstance(transcript.text, str)


def test_transcribe_video_file_forces_chunking_end_to_end(synthetic_mp4: Path):
    # chunk_minutes is tiny relative to the 2s fixture, forcing the chunking
    # path (and therefore extract_audio -> chunk_audio -> merge) to run.
    options = TranscribeOptions(model_size="tiny", device="cpu", compute_type="int8", chunk_minutes=0.01)

    transcript = transcribe_file(synthetic_mp4, options)

    assert transcript.language
    assert isinstance(transcript.text, str)
