from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from pydub import AudioSegment
from pydub.generators import Sine


@pytest.fixture
def synthetic_speech_like_audio(tmp_path: Path) -> Path:
    """A short audio file alternating tone/silence, standing in for speech
    with pauses -- enough to exercise chunking's silence-detection without
    depending on any real recording."""

    tone = Sine(440).to_audio_segment(duration=800)
    silence = AudioSegment.silent(duration=1500)
    audio = tone + silence + tone + silence + tone

    path = tmp_path / "synthetic.wav"
    audio.export(path, format="wav")
    return path


@pytest.fixture
def synthetic_mp4(tmp_path: Path) -> Path:
    """A tiny synthetic video (color bar + tone) generated with ffmpeg, used
    to exercise extract_audio without any real-media licensing question."""

    path = tmp_path / "synthetic.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=size=64x64:rate=1:duration=2",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
            "-shortest",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path
