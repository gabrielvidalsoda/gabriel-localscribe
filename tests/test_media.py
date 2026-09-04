from __future__ import annotations

from pathlib import Path

import pytest
from pydub import AudioSegment

from localscribe.media import UnsupportedMediaError, classify_input, extract_audio, probe_duration_seconds


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("a.mp3", "audio"),
        ("a.wav", "audio"),
        ("a.m4a", "audio"),
        ("a.flac", "audio"),
        ("a.mp4", "video"),
        ("a.mov", "video"),
        ("a.mkv", "video"),
        ("A.MP4", "video"),  # extension matching is case-insensitive
    ],
)
def test_classify_input(filename: str, expected: str):
    assert classify_input(Path(filename)) == expected


def test_classify_input_raises_for_unknown_extension():
    with pytest.raises(UnsupportedMediaError):
        classify_input(Path("a.xyz"))


def test_probe_duration_seconds(tmp_path: Path):
    audio = AudioSegment.silent(duration=2500)  # 2.5s
    path = tmp_path / "silence.wav"
    audio.export(path, format="wav")

    duration = probe_duration_seconds(path)

    assert abs(duration - 2.5) < 0.1


def test_extract_audio_produces_mono_16khz_mp3(tmp_path: Path, synthetic_mp4: Path):
    out_path = extract_audio(synthetic_mp4, tmp_path)

    assert out_path.exists()
    assert out_path.suffix == ".mp3"

    extracted = AudioSegment.from_file(out_path)
    assert extracted.frame_rate == 16000
    assert extracted.channels == 1
