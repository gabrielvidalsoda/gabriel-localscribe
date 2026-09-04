from __future__ import annotations

import json
from pathlib import Path

from localscribe.models import Segment, Transcript
from localscribe.writers import write_json, write_outputs, write_srt, write_txt, write_vtt


def _sample_transcript() -> Transcript:
    return Transcript(
        language="en",
        task="transcribe",
        segments=[
            Segment(id=0, start=0.0, end=1.5, text="Hello there."),
            Segment(id=1, start=1.5, end=3.25, text="General Kenobi."),
        ],
    )


def test_write_txt(tmp_path: Path):
    path = tmp_path / "out.txt"
    write_txt(_sample_transcript(), path)

    assert path.read_text(encoding="utf-8") == "Hello there.\nGeneral Kenobi."


def test_write_srt(tmp_path: Path):
    path = tmp_path / "out.srt"
    write_srt(_sample_transcript(), path)

    expected = (
        "1\n"
        "00:00:00,000 --> 00:00:01,500\n"
        "Hello there.\n"
        "\n"
        "2\n"
        "00:00:01,500 --> 00:00:03,250\n"
        "General Kenobi.\n"
    )
    assert path.read_text(encoding="utf-8") == expected


def test_write_vtt_nudges_zero_start(tmp_path: Path):
    path = tmp_path / "out.vtt"
    write_vtt(_sample_transcript(), path)

    content = path.read_text(encoding="utf-8")
    assert content.startswith("WEBVTT\n\n00:00:00.001 --> 00:00:01.500\nHello there.\n")
    assert "00:00:01.500 --> 00:00:03.250\nGeneral Kenobi." in content


def test_write_json(tmp_path: Path):
    path = tmp_path / "out.json"
    write_json(_sample_transcript(), path)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["language"] == "en"
    assert data["task"] == "transcribe"
    assert data["text"] == "Hello there.\nGeneral Kenobi."
    assert len(data["segments"]) == 2
    assert data["segments"][0] == {"id": 0, "start": 0.0, "end": 1.5, "text": "Hello there."}


def test_write_outputs_writes_requested_formats_only(tmp_path: Path):
    written = write_outputs(_sample_transcript(), tmp_path, "myfile", ["txt", "srt"])

    assert {p.name for p in written} == {"myfile.txt", "myfile.srt"}
    assert (tmp_path / "myfile.txt").exists()
    assert (tmp_path / "myfile.srt").exists()
    assert not (tmp_path / "myfile.vtt").exists()
