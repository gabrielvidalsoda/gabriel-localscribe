from __future__ import annotations

from pathlib import Path

import pytest

from localscribe.cli import build_parser, main


def test_parser_defaults():
    args = build_parser().parse_args(["input.mp3"])

    assert args.files == ["input.mp3"]
    assert args.task == "transcribe"
    assert args.model == "small"
    assert args.language is None
    assert args.chunk_minutes == 10
    assert args.no_chunk is False
    assert args.vad_filter is True
    assert args.output_formats == "txt,srt,vtt,json"
    assert args.output_dir == "transcriptions"
    assert args.device == "cpu"


def test_parser_accepts_multiple_files_and_overrides():
    args = build_parser().parse_args(
        [
            "a.mp3", "b.mp4",
            "--task", "translate",
            "--model", "medium",
            "--language", "pt",
            "--no-chunk",
            "--no-vad",
            "--output-formats", "txt,srt",
            "--output-dir", "out",
        ]
    )

    assert args.files == ["a.mp3", "b.mp4"]
    assert args.task == "translate"
    assert args.model == "medium"
    assert args.language == "pt"
    assert args.no_chunk is True
    assert args.vad_filter is False
    assert args.output_formats == "txt,srt"
    assert args.output_dir == "out"


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(["--help"])

    assert exc_info.value.code == 0
    assert "localscribe" in capsys.readouterr().out


def test_main_errors_cleanly_on_missing_file(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["does-not-exist.mp3"])

    assert exc_info.value.code == 2
    assert "not found" in capsys.readouterr().err


def test_main_rejects_unknown_output_format(tmp_path: Path, capsys):
    real_file = tmp_path / "input.mp3"
    real_file.write_bytes(b"")

    with pytest.raises(SystemExit) as exc_info:
        main([str(real_file), "--output-formats", "txt,bogus"])

    assert exc_info.value.code == 2
    assert "bogus" in capsys.readouterr().err
