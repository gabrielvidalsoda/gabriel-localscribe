from __future__ import annotations

import argparse
import sys
from pathlib import Path

from localscribe.constants import (
    DEFAULT_CHUNK_MINUTES,
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_DEVICE,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_FORMATS,
    MODEL_CHOICES,
    OUTPUT_FORMAT_CHOICES,
)
from localscribe.ffmpeg_utils import FFmpegNotFoundError, check_ffmpeg_available
from localscribe.logging_utils import configure_stdout_utf8, log
from localscribe.models import TranscribeOptions
from localscribe.pipeline import PartialFileProgressReporter, transcribe_and_write


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="localscribe",
        description="Transcribe audio and video files to text and subtitles, locally, using faster-whisper.",
        epilog=(
            "Examples:\n"
            "  localscribe interview.mp4\n"
            "  localscribe podcast.mp3 --task translate --model medium\n"
            "  localscribe *.wav --output-formats txt,srt --output-dir out/\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("files", metavar="FILE", nargs="+", help="one or more audio or video files to transcribe")

    parser.add_argument(
        "--task", choices=["transcribe", "translate"], default="transcribe",
        help="transcribe in the source language (default), or translate to English",
    )
    parser.add_argument(
        "--model", choices=MODEL_CHOICES, default=DEFAULT_MODEL,
        help=f"faster-whisper model size (default: {DEFAULT_MODEL}; CPU-friendly)",
    )
    parser.add_argument(
        "--language", default=None,
        help="source language, e.g. 'en', 'pt' (default: auto-detect on the first chunk)",
    )
    parser.add_argument("--prompt", default=None, help="optional context/style hint for the model")
    parser.add_argument(
        "--no-condition-on-previous-text", action="store_false", dest="condition_on_previous_text",
        default=True, help="disable using prior chunk text as context (default: enabled)",
    )
    parser.add_argument(
        "--chunk-minutes", type=float, default=DEFAULT_CHUNK_MINUTES,
        help=f"split audio into chunks of at most N minutes on silence boundaries (default: {DEFAULT_CHUNK_MINUTES})",
    )
    parser.add_argument(
        "--no-chunk", action="store_true",
        help="never split, transcribe the whole file in one pass",
    )
    parser.add_argument(
        "--vad", action="store_true", dest="vad_filter", default=True,
        help="enable built-in voice-activity filtering (default: enabled)",
    )
    parser.add_argument(
        "--no-vad", action="store_false", dest="vad_filter",
        help="disable built-in voice-activity filtering",
    )
    parser.add_argument(
        "--output-formats", default=",".join(DEFAULT_OUTPUT_FORMATS),
        help=f"comma-separated: {','.join(OUTPUT_FORMAT_CHOICES)} (default: all four)",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help=f"where to write results (default: ./{DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default=DEFAULT_DEVICE,
        help="inference device (default: cpu; pass --device cuda only if you have a working CUDA setup)",
    )
    parser.add_argument(
        "--compute-type", default=DEFAULT_COMPUTE_TYPE,
        help=f"faster-whisper compute type, e.g. int8, float16 (default: {DEFAULT_COMPUTE_TYPE})",
    )

    return parser


def _parse_output_formats(raw: str) -> list[str]:
    formats = [fmt.strip() for fmt in raw.split(",") if fmt.strip()]

    for fmt in formats:
        if fmt not in OUTPUT_FORMAT_CHOICES:
            raise ValueError(
                f"Unknown output format '{fmt}'. Choose from: {', '.join(OUTPUT_FORMAT_CHOICES)}"
            )

    return formats


def main(argv: list[str] | None = None) -> int:
    configure_stdout_utf8()

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        output_formats = _parse_output_formats(args.output_formats)
    except ValueError as e:
        parser.error(str(e))
        return 2  # unreachable, parser.error exits; keeps type-checkers happy

    input_paths = [Path(f) for f in args.files]
    missing = [str(p) for p in input_paths if not p.is_file()]

    if missing:
        parser.error(f"file(s) not found: {', '.join(missing)}")

    try:
        check_ffmpeg_available()
    except FFmpegNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    options = TranscribeOptions(
        task=args.task,
        model_size=args.model,
        language=args.language,
        initial_prompt=args.prompt,
        condition_on_previous_text=args.condition_on_previous_text,
        vad_filter=args.vad_filter,
        device=args.device,
        compute_type=args.compute_type,
        chunk_minutes=0 if args.no_chunk else args.chunk_minutes,
    )

    output_dir = Path(args.output_dir)

    for input_path in input_paths:
        log(f"processing {input_path.name}")
        progress = PartialFileProgressReporter(output_dir, input_path.stem)
        written = transcribe_and_write(input_path, options, output_dir, output_formats, progress=progress)

        for path in written:
            print(path)

        if progress.partial_path.exists():
            progress.partial_path.unlink()

    return 0


if __name__ == "__main__":
    sys.exit(main())
