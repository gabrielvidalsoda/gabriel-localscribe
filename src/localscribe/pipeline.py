from __future__ import annotations

import dataclasses
import tempfile
import time
from pathlib import Path
from typing import Protocol

from localscribe.chunking import chunk_audio
from localscribe.logging_utils import log, timed_step
from localscribe.media import classify_input, extract_audio, probe_duration_seconds
from localscribe.merge import merge_chunks
from localscribe.models import Chunk, Segment, TranscribeOptions, Transcript
from localscribe.transcription import load_model, transcribe_chunk
from localscribe.writers import write_outputs


class ProgressReporter(Protocol):
    def chunk_done(
        self,
        chunk_index: int,
        total_chunks: int,
        chunk: Chunk,
        transcript_so_far: Transcript,
        elapsed_seconds: float,
    ) -> None: ...


class PartialFileProgressReporter:
    """Writes an ASCII progress line (with per-chunk elapsed time) and
    overwrites a `.partial.txt` with the merge-so-far text after every
    chunk, so a killed long-running process doesn't lose everything it
    already transcribed."""

    def __init__(self, output_dir: Path, stem: str):
        self.partial_path = output_dir / f"{stem}.partial.txt"
        output_dir.mkdir(parents=True, exist_ok=True)

    def chunk_done(
        self,
        chunk_index: int,
        total_chunks: int,
        chunk: Chunk,
        transcript_so_far: Transcript,
        elapsed_seconds: float,
    ) -> None:
        log(
            f"chunk {chunk_index + 1}/{total_chunks} done in {elapsed_seconds:.1f}s "
            f"({len(transcript_so_far.segments)} segments so far)"
        )
        self.partial_path.write_text(transcript_so_far.text, encoding="utf-8")


def transcribe_file(
    input_path: Path,
    options: TranscribeOptions,
    progress: ProgressReporter | None = None,
) -> Transcript:
    file_start = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="localscribe_") as tmp:
        tmp_path = Path(tmp)

        if classify_input(input_path) == "video":
            with timed_step(f"extracting audio from {input_path.name}"):
                audio_path = extract_audio(input_path, tmp_path)
        else:
            audio_path = input_path

        duration_seconds = probe_duration_seconds(audio_path)

        if options.chunk_minutes > 0 and duration_seconds > options.chunk_minutes * 60:
            with timed_step(f"splitting {duration_seconds / 60:.1f} min of audio into chunks"):
                chunks = chunk_audio(audio_path, tmp_path, options.chunk_minutes)
        else:
            chunks = [Chunk(path=audio_path, duration_ms=int(duration_seconds * 1000), index=0)]

        with timed_step(f"loading model '{options.model_size}' on {options.device}"):
            model = load_model(options)

        chunk_results: list[tuple[Chunk, list[Segment]]] = []
        detected_language = options.language

        for chunk in chunks:
            chunk_start = time.perf_counter()
            chunk_options = dataclasses.replace(options, language=detected_language)
            segments, language = transcribe_chunk(model, chunk.path, chunk_options)
            chunk_elapsed = time.perf_counter() - chunk_start
            detected_language = detected_language or language
            chunk_results.append((chunk, segments))

            if progress:
                transcript_so_far = merge_chunks(chunk_results, detected_language, options.task)
                progress.chunk_done(chunk.index, len(chunks), chunk, transcript_so_far, chunk_elapsed)

        transcript = merge_chunks(chunk_results, detected_language or "unknown", options.task)

    log(f"{input_path.name} done in {time.perf_counter() - file_start:.1f}s total")
    return transcript


def transcribe_and_write(
    input_path: Path,
    options: TranscribeOptions,
    output_dir: Path,
    output_formats: list[str],
    progress: ProgressReporter | None = None,
) -> list[Path]:
    transcript = transcribe_file(input_path, options, progress=progress)
    return write_outputs(transcript, output_dir, input_path.stem, output_formats)
