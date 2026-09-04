from __future__ import annotations

from pathlib import Path

from faster_whisper import WhisperModel

from localscribe.models import Segment, TranscribeOptions

TEMPERATURE_LADDER = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def load_model(options: TranscribeOptions) -> WhisperModel:
    return WhisperModel(
        options.model_size,
        device=options.device,
        compute_type=options.compute_type,
    )


def transcribe_chunk(
    model: WhisperModel,
    chunk_path: Path,
    options: TranscribeOptions,
) -> tuple[list[Segment], str]:
    """Transcribe one chunk. Returns (segments, detected_language) with
    segment timestamps relative to the START of this chunk -- the caller is
    responsible for applying the chunk's offset when merging."""

    segments_iter, info = model.transcribe(
        str(chunk_path),
        task=options.task,
        language=options.language,
        beam_size=options.beam_size,
        temperature=TEMPERATURE_LADDER,
        condition_on_previous_text=options.condition_on_previous_text,
        initial_prompt=options.initial_prompt,
        vad_filter=options.vad_filter,
    )

    segments = [
        Segment(id=i, start=s.start, end=s.end, text=s.text.strip())
        for i, s in enumerate(segments_iter)
    ]

    return segments, info.language
