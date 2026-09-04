from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from localscribe.constants import (
    DEFAULT_CHUNK_MINUTES,
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_DEVICE,
    DEFAULT_MODEL,
)


@dataclass
class Segment:
    """A transcribed span of speech, timestamped in seconds against the full
    (post-conversion) source file's timeline."""

    id: int
    start: float
    end: float
    text: str


@dataclass
class Transcript:
    """The final, merged result for one input file."""

    language: str
    task: str
    segments: list[Segment] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(segment.text for segment in self.segments)


@dataclass
class Chunk:
    """One piece of audio handed to Whisper. `duration_ms` is measured up
    front (via ffprobe or the source AudioSegment), never taken from
    Whisper's own output, so timestamp offsets stay correct regardless of
    trailing silence trimmed by VAD."""

    path: Path
    duration_ms: int
    index: int


@dataclass
class TranscribeOptions:
    task: str = "transcribe"
    model_size: str = DEFAULT_MODEL
    language: str | None = None
    initial_prompt: str | None = None
    condition_on_previous_text: bool = True
    vad_filter: bool = True
    device: str = DEFAULT_DEVICE
    compute_type: str = DEFAULT_COMPUTE_TYPE
    beam_size: int = 5
    chunk_minutes: float = DEFAULT_CHUNK_MINUTES
