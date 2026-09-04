AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".opus"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".flv", ".wmv", ".3gp", ".m4v"}

MODEL_CHOICES = [
    "tiny", "tiny.en",
    "base", "base.en",
    "small", "small.en",
    "medium", "medium.en",
    "large-v2", "large-v3",
    "turbo",
]

OUTPUT_FORMAT_CHOICES = ["txt", "srt", "vtt", "json"]

DEFAULT_MODEL = "small"
DEFAULT_CHUNK_MINUTES = 10
DEFAULT_OUTPUT_FORMATS = ["txt", "srt", "vtt", "json"]
DEFAULT_OUTPUT_DIR = "transcriptions"
DEFAULT_DEVICE = "cpu"
DEFAULT_COMPUTE_TYPE = "int8"

# Extraction target for video -> audio conversion (Whisper's native input rate)
EXTRACT_SAMPLE_RATE = 16000
EXTRACT_CHANNELS = 1
