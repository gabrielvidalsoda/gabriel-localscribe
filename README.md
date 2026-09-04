# gabriel-localscribe

Transcribe audio and video files to text and subtitles, fully locally, using [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

No cloud API, no account, no upload — everything runs on your own machine.

## How it works

```
input file
  |
  v
[video?] --yes--> extract audio track (ffmpeg: mono, 16kHz, small mp3)
  |no                        |
  v                          v
                    audio (mp3/wav/m4a/...)
                          |
                          v
              chunk on silence, if long
                          |
                          v
            faster-whisper, one chunk at a time
                          |
                          v
        merge segments, correct timestamps by
           cumulative chunk duration offset
                          |
                          v
              write txt / srt / vtt / json
```

Long files are split into chunks on natural pauses in the speech (not mid-
sentence) before transcription, so they fit comfortably in memory and a
killed/interrupted run only loses the chunk it was on — a `<file>.partial.txt`
is updated after every completed chunk while a run is in progress.

## Quickstart

1. Install [ffmpeg](https://ffmpeg.org/download.html) and make sure it's on your `PATH`:
   ```bash
   # Windows
   winget install Gyan.FFmpeg
   # macOS
   brew install ffmpeg
   # Debian/Ubuntu
   sudo apt install ffmpeg
   ```
2. Install `gabriel-localscribe`:
   ```bash
   pipx install git+https://github.com/gabrielvidalsoda/gabriel-localscribe
   ```
   (or clone the repo and run `poetry install`)
3. Run it on a file:
   ```bash
   localscribe interview.mp4
   ```
4. Find your transcript in `./transcriptions/interview.txt` (plus `.srt`, `.vtt`, `.json`).

## Usage

```
localscribe interview.mp4
localscribe podcast.mp3 --task translate --model medium
localscribe *.wav --output-formats txt,srt --output-dir out/
```

| Flag | Default | Description |
|---|---|---|
| `--task {transcribe,translate}` | `transcribe` | `translate` converts speech in any language to English text |
| `--model` | `small` | faster-whisper model size: `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`, `turbo` (`.en` variants also available) |
| `--language` | auto-detect | source language code, e.g. `en`, `pt` |
| `--prompt` | none | context/style hint passed to the model |
| `--no-condition-on-previous-text` | off | disable using prior chunk text as decoding context |
| `--chunk-minutes N` | `10` | split on silence into chunks of at most N minutes |
| `--no-chunk` | off | never split, transcribe the whole file in one pass |
| `--vad` / `--no-vad` | `--vad` | built-in voice-activity filtering (suppresses hallucinated text during silence) |
| `--output-formats` | `txt,srt,vtt,json` | comma-separated |
| `--output-dir` | `transcriptions` | where results are written |
| `--device {cpu,cuda}` | `cpu` | inference device; `cuda` is opt-in only, not auto-detected |
| `--compute-type` | `int8` | faster-whisper compute type, e.g. `int8`, `float16` |

## Known limitation

Temp chunk files are cleaned up on success, on any error, and on Ctrl+C — but
not if the process is force-killed (e.g. Task Manager, `SIGKILL`), since that
skips Python's cleanup entirely.

## Development

```bash
poetry install
poetry run pytest              # fast unit tests, no model download
poetry run pytest -m slow      # real end-to-end tests (downloads a Whisper model)
```

## Project layout

| Path | Purpose |
|---|---|
| `src/localscribe/media.py` | classify audio vs video, extract audio from video via ffmpeg |
| `src/localscribe/chunking.py` | silence-aware audio splitting for long files |
| `src/localscribe/transcription.py` | faster-whisper model loading and per-chunk transcription |
| `src/localscribe/merge.py` | stitches chunk results into one correctly timestamped transcript |
| `src/localscribe/writers.py` | txt / srt / vtt / json output |
| `src/localscribe/pipeline.py` | orchestrates the full flow, owns temp-file cleanup |
| `src/localscribe/cli.py` | command-line entry point |
| `tests/` | pytest suite (synthetic audio/video fixtures, no vendored media) |

## License

MIT — see [LICENSE](LICENSE).
