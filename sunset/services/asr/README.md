# ASR Service

Automatic Speech Recognition powered by [Deepgram](https://deepgram.com). Transcribes audio from binary data, local files, or public URLs. Supports pre-recorded and streaming modes with automatic language detection.

## Setup

**Install the optional dependency:**

```bash
pip install "sunset-core[asr]"
```

**Environment variables:**

| Variable | Required | Description |
|---|---|---|
| `DEEPGRAM_API_KEY` | Yes | Deepgram API key |

## Usage

### Pre-recorded transcription

```python
from sunset.services import ASRService

asr = ASRService()  # reads DEEPGRAM_API_KEY from env

# From binary data
transcript = await asr.transcribe(audio_bytes)

# From local file path
transcript = await asr.transcribe("/path/to/audio.wav")

# From public URL
transcript = await asr.transcribe("https://example.com/audio.mp3")

print(transcript.text)
print(transcript.language)  # auto-detected language code
print(transcript.words)     # list of Word(word, start, end, confidence, speaker)
```

### Streaming transcription

Returns an async iterator yielding partial transcript strings as they arrive:

```python
stream = await asr.transcribe(audio_bytes, streaming=True)
async for partial in stream:
    print(partial, end=" ", flush=True)
```

### Options

```python
transcript = await asr.transcribe(
    source,
    language="fr",         # explicit language (skips auto-detection)
    diarize=True,          # speaker identification
    smart_format=True,     # format dates, numbers, etc. (default: True)
    punctuate=True,        # add punctuation (default: True)
)
```

## API Reference

### `ASRService(api_key=None, model="nova-3")`

- `api_key` — Deepgram API key. Falls back to `DEEPGRAM_API_KEY` env var.
- `model` — Deepgram model to use. Default `nova-3`.

### `await transcribe(source, *, streaming=False, language=None, punctuate=True, smart_format=True, diarize=False, **kwargs) -> Transcript | AsyncIterator[str]`

- `source` — `bytes`, file path (`str`/`Path`), or URL (`str` starting with `http`).
- `streaming` — If `True`, returns an async iterator of partial transcripts.
- `language` — Explicit language code. If `None`, auto-detection is enabled.
- `**kwargs` — Passed through to Deepgram API options.

### `Transcript`

| Field | Type | Description |
|---|---|---|
| `text` | `str` | Full transcript text |
| `words` | `list[Word]` | Word-level timestamps and confidence |
| `language` | `str \| None` | Detected language code |
| `duration` | `float \| None` | Audio duration in seconds |

### `Word`

| Field | Type | Description |
|---|---|---|
| `word` | `str` | The word |
| `start` | `float` | Start time (seconds) |
| `end` | `float` | End time (seconds) |
| `confidence` | `float` | Confidence score (0-1) |
| `speaker` | `int \| None` | Speaker index (when `diarize=True`) |
