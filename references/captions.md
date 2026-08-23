# Word-timed captions

Vox supports local word-timed ASS captions. The caption subsystem uses
faster-whisper and ffmpeg/libass; it is separate from MuAPI generation.

## Install

```bash
python -m pip install -r requirements-captions.txt
```

`ffmpeg` must include the `subtitles` / libass filter.

## B-roll and C-roll data flow

```text
per-beat narration audio
→ faster-whisper word timestamps
→ project timeline offsets
→ out/<project>/captions/transcript.json
→ ASS caption renderer
→ out/<project>/captions/captions.ass
→ ffmpeg subtitles filter
→ final.mp4
```

### B-roll/C-roll cache identity

The B-roll/C-roll cache is a timeline fingerprint: each beat's ID,
narration-audio SHA-256, final timeline start/duration, model size, requested
language, and compute type. Changing any of those causes a new transcription.
The execution device is not part of the cache identity because it selects where
the same model computation runs rather than requesting different transcript
semantics.

## Shared transcript semantic core

B-roll, C-roll, and A-roll use this shared semantic core for the transcript
content and word timestamps:

```json
{
  "language": "en",
  "segments": [
    {
      "start": 0.0,
      "end": 1.2,
      "text": "Example words",
      "words": [
        {"word": "Example", "start": 0.0, "end": 0.6},
        {"word": "words", "start": 0.6, "end": 1.2}
      ]
    }
  ]
}
```

This is not the complete persisted `captions/transcript.json` shape. That file
surrounds the shared core with `schema_version`, `source_fingerprint`, `model`,
`requested_language`, and `compute_type` metadata. B-roll/C-roll segments may
also include `beat_id` to identify their timeline beat. A-roll's semantic
contract for `language`, segments, and word timestamps remains the same; its
source transcript is remapped to edit-time segments during assembly.

## beats.json controls

```json
{
  "caption_mode": "word",
  "caption_style": "editorial",
  "caption_position": 10,
  "caption_whisper_model": "base",
  "caption_whisper_device": "auto",
  "caption_whisper_compute_type": "default",
  "caption_required": false
}
```

- `caption_mode`: `word` (default), `static`, or `off` for B-roll/C-roll.
- `caption_style`: `editorial`, `classic`, `hormozi`, `mrbeast`, `karaoke`, `minimal`, or `bounce`. Legacy `white` and `paper` remain accepted aliases.
- `caption_position`: percentage from the bottom, `0`–`50`.
- `caption_whisper_model`: faster-whisper model size. Environment fallback: `VOX_CAPTION_WHISPER_MODEL`; default `base`.
- `caption_required`: when `true`, missing faster-whisper is a hard failure. When `false`, Vox reports the missing dependency and falls back to the previous static beat captions.

The caption renderer writes word timing from the transcription result; it does not estimate timestamps from narration text.

## A-roll captions

A-roll uses one canonical local transcript for both beat construction and captions:

```text
original source video/audio
→ SHA-256-keyed faster-whisper transcript
→ out/<project>/captions/transcript.json
├→ timed A-roll beats
└→ successful edit-span remapping → shared ASS renderer → captions/captions.ass
```

`asr_beats.py` directly hashes and transcribes the original source media once,
then writes a new A-roll `beats.json` with `"caption_mode": "word"`. During
assembly, A-roll reads that same cached transcript and remaps its source word
timestamps onto the final edit timeline. It does not transcribe again or invent
word timings. The final A-roll retains the original source-audio segments.

### A-roll edit timing

Assembly normalizes each A-roll beat's source `start`, `end`, and requested
duration independently to centiseconds (round-half-up), then safely floors the
encoded cut so it never reaches past that source interval. To make short
24-fps/AAC outputs reliable, assembly rejects an effective cut below one second;
this is only an A-roll rendering floor (normal ASR beats are at least two
seconds), not a transcription rule.

Each accepted source cut is extracted as lossless PCM, paired with its reset-PTS
video in a PCM MOV intermediate, and those intermediates are concatenated. The
final has one original-speech AAC encode: `off` stream-copies the concatenated
video, while `word` encodes only the video needed to burn subtitles. Neither
mode mixes, pads, or duplicates speech. Container, audio, and video timestamps
can differ from the requested edit timeline by at most one 24-fps frame plus one
AAC packet (1024 samples, about 62.5 ms at 48 kHz); stream starts are reset near
zero and that tolerance does not accumulate per beat.

### A-roll cache identity

Unlike the B-roll/C-roll timeline fingerprint, the A-roll cache is keyed by the
original source-media SHA-256, model, requested language, and compute type.
Device is deliberately excluded: it selects where the same model computation
runs rather than changing the requested transcript semantics.

A-roll accepts only `"caption_mode": "word"` and `"caption_mode": "off"`.
`word` burns ASS captions through the shared renderer; `off` skips transcript
loading and caption rendering. A-roll projects created before this support that
omit `caption_mode` remain `off` for compatibility. A-roll does not add a
`static` mode; B-roll/C-roll `word` (default), `static`, and `off` behavior is
unchanged.

All modes share these caption styles: `editorial`, `classic`, `hormozi`,
`mrbeast`, `karaoke`, `minimal`, and `bounce`. The legacy `white` and `paper`
names remain aliases.

### A-roll transcription profiles

The operational multilingual A-roll profile is `small` / `cpu` / `int8`.
`large-v3-turbo` / `cpu` / `int8` remains a configurable accurate profile.
This operational choice is measured, rather than an accuracy ranking: on a
15.583333 s fixture, warm uncached runs took 6.198 s (RTF 0.398) for small and
17.496 s (RTF 1.123) for large (2.82x slower); cold-run observed peaks were at
least 519 MiB and 1.64 GiB, respectively. Both runs yielded the expected 20
monotonic words in 3 segments. English-only projects may explicitly select a
Distil-Whisper model supported by faster-whisper with `cpu` / `int8`; it is not
the multilingual default. `caption_whisper_model`,
`caption_whisper_device`, and `caption_whisper_compute_type` remain
configurable. These profiles do not depend on a 2 GB GPU.

This phase adds no original `openai-whisper`, Qwen3-ASR, `whisper.cpp`,
`llama.cpp`, vLLM, or bitsandbytes backend. It also adds no YouTube ingestion,
`pytube`, or `yt-dlp`; remote `source_audio_url` flow is removed and unneeded.
