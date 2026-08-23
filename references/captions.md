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

The cached transcript is keyed by narration-audio SHA-256 plus each beat's final timeline start/duration, model size, requested language, and compute type. Changing any of those causes a new transcription. The execution device is not part of the cache identity because it selects where the same model computation runs rather than requesting different transcript semantics.

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

The initial accurate multilingual A-roll profile is
`large-v3-turbo` / `cpu` / `int8`. The configurable light fallback is
`small` / `cpu` / `int8`. English-only projects may explicitly select a
Distil-Whisper model supported by faster-whisper with `cpu` / `int8`; it is not
the multilingual default. `caption_whisper_model`,
`caption_whisper_device`, and `caption_whisper_compute_type` remain
configurable. These profiles do not depend on a 2 GB GPU.

This phase adds no original `openai-whisper`, Qwen3-ASR, `whisper.cpp`,
`llama.cpp`, vLLM, or bitsandbytes backend. It also adds no YouTube ingestion,
`pytube`, or `yt-dlp`; remote `source_audio_url` flow is removed and unneeded.
