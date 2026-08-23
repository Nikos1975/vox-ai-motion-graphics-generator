# Word-timed captions

Vox supports word-timed ASS captions for B-roll and C-roll assembly. The caption subsystem is local and separate from MuAPI generation.

## Install

```bash
python -m pip install -r requirements-captions.txt
```

`ffmpeg` must include the `subtitles` / libass filter.

## Data flow

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

The cached transcript is keyed by narration-audio SHA-256 plus each beat's final timeline start/duration, model size, and requested language. Changing any of those causes a new transcription.

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

- `caption_mode`: `word` (default), `static`, or `off`.
- `caption_style`: `editorial`, `classic`, `hormozi`, `mrbeast`, `karaoke`, `minimal`, or `bounce`. Legacy `white` and `paper` remain accepted aliases.
- `caption_position`: percentage from the bottom, `0`–`50`.
- `caption_whisper_model`: faster-whisper model size. Environment fallback: `VOX_CAPTION_WHISPER_MODEL`; default `base`.
- `caption_required`: when `true`, missing faster-whisper is a hard failure. When `false`, Vox reports the missing dependency and falls back to the previous static beat captions.

The caption renderer writes word timing from the transcription result; it does not estimate timestamps from narration text.

## Scope

This first integration wires word-timed captions into the shared B-roll/C-roll `assemble.py` path. A-roll retains its existing assembly path for now; the caption package is intentionally reusable so A-roll can adopt the same transcript contract in a later change.
