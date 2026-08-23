# MuAPI Director — Agent Guide

This repository is an **Agent Skill**: a self-contained workflow that turns a topic, talking-head video, or photo into a finished Vox-style paper-collage video (script → keyframes → motion → voice-over → music → captions) powered by **MuAPI** (`api.muapi.ai`). It also includes custom ComfyUI nodes for Google Veo 3.1.

## How to use it (for the agent)

1. Read **`SKILL.md`** — the full workflow covering B-roll, A-roll, and C-roll modes and approval gates.
2. Work one project at a time under `out/<project>/`, driven by a single `beats.json`.
3. Run the pipeline stages in **`scripts/`**:
   - **B-roll (topic → film):** `style_bakeoff.py → keyframes.py → clips.py → audio.py → assemble.py`
   - **A-roll (talking-head → collage):** `asr_beats.py → aroll_clips.py → aroll_assemble.py`
   - **C-roll (photo anchor → collage):** `croll_keyframes.py → clips.py → audio.py → assemble.py`

## Requirements

- `MUAPI_API_KEY` in the environment — https://muapi.ai
- `ffmpeg` + `ffprobe`
- Python 3 with `Pillow`
- A-roll requires local faster-whisper for `asr_beats.py`: `python -m pip install -r requirements-captions.txt`. B-roll/C-roll word captions use the same dependency but retain their static-caption fallback when it is absent (unless captions are required).

Caption behavior:

- B-roll/C-roll retain `caption_mode` values `word` (default), `static`, and `off`.
- A-roll has explicit `word` and `off` modes; a missing A-roll mode remains legacy `off`. `asr_beats.py` creates new A-roll projects with `word` and writes one canonical, original-source transcript to `captions/transcript.json`; assembly reuses it for edit-timeline caption remapping, without a second transcription or fabricated timings. Original source-audio segments are preserved.
- A-roll's operational multilingual default is `small` / `cpu` / `int8`; `large-v3-turbo` / `cpu` / `int8` remains the configurable accurate profile. On the 15.583333 s fixture, warm uncached runs were 6.198 s (RTF 0.398) for small and 17.496 s (RTF 1.123) for large (2.82x slower); cold-run observed peaks were at least 519 MiB and 1.64 GiB, respectively. Both produced the expected 20 monotonic words in 3 segments. An explicit English-only Distil-Whisper model through faster-whisper may use CPU/int8, but is not the multilingual default. No profile depends on a 2 GB GPU.
- Shared styles are `editorial`, `classic`, `hormozi`, `mrbeast`, `karaoke`, `minimal`, and `bounce`; `white` and `paper` remain aliases.
- This phase adds no original `openai-whisper`, Qwen3-ASR, `whisper.cpp`, `llama.cpp`, vLLM, bitsandbytes, YouTube ingest, `pytube`, `yt-dlp`, or remote `source_audio_url` flow.

See `references/captions.md` for caption modes, styles, and local faster-whisper configuration.
