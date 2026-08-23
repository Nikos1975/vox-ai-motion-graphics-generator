# A-roll Word-timed Captions Design

## Scope

Add word-timed captions to the existing A-roll pipeline by sharing the canonical faster-whisper transcript and ASS renderer already used by B-roll and C-roll. A-roll must transcribe its source once, reuse that persisted transcript for both beat segmentation and captions, preserve source audio and existing visual editing, and support `caption_mode` values `word` and `off`. Newly generated A-roll projects explicitly write `"caption_mode": "word"`; legacy A-roll projects with no `caption_mode` assemble as `off` so their prior output behavior remains unchanged.

This change does not add YouTube URL ingestion, `pytube`, `yt-dlp`, playlist or channel handling, new production configuration, new caption styles, or unrelated B-roll/C-roll refactoring. It does not change A-roll watermark behavior.

## Existing Architecture

`scripts/asr_beats.py` extracts `_source_audio.mp3`, uploads it through the configured MuAPI provider, submits a remote `whisper` job, and expects a flat list of words shaped as `{text, start, end}`. It converts those words into beats using maximum duration, minimum duration, sentence endings, and pauses. The word timing is then discarded: `beats.json` retains only each beat's `start`, `end`, `dur`, and narration text. Failed or empty ASR produces dummy text with fabricated beat-level timing.

`scripts/aroll_assemble.py` extracts original source audio for each beat, muxes each available generated visual clip with that audio, and concatenates the muxed segments. It does not read a transcript, generate ASS, or implement caption modes. The current remote ASR path requires an upload and is incompatible with the reusable local caption transcription cache. The provider's completed-job adapter normally exposes an output URL, while `asr_beats.py` expects a word dictionary.

## Chosen Architecture

Narrowly generalize `scripts/captions/transcription.py` with a source-file transcript builder. The builder accepts the original A-roll source media directly, invokes the existing faster-whisper loading, word timestamp extraction, and automatic CUDA-to-CPU fallback code, validates the result, and persists the canonical transcript at:

```text
out/<project>/captions/transcript.json
```

The semantic transcript remains:

```json
{
  "language": "en",
  "segments": [
    {
      "start": 0.0,
      "end": 2.4,
      "text": "Crete matters",
      "words": [
        {"word": "Crete", "start": 0.0, "end": 0.41}
      ]
    }
  ]
}
```

Cache metadata surrounds, but does not alter, these semantics. `scripts/asr_beats.py` calls this builder once and derives its beats from the returned transcript. `scripts/aroll_assemble.py` reads the same persisted transcript; it never calls a transcriber. It maps source-relative words into the concatenated edit timeline in memory, assigns beat identifiers to prevent inappropriate grouping across cuts, and passes that derived view directly to the existing `generate_ass()` renderer. No second transcript file is written.

Local faster-whisper becomes the canonical A-roll ASR. The safe multilingual A-roll default is model `small`, device `cpu`, and compute type `int8`, so the workflow does not depend on the available 2 GB GPU. English-only users may explicitly select `distil-small.en` with `cpu` / `int8` as an optimization; Distil-Whisper is not the universal default because it is not the multilingual choice. The existing MuAPI Whisper path is superseded rather than retained as a second backend because it adds a paid remote request, has an incompatible provider result contract, and would reintroduce parallel transcription behavior. The former `source_audio_url` field is not preserved because neither `aroll_clips.py` nor `aroll_assemble.py` consumes it. MuAPI remains unchanged for A-roll visual generation.

## Source Transcript Cache

The source transcript cache identity includes:

- SHA-256 of the original source-media content passed to faster-whisper
- Whisper model
- requested language
- compute type

The original `source_video` or source-media path is both the faster-whisper input and the SHA-256 fingerprint input, so the cached identity always describes the bytes that were transcribed. `_source_audio.mp3` is not extracted, regenerated, or retained unless real implementation testing proves direct source-media transcription cannot satisfy the workflow. The execution device is excluded because it does not request different transcript semantics. A valid unchanged cache is returned without rewriting the file. Changed source content, model, language, or compute type invalidates it. Invalid JSON, missing required structure, malformed timing, non-finite timing, negative timing, or `end <= start` causes retranscription.

The source builder reuses the existing `_load_model()`, `_transcribe_with_model()`, missing-CUDA detection, and CPU fallback logic. Shared helpers may be narrowed or generalized to avoid duplicated cache and validation code, but the B-roll/C-roll timeline transcript behavior and cache identity remain compatible.

## Beat Segmentation

A-roll beat segmentation flattens valid canonical words while preserving their genuine `word`, `start`, and `end` values. It accepts either the canonical `word` field or the legacy `text` field at the segmentation boundary only, so the existing segmentation helper remains compatible with callers while canonical transcripts stay unchanged.

Words are ordered deterministically. Invalid, empty, negative, non-finite, zero-length, reversed, or wholly out-of-duration words are ignored. Words partially overlapping the source duration are clipped to its bounds. Beat boundaries continue to use the current maximum-duration, minimum-duration, sentence-ending, and pause rules. A boundary is placed only after a complete word, never by inventing an interpolated timestamp. Each beat start is the first included word start and each beat end is the last included word end, constrained to the source duration; every emitted beat has `0 <= start < end <= source_duration`.

Dummy transcript text and fabricated fallback timestamps are removed. Missing faster-whisper or a transcript with no valid timed words is a clear error because A-roll cannot build trustworthy beats without source timing.

## A-roll Caption Timeline

The persisted transcript uses source time. A-roll output can concatenate only beats with available generated clips, so assembly derives an edit-timeline view without changing the cache:

1. For each included beat, select canonical words that overlap its source interval.
2. Clip overlapping words to the beat interval.
3. Shift each word by `output_segment_start - beat_start`.
4. Clip to the actual muxed segment duration.
5. Drop any word whose mapped end is not greater than its mapped start.
6. Attach the beat identifier to its derived segment so caption groups do not span cuts.

This mapping preserves genuine within-source word timing and performs no text-based estimation. The same cached transcript object is therefore the source for both beat generation and ASS captions, while the ASS times match the final concatenated edit.

## Assembly

`scripts/aroll_assemble.py` keeps its existing per-beat original-audio extraction and visual scaling/cropping/fps behavior. It records deterministic output segment spans while muxing. The concat result remains the basis of `final.mp4`.

For explicit `caption_mode=word`, assembly loads `captions/transcript.json`, creates the edit-timeline view, writes `captions/captions.ass` using the shared renderer and requested `caption_style` / `caption_position`, and burns it into the concatenated video with ffmpeg/libass. Newly generated `beats.json` files explicitly select this mode. The subtitles filter uses the shared `ffmpeg_filter_path()` helper so Windows drive letters, spaces, and apostrophes are safe. The final encode maps video from the filtered concat and audio from the concatenated A-roll source-audio track exactly once.

For explicit `caption_mode=off`, assembly skips transcript loading, ASS generation, and the subtitle filter. A missing `caption_mode` is also treated as `off` for compatibility with A-roll projects created before caption support. A-roll has no existing static caption mode, so this design does not introduce one. Unsupported modes fail with a clear `ValueError`.

The final mapping explicitly preserves one audio stream derived from the source recording. Caption filtering changes only video. Output duration is bounded to the assembled segment timeline so captions cannot extend or truncate audio. Existing behavior for missing generated clips and the absence of an A-roll watermark remains unchanged.

## Error Handling

- Missing faster-whisper during `asr_beats.py` raises the shared dependency error with the existing installation instruction.
- Corrupt or semantically invalid transcript cache triggers retranscription.
- Empty valid-word output raises an actionable error instead of producing fabricated beats.
- Missing transcript during word-mode assembly raises an actionable error directing the user to run `asr_beats.py`.
- Empty edit-timeline captions raise an error in word mode rather than silently producing an uncaptioned render.
- `caption_mode=off` remains independent of transcript and ASS availability.
- Invalid caption styles and positions continue to use the shared renderer's validation.

## Testing and Validation

Focused unit tests will cover canonical source structure, genuine timestamp preservation, beat derivation, the single-transcription data path, source digest reuse/invalidation, model/language/compute-type invalidation, corrupt cache recovery, word/off modes, audio mapping, output duration mapping, Windows ASS escaping, beat boundaries, invalid times, valid ASS event durations, temporal grouping, and existing A-roll compatibility.

Implementation follows red-green-refactor cycles. The existing 28-test baseline and four-file compile baseline are recorded before changes. After implementation, all tests and compile checks are rerun, followed by `git diff --check` and a full diff review.

Real local validation uses an ignored copy of a safe existing A-roll project, or a locally generated short talking-head-style fixture if none exists. It uses the multilingual A-roll defaults faster-whisper `small`, `device=cpu`, and `compute_type=int8`, confirms real monotonic words and cache reuse, renders word and off variants without MuAPI generation, probes video/audio streams and duration, and inspects representative frames for caption burn-in. Generated media, model cache data, source media, secrets, and the ignored validation project are not committed.

## Documentation and Delivery

`references/captions.md`, `SKILL.md`, and `AGENTS.md` will be updated narrowly to describe shared canonical semantics, B-roll/C-roll per-beat narration mapping, A-roll source-speech transcription reuse, shared styles, supported A-roll caption modes, and deferred YouTube ingestion. `THIRD_PARTY_NOTICES.md` remains unchanged unless implementation introduces a new adapted third-party work.

After validation, only intentional files are staged explicitly. The feature is committed as `feat: add word-timed captions to A-roll`, pushed without force to `codex/aroll-word-timed-captions`, and opened as a draft PR into `main`. It is not marked ready and is not merged.
