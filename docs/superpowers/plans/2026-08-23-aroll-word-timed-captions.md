# A-roll Word-timed Captions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give newly generated A-roll projects word-timed ASS captions from the same one-time canonical source transcript used for beat segmentation, while legacy projects with no caption setting continue to render without captions.

**Architecture:** Generalize the existing caption transcription cache with a direct source-media builder, then make `asr_beats.py` derive beats from that returned canonical transcript. A-roll assembly only reads `captions/transcript.json`, remaps its genuine source word times onto the concatenated edit timeline in memory, and passes that derived view to the existing ASS renderer; it never invokes ASR or writes another transcript.

**Tech Stack:** Python 3 standard library, `faster-whisper`, ffmpeg/ffprobe with libass, `unittest`, existing `scripts/captions` package, Git, GitHub CLI.

**Canonical ASR profiles:** A-roll uses local faster-whisper only. Measured validation makes `small` / `cpu` / `int8` the operational multilingual default; `large-v3-turbo` / `cpu` / `int8` remains the configurable accurate profile. On the 15.583333 s fixture, warm uncached transcription was 6.198 s (RTF 0.398) for small and 17.496 s (RTF 1.123) for large (2.82x slower); cold-run observed peaks were at least 519 MiB and 1.64 GiB, respectively. Both produced the expected 20 monotonic words in 3 segments. English-only callers may explicitly choose a Distil-Whisper model through faster-whisper with `cpu` / `int8`, but Distil-Whisper is not the multilingual default. Model, device, and compute type remain configurable. No profile depends on the target PC's 2 GB GPU. B-roll/C-roll retain their existing `base` / `auto` / `default` settings unless explicitly configured. Both paths preserve one canonical `captions/transcript.json`; A-roll assembly never transcribes. Do not add original `openai-whisper`, Qwen3-ASR, whisper.cpp, llama.cpp, vLLM, or bitsandbytes.

---

## File map

- Create `tests/test_aroll_captions.py`: focused source-cache, segmentation, remapping, assembly, compatibility, and no-double-transcription tests.
- Modify `scripts/captions/transcription.py`: reusable source-media cache builder plus shared model/CPU-fallback execution; preserve B/C timeline behavior.
- Modify `scripts/asr_beats.py`: local canonical transcript consumption, safe word flattening, explicit word mode, and removal of unused remote-ASR artifacts.
- Modify `scripts/aroll_assemble.py`: edit-timeline transcript remapping, shared ASS generation, word/off mode handling, and deterministic source-audio mapping.
- Modify `references/captions.md`: distinguish B/C narration transcription from A-roll source transcription and document compatibility defaults.
- Modify `SKILL.md`: document A-roll shared transcript and caption controls.
- Modify `AGENTS.md`: update A-roll prerequisites and shared transcript summary.
- Keep `scripts/captions/subtitles.py`, `scripts/captions/subtitle_utils.py`, and `scripts/captions/caption_styles.py` unchanged unless a failing test proves a defect.
- Keep `THIRD_PARTY_NOTICES.md`, dependency files, production configuration, and all YouTube-related functionality unchanged.

## Task 1: Add the direct source-media transcript cache

**Files:**

- Modify: `scripts/captions/transcription.py`
- Create: `tests/test_aroll_captions.py`

- [ ] **Step 1: Write source-cache RED tests**

Create `tests/test_aroll_captions.py` with normal repository path setup and a `SourceTranscriptCacheTests` class. The tests must patch only model loading/transcription, use real temporary source files and real cache JSON, and include these exact behaviors:

```python
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from captions.transcription import build_source_transcript
from captions.subtitles import generate_ass
from captions.subtitle_utils import ffmpeg_filter_path


class SourceTranscriptCacheTests(unittest.TestCase):
    @staticmethod
    def transcript():
        return {
            "language": "en",
            "segments": [{
                "start": 0.1,
                "end": 0.8,
                "text": "real words",
                "words": [
                    {"word": "real", "start": 0.1, "end": 0.4},
                    {"word": "words", "start": 0.45, "end": 0.8},
                ],
            }],
        }

    def call(self, project, source, **kwargs):
        return build_source_transcript(
            project, source, language=kwargs.get("language", "en"),
            model_size=kwargs.get("model_size", "large-v3-turbo"),
            device=kwargs.get("device", "cpu"),
            compute_type=kwargs.get("compute_type", "int8"),
        )

    def test_source_media_is_both_fingerprint_and_transcription_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            project.mkdir()
            source = Path(tmp) / "source video.mp4"
            source.write_bytes(b"canonical-source")
            with patch("captions.transcription._load_model", return_value=object()), patch(
                "captions.transcription._transcribe_with_model",
                return_value=self.transcript(),
            ) as transcribe:
                result = self.call(project, source)
            self.assertEqual(transcribe.call_args.args[1], source)
            self.assertEqual(result["segments"][0]["words"][0]["start"], 0.1)
            self.assertFalse((project / "_source_audio.mp3").exists())

    def test_unchanged_source_reuses_one_cache_without_rewrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            project.mkdir()
            source = Path(tmp) / "source.mp4"
            source.write_bytes(b"same")
            with patch("captions.transcription._load_model", return_value=object()), patch(
                "captions.transcription._transcribe_with_model",
                return_value=self.transcript(),
            ) as transcribe:
                first = self.call(project, source)
                path = project / "captions" / "transcript.json"
                original = path.read_bytes()
                second = self.call(project, source)
            self.assertEqual(transcribe.call_count, 1)
            self.assertEqual(first, second)
            self.assertEqual(path.read_bytes(), original)

    def test_source_content_change_invalidates_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            project.mkdir()
            source = Path(tmp) / "source.mp4"
            source.write_bytes(b"first")
            with patch("captions.transcription._load_model", return_value=object()), patch(
                "captions.transcription._transcribe_with_model",
                return_value=self.transcript(),
            ) as transcribe:
                self.call(project, source)
                source.write_bytes(b"second")
                self.call(project, source)
            self.assertEqual(transcribe.call_count, 2)

    def test_model_language_and_compute_type_each_invalidate_cache(self):
        changes = [
            ({}, {"model_size": "small"}),
            ({}, {"language": "el"}),
            ({}, {"compute_type": "int8"}),
        ]
        for first_kwargs, second_kwargs in changes:
            with self.subTest(second_kwargs=second_kwargs), tempfile.TemporaryDirectory() as tmp:
                project = Path(tmp) / "project"
                project.mkdir()
                source = Path(tmp) / "source.mp4"
                source.write_bytes(b"source")
                with patch("captions.transcription._load_model", return_value=object()), patch(
                    "captions.transcription._transcribe_with_model",
                    return_value=self.transcript(),
                ) as transcribe:
                    self.call(project, source, **first_kwargs)
                    self.call(project, source, **second_kwargs)
                self.assertEqual(transcribe.call_count, 2)

    def test_invalid_json_and_invalid_word_times_recover_by_retranscribing(self):
        invalid_words = [
            None,
            {"word": "bad", "start": math.nan, "end": 0.5},
            {"word": "bad", "start": 0.1, "end": math.inf},
            {"word": "bad", "start": -0.1, "end": 0.5},
            {"word": "bad", "start": 0.5, "end": 0.5},
        ]
        for invalid_word in invalid_words:
            with self.subTest(invalid_word=invalid_word), tempfile.TemporaryDirectory() as tmp:
                project = Path(tmp) / "project"
                project.mkdir()
                source = Path(tmp) / "source.mp4"
                source.write_bytes(b"source")
                with patch("captions.transcription._load_model", return_value=object()), patch(
                    "captions.transcription._transcribe_with_model",
                    return_value=self.transcript(),
                ) as transcribe:
                    self.call(project, source)
                    path = project / "captions" / "transcript.json"
                    if invalid_word is None:
                        path.write_text("{broken", encoding="utf-8")
                    else:
                        cached = json.loads(path.read_text(encoding="utf-8"))
                        cached["segments"][0]["words"] = [invalid_word]
                        path.write_text(json.dumps(cached), encoding="utf-8")
                    self.call(project, source)
                self.assertEqual(transcribe.call_count, 2)

    def test_auto_device_reuses_existing_cpu_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            project.mkdir()
            source = Path(tmp) / "source.mp4"
            source.write_bytes(b"source")
            error = RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
            with patch(
                "captions.transcription._load_model", side_effect=[error, object()]
            ) as load_model, patch(
                "captions.transcription._transcribe_with_model",
                return_value=self.transcript(),
            ):
                result = self.call(project, source, device="auto")
            self.assertEqual(result["language"], "en")
            self.assertEqual(load_model.call_args_list[0].args[1], "auto")
            self.assertEqual(load_model.call_args_list[1].args[1], "cpu")
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
python -m unittest tests.test_aroll_captions.SourceTranscriptCacheTests -v
```

Expected: import failure for missing `build_source_transcript`. This is the required RED evidence, not a syntax or fixture failure.

- [ ] **Step 3: Implement the minimal shared cache generalization**

In `scripts/captions/transcription.py`:

1. Add a shared `_transcribe_with_fallback(model_size, device, compute_type, operation)` that preserves the current automatic missing-CUDA retry behavior for both model construction and transcription.
2. Add a small cache reader that accepts expected metadata and returns only schema-valid canonical transcripts.
3. Strengthen cache validation so every persisted word has a non-empty string, finite `start`/`end`, `start >= 0`, and `end > start`.
4. Add this public source builder:

```python
def build_source_transcript(
    project_dir: str | Path,
    source_path: str | Path,
    *,
    language: str | None = None,
    model_size: str = "large-v3-turbo",
    device: str = "cpu",
    compute_type: str = "int8",
) -> dict[str, Any]:
    project = Path(project_dir)
    source = Path(source_path)
    if not source.is_file():
        raise FileNotFoundError(f"A-roll source media not found: {source}")
    transcript_path = project / "captions" / "transcript.json"
    metadata = {
        "schema_version": 1,
        "source_fingerprint": _sha256_file(source),
        "model": model_size,
        "requested_language": language,
        "compute_type": compute_type,
    }
    cached = _read_cached_transcript(transcript_path, metadata)
    if cached is not None:
        return cached
    transcript = _transcribe_with_fallback(
        model_size, device, compute_type,
        lambda model: _transcribe_with_model(model, source, language),
    )
    transcript.update(metadata)
    if not _is_valid_cached_transcript(transcript):
        raise RuntimeError("Transcription produced no valid canonical word timestamps")
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(
        json.dumps(transcript, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return transcript
```

5. Refactor `build_timeline_transcript()` to use the same cache reader and fallback helper without changing its fingerprint payload, metadata keys, public signature, per-beat offset behavior, or device-independent cache identity.

The original source media must be passed directly to both `_sha256_file()` and `_transcribe_with_model()`. Do not call ffmpeg and do not create `_source_audio.mp3`.

- [ ] **Step 4: Run source-cache tests GREEN and B/C cache tests for regression safety**

Run:

```powershell
python -m unittest tests.test_aroll_captions.SourceTranscriptCacheTests tests.test_captions.TranscriptCacheTests tests.test_captions.TranscriptMergeTests -v
```

Expected: all selected tests pass, including the existing CUDA fallback, B/C digest, and B/C timeline invalidation tests.

## Task 2: Derive A-roll beats from the canonical transcript

**Files:**

- Modify: `scripts/asr_beats.py`
- Modify: `tests/test_aroll_captions.py`

- [ ] **Step 1: Write beat-generation RED tests**

Add `ArollBeatTests` covering:

```python
import asr_beats


class ArollBeatTests(unittest.TestCase):
    @staticmethod
    def canonical_fixture():
        return {"language": "en", "segments": [{"words": [
            {"word": "Hello", "start": 0.25, "end": 0.7},
            {"word": "world.", "start": 0.8, "end": 1.2},
            {"word": "Again", "start": 2.0, "end": 2.5},
        ]}]}

    def test_canonical_word_field_and_genuine_times_drive_beats(self):
        transcript = {"segments": [{"words": [
            {"word": "Hello", "start": 0.25, "end": 0.7},
            {"word": "world.", "start": 0.8, "end": 1.2},
            {"word": "Again", "start": 2.0, "end": 2.5},
        ]}]}
        words = asr_beats.words_from_transcript(transcript, source_duration=3.0)
        beats = asr_beats.segment_words(words, min_dur=0.5, pause_gap=0.35)
        self.assertEqual(words[0], {"text": "Hello", "start": 0.25, "end": 0.7})
        self.assertEqual(beats[0]["start"], 0.25)
        self.assertEqual(beats[-1]["end"], 2.5)

    def test_invalid_words_are_ignored_and_source_bounds_are_clipped(self):
        transcript = {"segments": [{"words": [
            {"word": "negative", "start": -2.0, "end": -1.0},
            {"word": "nan", "start": math.nan, "end": 0.5},
            {"word": "reversed", "start": 1.0, "end": 0.5},
            {"word": "clipped", "start": 2.8, "end": 3.4},
        ]}]}
        self.assertEqual(
            asr_beats.words_from_transcript(transcript, 3.0),
            [{"text": "clipped", "start": 2.8, "end": 3.0}],
        )

    def test_run_transcribes_once_writes_word_mode_and_omits_remote_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            source = Path(tmp) / "source.mp4"
            source.write_bytes(b"source")
            transcript = self.canonical_fixture()
            with patch.object(asr_beats, "build_source_transcript", return_value=transcript) as build, patch.object(
                asr_beats, "probe_dims", return_value=(1080, 1920)
            ), patch.object(asr_beats, "probe_dur", return_value=4.0):
                asr_beats.run(str(project), str(source), language="en")
            doc = json.loads((project / "beats.json").read_text(encoding="utf-8"))
            self.assertEqual(build.call_count, 1)
            self.assertEqual(build.call_args.args[1], source)
            self.assertEqual(doc["caption_mode"], "word")
            self.assertNotIn("source_audio_url", doc)
            self.assertFalse((project / "_source_audio.mp3").exists())
```

Also retain a direct `segment_words()` legacy-shape test using `{text,start,end}` to prove existing segmentation behavior, and add a test that no valid words raises a clear error rather than producing dummy text or fabricated timing.

- [ ] **Step 2: Run beat tests and verify RED**

Run:

```powershell
python -m unittest tests.test_aroll_captions.ArollBeatTests -v
```

Expected: failure because `words_from_transcript` and `probe_dur` do not exist and `run()` still uses the remote provider.

- [ ] **Step 3: Implement canonical A-roll beat generation**

In `scripts/asr_beats.py`:

- Remove `extract_audio`, `get_provider`, upload, remote job polling, `source_audio_url`, fallback dummy text, and unused imports.
- Import `math` and `build_source_transcript`.
- Add `probe_dur()` using ffprobe's `format=duration`, requiring a finite positive result.
- Add `words_from_transcript(transcript, source_duration)` that flattens segments, reads canonical `word` (with legacy `text` accepted only at this boundary), strips empty text, rejects invalid/non-finite/reversed/out-of-range timing, clips partial overlaps to `[0, source_duration]`, sorts by `(start, end)`, and returns the current `segment_words()` input shape.
- Keep the current sentence/pause/min/max segmentation philosophy and merge behavior. A beat boundary occurs only after a complete input word.
- Make `run()` call `build_source_transcript()` exactly once on `Path(src_path)`, then use its returned object for beats.
- Raise `RuntimeError("A-roll transcription produced no valid timed words")` if flattening or segmentation yields no beats.
- Write `"caption_mode": "word"` explicitly, plus the selected `caption_whisper_model`, `caption_whisper_device`, and `caption_whisper_compute_type` values.
- Keep `source_video`, aspect detection, theme, model, and beat fields used by `aroll_clips.py`.
- Add CLI flags `--model`, `--device`, and `--compute-type` with the initial accurate A-roll defaults `large-v3-turbo`, `cpu`, and `int8`; pass them through without adding dependencies. Document `small`, `cpu`, `int8` as the light fallback and Distil-Whisper through faster-whisper on `cpu`, `int8` as an explicit English-only optimization, not a universal default. If Task 7 measures unreasonable accurate-profile wall time, change only the operational model default to `small` through a focused RED–GREEN update.

Use this call shape:

```python
transcript = build_source_transcript(
    project_dir,
    Path(src_path),
    language=language,
    model_size=model_size,
    device=device,
    compute_type=compute_type,
)
words = words_from_transcript(transcript, probe_dur(src_path))
beats = segment_words(words, max_dur=max_beat_dur)
```

- [ ] **Step 4: Run beat tests GREEN**

Run:

```powershell
python -m unittest tests.test_aroll_captions.ArollBeatTests -v
```

Expected: all A-roll beat tests pass and the test proves one builder call, explicit word mode, direct source input, no `_source_audio.mp3`, no `source_audio_url`, and no fabricated fallback.

## Task 3: Remap source timestamps onto the A-roll edit timeline

**Files:**

- Modify: `scripts/aroll_assemble.py`
- Modify: `tests/test_aroll_captions.py`

- [ ] **Step 1: Write edit-remapping RED tests**

Add `ArollTimelineTests` using genuine source word times and explicit assembled spans:

```python
import aroll_assemble


class ArollTimelineTests(unittest.TestCase):
    def test_words_are_clipped_shifted_and_grouped_by_beat(self):
        transcript = {"language": "en", "segments": [{"words": [
            {"word": "before", "start": 0.0, "end": 0.2},
            {"word": "first", "start": 1.0, "end": 1.4},
            {"word": "boundary", "start": 2.8, "end": 3.2},
            {"word": "second", "start": 5.2, "end": 5.7},
        ]}]}
        spans = [
            {"beat": {"id": 1, "start": 1.0, "end": 3.0}, "output_start": 0.0, "dur": 2.0},
            {"beat": {"id": 2, "start": 5.0, "end": 6.0}, "output_start": 2.0, "dur": 1.0},
        ]
        mapped = aroll_assemble.remap_source_transcript(transcript, spans)
        self.assertEqual(mapped["segments"][0]["beat_id"], 1)
        self.assertEqual(mapped["segments"][0]["words"][0]["start"], 0.0)
        self.assertEqual(mapped["segments"][0]["words"][-1]["end"], 2.0)
        self.assertEqual(mapped["segments"][1]["words"][0]["start"], 2.2)

    def test_non_finite_and_non_positive_mapped_ranges_are_ignored(self):
        transcript = {"language": "en", "segments": [{"words": [
            {"word": "nan", "start": math.nan, "end": 0.5},
            {"word": "inf", "start": 0.5, "end": math.inf},
            {"word": "reversed", "start": 0.8, "end": 0.7},
            {"word": "zero", "start": 0.9, "end": 0.9},
            {"word": "valid", "start": 1.0, "end": 1.4},
        ]}]}
        spans = [{
            "beat": {"id": 1, "start": 0.0, "end": 2.0},
            "output_start": 0.0,
            "dur": 2.0,
        }]
        mapped = aroll_assemble.remap_source_transcript(transcript, spans)
        self.assertEqual(
            [word["word"] for word in mapped["segments"][0]["words"]],
            ["valid"],
        )

    def test_generated_ass_has_only_positive_events_and_sensible_groups(self):
        def ass_seconds(value):
            hours, minutes, rest = value.split(":")
            seconds, centiseconds = rest.split(".")
            return (
                int(hours) * 3600 + int(minutes) * 60 + int(seconds)
                + int(centiseconds) / 100
            )

        transcript = {"language": "en", "segments": [{"words": [
            {"word": "first", "start": 0.1, "end": 0.4},
            {"word": "second", "start": 2.1, "end": 2.5},
        ]}]}
        spans = [
            {"beat": {"id": 1, "start": 0.0, "end": 1.0}, "output_start": 0.0, "dur": 1.0},
            {"beat": {"id": 2, "start": 2.0, "end": 3.0}, "output_start": 1.0, "dur": 1.0},
        ]
        mapped = aroll_assemble.remap_source_transcript(transcript, spans)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "captions.ass"
            self.assertTrue(generate_ass(mapped, output))
            events = [
                line for line in output.read_text(encoding="utf-8").splitlines()
                if line.startswith("Dialogue:")
            ]
        self.assertEqual(len(events), 2)
        self.assertIn("first", events[0])
        self.assertNotIn("second", events[0])
        self.assertIn("second", events[1])
        for event in events:
            fields = event.split(",", 3)
            self.assertGreater(ass_seconds(fields[2]), ass_seconds(fields[1]))
```

- [ ] **Step 2: Run timeline tests and verify RED**

Run:

```powershell
python -m unittest tests.test_aroll_captions.ArollTimelineTests -v
```

Expected: failure because `remap_source_transcript` does not exist.

- [ ] **Step 3: Implement the deterministic remapper**

In `scripts/aroll_assemble.py`, add `remap_source_transcript(transcript, edit_spans)` that:

- iterates each included edit span in output order;
- selects words overlapping `[beat.start, beat.start + span.dur]`, constrained by `beat.end`;
- clips a word that crosses a beat boundary instead of estimating a new time;
- maps with `output_start + (clipped_source_time - beat.start)`;
- clips mapped values to `[output_start, output_start + span.dur]`;
- ignores empty, non-finite, negative, reversed, or zero-length values;
- preserves word text and detected language;
- emits one segment per beat with `beat_id`, `start`, `end`, `text`, and canonical `words`;
- returns no metadata pretending that the derived in-memory edit view is a second cached transcript.

- [ ] **Step 4: Run timeline and shared renderer tests GREEN**

Run:

```powershell
python -m unittest tests.test_aroll_captions.ArollTimelineTests tests.test_captions.SubtitleGenerationTests -v
```

Expected: all tests pass; no ASS event has `end <= start`, and shared group/style behavior remains intact.

## Task 4: Integrate A-roll word/off assembly without changing legacy default

**Files:**

- Modify: `scripts/aroll_assemble.py`
- Modify: `tests/test_aroll_captions.py`

- [ ] **Step 1: Write assembly RED tests**

Add `ArollAssemblyTests` that patch `aroll_assemble.ff`, `probe_dur`, and `generate_ass` only where external media execution is unavoidable. Use real temporary `beats.json`, transcript JSON, and dummy clip/source files. Cover:

```python
class ArollAssemblyTests(unittest.TestCase):
    def make_project(self, root, caption_mode="missing"):
        project = Path(root) / "project's"
        project.mkdir()
        source = project / "source.mp4"
        clip = project / "clip.mp4"
        source.write_bytes(b"source")
        clip.write_bytes(b"clip")
        doc = {
            "mode": "aroll",
            "source_video": str(source),
            "aspect": "9:16",
            "beats": [{
                "id": 1, "start": 1.0, "end": 2.0, "dur": 1.0,
                "narration": "hello", "clip_path": str(clip),
            }],
        }
        if caption_mode != "missing":
            doc["caption_mode"] = caption_mode
        (project / "beats.json").write_text(json.dumps(doc), encoding="utf-8")
        return project

    def run_captured(self, project, generate_result=True):
        calls = []
        def fake_generate(_transcript, output, **_kwargs):
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            Path(output).write_text("[Events]\n", encoding="utf-8")
            return generate_result
        with patch.object(aroll_assemble, "ff", side_effect=calls.append), patch.object(
            aroll_assemble, "probe_dur", return_value=1.0
        ), patch.object(
            aroll_assemble, "generate_ass", side_effect=fake_generate
        ) as generate:
            aroll_assemble.run(str(project))
        return calls, generate

    def test_missing_caption_mode_preserves_legacy_off_behavior(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self.make_project(tmp)
            calls, generate = self.run_captured(project)
        self.assertEqual(generate.call_count, 0)
        self.assertEqual(calls[-1][-2:], ["copy", str(project / "final.mp4")])

    def test_explicit_off_skips_transcript_and_ass(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self.make_project(tmp, "off")
            calls, generate = self.run_captured(project)
            self.assertFalse((project / "captions").exists())
        self.assertEqual(generate.call_count, 0)
        self.assertTrue(calls)

    def test_word_mode_reuses_cache_and_maps_original_audio_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self.make_project(tmp, "word")
            caption_dir = project / "captions"
            caption_dir.mkdir()
            transcript = {"language": "en", "segments": [{"words": [
                {"word": "hello", "start": 1.1, "end": 1.5},
            ]}]}
            (caption_dir / "transcript.json").write_text(
                json.dumps(transcript), encoding="utf-8"
            )
            calls, generate = self.run_captured(project)
        mapped = generate.call_args.args[0]
        self.assertEqual(mapped["segments"][0]["words"][0]["start"], 0.1)
        self.assertFalse(hasattr(aroll_assemble, "build_source_transcript"))
        self.assertFalse(hasattr(aroll_assemble, "_transcribe_with_model"))
        mux_call = next(call for call in calls if "1:a:0" in call)
        self.assertEqual(mux_call.count("1:a:0"), 1)
        final_call = calls[-1]
        self.assertEqual(final_call.count("0:a:0"), 1)
        self.assertEqual(final_call.count("-map"), 2)

    def test_word_mode_uses_windows_safe_ass_filter_and_total_duration(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self.make_project(tmp, "word")
            caption_dir = project / "captions"
            caption_dir.mkdir()
            (caption_dir / "transcript.json").write_text(json.dumps({
                "language": "en", "segments": [{"words": [
                    {"word": "hello", "start": 1.1, "end": 1.5},
                ]}],
            }), encoding="utf-8")
            calls, _generate = self.run_captured(project)
        final_call = calls[-1]
        filter_graph = final_call[final_call.index("-filter_complex") + 1]
        expected = ffmpeg_filter_path(str(project / "captions" / "captions.ass"))
        self.assertIn(expected, filter_graph)
        self.assertEqual(final_call[final_call.index("-t") + 1], "1.00")

    def test_unknown_caption_mode_fails(self):
        with self.assertRaisesRegex(ValueError, "word, off"):
            aroll_assemble._caption_mode({"caption_mode": "static"})

    def test_missing_generated_clips_keep_existing_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self.make_project(tmp, "off")
            doc_path = project / "beats.json"
            doc = json.loads(doc_path.read_text(encoding="utf-8"))
            del doc["beats"][0]["clip_path"]
            doc_path.write_text(json.dumps(doc), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "No beats had a generated clip"):
                aroll_assemble.run(str(project))
```

The structural no-transcription assertions are included above.

- [ ] **Step 2: Run assembly tests and verify RED**

Run:

```powershell
python -m unittest tests.test_aroll_captions.ArollAssemblyTests -v
```

Expected: failures for missing caption-mode helper, no ASS integration, and absent edit-span recording.

- [ ] **Step 3: Implement caption-mode and ASS assembly**

In `scripts/aroll_assemble.py`:

- Import `Path`, `generate_ass`, and `ffmpeg_filter_path`; do not import any transcription producer.
- Add `_caption_mode(doc)` using `doc.get("caption_mode", "off")`. Accept only `word` and `off`; missing means `off`.
- During the existing mux loop, record only successfully included beats as `{beat, output_start, dur}` where `dur` is the same probed duration used by ffmpeg. Increment output time only for included muxed files.
- Keep original source audio extraction, `-map 1:a:0`, and exactly one audio stream in each muxed segment.
- For missing/off mode, retain the existing concat stream-copy path directly to `final.mp4`.
- For word mode, concatenate to `_seg/aroll_concat.mp4`, read only `captions/transcript.json`, call `remap_source_transcript()`, and write `captions/captions.ass` through shared `generate_ass()` with `caption_style` defaulting to `editorial`, `caption_position` defaulting to `10`, and the current aspect dimensions.
- Raise clear errors for a missing/invalid transcript or no rendered events.
- Burn subtitles in a final command shaped as:

```python
ff([
    "-i", concat_path,
    "-filter_complex", f"[0:v]subtitles=filename='{ffmpeg_filter_path(str(ass_path))}'[v]",
    "-map", "[v]", "-map", "0:a:0",
    "-t", f"{total:.2f}",
    "-c:v", "libx264", "-preset", "fast", "-crf", "18",
    "-pix_fmt", "yuv420p", "-c:a", "copy", final,
])
```

This filter changes video only. Do not add silence, a second audio input, a mix, a watermark, or another transcription call.

### Timing refinement implemented after this plan

For robust source-cut and container timing, assembly now independently rounds
parsed A-roll `start`, `end`, and requested duration to centiseconds with
round-half-up before calculating the source interval, then safely floors the
encoded duration. It rejects effective cuts shorter than one second; that is a
narrow 24-fps/AAC rendering constraint, while normal ASR beats remain at least
two seconds. Accepted cuts use lossless PCM WAV audio, trimmed to the effective
encoded duration before concatenation. Video-only H.264 segments are allocated
frames from the cumulative PCM-audio timeline on the global 24-fps grid, then
video and PCM audio are concatenated separately.
Finalization AAC-encodes the single original-speech stream once and
stream-copies the `off` video; `word` re-encodes only its subtitle-filtered
video. No silence, mix, or duplicate speech stream is introduced. Probe
assertions allow at most one 24-fps frame plus one 1024-sample AAC packet
(about 62.5 ms at 48 kHz) for apparent container/stream duration, require
starts near zero, and apply video-frame rounding only once globally rather than
allowing accumulated per-beat drift.

- [ ] **Step 4: Run all focused A-roll tests GREEN**

Run:

```powershell
python -m unittest tests.test_aroll_captions -v
```

Expected: every focused test passes, including missing-mode compatibility, word/off semantics, source-audio mapping, duration, path escaping, remapping, and structural no-double-transcription assertions.

## Task 5: Document only implemented behavior

**Files:**

- Modify: `references/captions.md`
- Modify: `SKILL.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: Update caption documentation after tests are green**

In `references/captions.md`, replace the B/C-only scope text with separate flows:

```text
B-roll/C-roll narration audio per beat
→ one local faster-whisper transcript per narration source
→ final-timeline offsets
→ canonical captions/transcript.json

A-roll original source media
→ one local faster-whisper canonical captions/transcript.json
├→ beat segmentation
└→ ASS captions remapped onto the edit timeline
```

Document:

- shared canonical `{language, segments[{start,end,text,words[{word,start,end}]}]}` semantics;
- B/C modes `word` default, `static`, and `off` unchanged;
- newly generated A-roll writes `word`, explicit A-roll `off` disables captions, and missing A-roll mode means `off` for legacy compatibility;
- A-roll's accurate profile is multilingual `large-v3-turbo` / `cpu` / `int8`, its light fallback is `small` / `cpu` / `int8`, it needs no 2 GB GPU, and it allows explicit English-only Distil-Whisper through faster-whisper on `cpu` / `int8` without making Distil-Whisper universal;
- original `openai-whisper`, Qwen3-ASR, whisper.cpp, llama.cpp, vLLM, and bitsandbytes are not added;
- the A-roll cache hashes and transcribes the same original source media;
- source cache identity is digest + model + requested language + compute type, excluding device;
- A-roll is not transcribed again during assembly;
- all seven styles and `white`/`paper` aliases are shared;
- YouTube ingestion, `pytube`, and `yt-dlp` remain out of scope.

- [ ] **Step 2: Update repository workflow summaries**

In `SKILL.md` and `AGENTS.md`, state that A-roll `asr_beats.py` produces the shared source transcript and `aroll_assemble.py` remaps it for ASS. Update the faster-whisper prerequisite to cover A-roll as well as B/C. Do not describe static A-roll captions, remote ASR fallback, or YouTube input.

- [ ] **Step 3: Verify documentation scope**

Run:

```powershell
git diff -- THIRD_PARTY_NOTICES.md requirements-captions.txt
rg -n "pytube|yt-dlp|YouTube URL|source_audio_url" scripts tests requirements*.txt SKILL.md AGENTS.md references/captions.md
```

Expected: no diff for notices or dependency files; no implementation or dependency for YouTube downloaders; `source_audio_url` appears only in a test asserting its absence, if at all.

## Task 6: Run automated regression and compile validation

**Files:** all changed Python and documentation files.

- [ ] **Step 1: Run the full suite**

```powershell
python -m unittest discover -s tests -v
```

Expected: the original 28 tests plus every new A-roll test pass with zero failures/errors.

- [ ] **Step 2: Compile every changed Python file**

```powershell
python -m py_compile scripts/asr_beats.py scripts/aroll_assemble.py scripts/captions/transcription.py tests/test_aroll_captions.py
```

Expected: exit code 0 and no output.

- [ ] **Step 3: Run static repository checks**

```powershell
git diff --check
git status --short
git diff --stat
git diff
```

Expected: no whitespace errors; only intended source, test, and narrow documentation changes; no generated media, cache, source files, secrets, environment files, dependency additions, or unrelated refactors.

## Task 7: Perform real local transcription and render validation

**Files:** ignored files under `out/_aroll-caption-test/` only; do not stage or commit them.

- [ ] **Step 1: Create a local synthetic spoken A-roll fixture if no safe A-roll project exists**

Use Windows SAPI locally to produce a short WAV saying a fixed sentence, then ffmpeg to combine it with a short animated portrait-format visual. Keep every artifact under `out/_aroll-caption-test/synthetic/`. Do not call MuAPI.

```powershell
$testRoot = Resolve-Path 'out' -ErrorAction SilentlyContinue
if (-not $testRoot) { New-Item -ItemType Directory -Path 'out' | Out-Null }
New-Item -ItemType Directory -Force -Path 'out/_aroll-caption-test/synthetic' | Out-Null
$voice = New-Object -ComObject SAPI.SpVoice
$stream = New-Object -ComObject SAPI.SpFileStream
$wav = (Resolve-Path 'out/_aroll-caption-test/synthetic').Path + '\speech.wav'
$stream.Open($wav, 3, $false)
$voice.AudioOutputStream = $stream
$voice.Speak('Crete has a long history. Paper collage makes the story move.') | Out-Null
$stream.Close()
ffmpeg -y -f lavfi -i "testsrc2=size=540x960:rate=24" -i $wav -shortest -c:v libx264 -pix_fmt yuv420p -c:a aac 'out/_aroll-caption-test/synthetic/source.mp4'
```

If SAPI is unavailable, report the concrete failure and use another installed local speech synthesizer; do not use network TTS.

- [ ] **Step 2: Run real faster-whisper source transcription**

```powershell
python scripts/asr_beats.py out/_aroll-caption-test/synthetic out/_aroll-caption-test/synthetic/source.mp4 --language en --model large-v3-turbo --device cpu --compute-type int8
```

Measure wall time and, where practical, sample the Python process working set to estimate peak RAM while the command runs. Record actual detected/requested language, model, device, compute type, source duration, transcription wall time, approximate peak RAM, segment count, word count, first/last word times, word-timestamp quality, obvious transcription errors, and transcript SHA-256. Verify every word is finite, monotonic, inside source duration, and comes from the spoken fixture rather than Vox-fabricated dummy text.

If `large-v3-turbo` is unreasonably slow for this short fixture, run the same benchmark with `--model small --device cpu --compute-type int8`. Record the measured reason, then add a focused failing default-profile test, change only the A-roll operational model default to `small`, rerun the focused and full suites, and update the docs to report the measured decision. Do not redesign the cache, add an ASR engine, or introduce GPU-specific code.

- [ ] **Step 3: Prove real cache reuse**

Hash and timestamp `captions/transcript.json`, rerun the exact command, and verify the file bytes and last-write time remain unchanged. Record that the second run did not load or execute a model if observable; unchanged bytes/time are the required local cache evidence.

- [ ] **Step 4: Prepare local visual clips without MuAPI**

Copy `source.mp4` to one local `clips/clip_<id>.mp4` per beat and write each `clip_path` into the ignored `beats.json`. This exercises the real A-roll cut, scale/crop, mux, concat, original-source-audio, and caption burn path without pretending to validate a MuAPI generation call.

- [ ] **Step 5: Render and probe word mode**

```powershell
python scripts/aroll_assemble.py out/_aroll-caption-test/synthetic
ffprobe -v error -show_entries stream=index,codec_type,codec_name,width,height,duration -show_entries format=duration -of json out/_aroll-caption-test/synthetic/final.mp4
```

Record video/audio codec, dimensions, stream durations, container duration, and expected assembled duration. Confirm one video and one audio stream, plausible non-truncated duration, and original spoken audio is audible/present.

- [ ] **Step 6: Inspect representative caption frames**

Extract at least three frames at distinct word times into the ignored project and inspect them visually. Confirm captions are burned, the active-word highlight changes, text fits, groups do not bleed across beats, and the existing scale/crop visual path remains. A-roll currently adds no watermark; confirm this unchanged behavior rather than adding one.

- [ ] **Step 7: Render and probe off mode**

Copy the word render to an ignored comparison name, set `caption_mode` to `off`, rerun assembly, and probe it. Confirm no visible captions, one original-source audio stream, the same intended edit duration, and valid visuals. Restore `caption_mode` to `word` only if needed for final artifact inspection; nothing under `out/` is committed.

## Task 8: Final verification, explicit commit, push, and draft PR

**Files:** only intentionally changed tracked files.

- [ ] **Step 1: Run fresh final verification**

```powershell
python -m unittest discover -s tests -v
python -m py_compile scripts/asr_beats.py scripts/aroll_assemble.py scripts/captions/transcription.py tests/test_aroll_captions.py
git diff --check
git status --short
git diff --stat
git diff
```

Read the complete outputs. Do not claim completion unless all commands succeed and every changed line is reviewed.

- [ ] **Step 2: Audit prohibited and generated content**

```powershell
git status --short --untracked-files=all
git diff --name-only 69d5c81ba82c52a5dbd63a98efdf7e62c1ac893a
rg -n "MUAPI_API_KEY|OPENAI_API_KEY|api[_-]?key|pytube|yt-dlp" scripts tests docs SKILL.md AGENTS.md references requirements*.txt
```

Expected: no generated media/model cache/source media/API key values/`.env`; no `out/_aroll-caption-test` tracked; no YouTube downloader implementation or dependency; no unrelated B/C generation refactor. Existing documentation references to environment-variable names are acceptable only when they contain no values.

- [ ] **Step 3: Stage explicit files only and commit the feature**

```powershell
git add scripts/asr_beats.py
git add scripts/aroll_assemble.py
git add scripts/captions/transcription.py
git add tests/test_aroll_captions.py
git add references/captions.md
git add SKILL.md
git add AGENTS.md
git add docs/superpowers/plans/2026-08-23-aroll-word-timed-captions.md
git diff --cached --check
git diff --cached
git commit -m "feat: add word-timed captions to A-roll"
```

Do not use `git add .` or `git add -A`. The already committed design/spec remains part of the branch history. Do not stage `THIRD_PARTY_NOTICES.md`, dependencies, generated media, or unrelated files.

- [ ] **Step 4: Push without force**

```powershell
git push -u origin codex/aroll-word-timed-captions
```

- [ ] **Step 5: Create a draft PR into main**

```powershell
gh pr create --draft --base main --head codex/aroll-word-timed-captions --title "Add word-timed captions to A-roll" --body-file out/_aroll-caption-test/pr-body.md
```

Construct the body in an ignored temporary file and include canonical transcript reuse, shared renderer/styles, source-audio preservation, legacy missing-mode compatibility, explicit word/off behavior, unit totals, real transcription/cache evidence, real word/off render and ffprobe evidence, B/C regression status, and explicit statements that there is no YouTube ingest, no `pytube`, no `yt-dlp`, no merge authorization, and the PR must remain draft. Delete or leave the body file only under ignored `out/`; do not commit it.

- [ ] **Step 6: Verify remote and local final state**

```powershell
gh pr view --json url,isDraft,baseRefName,headRefName,state
git status -sb
git rev-parse HEAD
```

Expected: draft PR, base `main`, head `codex/aroll-word-timed-captions`, open state, clean working tree, and no merge.

## Completion report checklist

The final report must provide all 40 items requested by the task: starting SHA, branch, files changed, old/new architecture, transcript location/schema, remote/local ASR decision, proof of one transcription, cache identity/reuse/invalidation, beat behavior, shared styles/modes, ASS result, real model/device/compute type, transcript counts, render/ffprobe details, source audio, visual caption/off confirmations, B/C regression, test total, compile/diff checks, prohibited-content audits, commit SHA, draft PR URL, final status, and readiness recommendation.

Do not merge. End the final report exactly:

```text
VOX A-ROLL WORD-TIMED CAPTIONS READY FOR NIKOS REVIEW
```
