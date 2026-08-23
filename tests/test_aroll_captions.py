import hashlib
import ast
import json
import math
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from captions.transcription import build_source_transcript, load_cached_source_transcript
from captions.subtitle_utils import ffmpeg_filter_path
import asr_beats
import aroll_assemble
from captions.subtitles import generate_ass


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
        options = {
            "language": kwargs.get("language", "en"),
            "device": kwargs.get("device", "cpu"),
            "compute_type": kwargs.get("compute_type", "int8"),
        }
        if "model_size" in kwargs:
            options["model_size"] = kwargs["model_size"]
        return build_source_transcript(project, source, **options)

    def test_unconfigured_source_uses_operational_small_cpu_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            project.mkdir()
            source = Path(tmp) / "source.mp4"
            source.write_bytes(b"source")
            with patch("captions.transcription._load_model", return_value=object()) as load_model, patch(
                "captions.transcription._transcribe_with_model",
                return_value=self.transcript(),
            ):
                build_source_transcript(project, source)
            self.assertEqual(load_model.call_args.args, ("small", "cpu", "int8"))

    def test_light_cpu_profile_is_configurable(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            project.mkdir()
            source = Path(tmp) / "source.mp4"
            source.write_bytes(b"source")
            with patch("captions.transcription._load_model", return_value=object()) as load_model, patch(
                "captions.transcription._transcribe_with_model",
                return_value=self.transcript(),
            ):
                self.call(project, source, model_size="small", device="cpu", compute_type="int8")
            self.assertEqual(load_model.call_args.args, ("small", "cpu", "int8"))

    def test_unconfigured_source_cache_round_trips_through_read_only_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            project.mkdir()
            source = Path(tmp) / "source.mp4"
            source.write_bytes(b"source")
            with patch("captions.transcription._load_model", return_value=object()), patch(
                "captions.transcription._transcribe_with_model",
                return_value=self.transcript(),
            ) as transcribe:
                built = build_source_transcript(project, source, language="en")
                loaded = load_cached_source_transcript(project, source, language="en")
        self.assertEqual(transcribe.call_count, 1)
        self.assertEqual(loaded, built)

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

    def test_invalid_utf8_cache_is_retranscribed(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            project.mkdir()
            source = Path(tmp) / "source.mp4"
            source.write_bytes(b"source")
            cache_path = project / "captions" / "transcript.json"
            cache_path.parent.mkdir()
            cache_path.write_bytes(b"\xff")
            with patch("captions.transcription._load_model", return_value=object()), patch(
                "captions.transcription._transcribe_with_model",
                return_value=self.transcript(),
            ) as transcribe:
                result = self.call(project, source)
            self.assertEqual(transcribe.call_count, 1)
            self.assertEqual(result["segments"][0]["words"][0]["word"], "real")

    def test_device_change_reuses_cache_without_retranscribing_or_rewriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            project.mkdir()
            source = Path(tmp) / "source.mp4"
            source.write_bytes(b"same")
            with patch("captions.transcription._load_model", return_value=object()) as load_model, patch(
                "captions.transcription._transcribe_with_model",
                return_value=self.transcript(),
            ) as transcribe:
                self.call(project, source, device="cpu")
                path = project / "captions" / "transcript.json"
                original = path.read_bytes()
                self.call(project, source, device="auto")
            self.assertEqual(load_model.call_count, 1)
            self.assertEqual(transcribe.call_count, 1)
            self.assertEqual(path.read_bytes(), original)

    def test_empty_source_transcripts_are_not_persisted_or_reused(self):
        empty_transcripts = [
            {"language": "en", "segments": []},
            {"language": "en", "segments": [{"words": []}]},
        ]
        for transcript in empty_transcripts:
            with self.subTest(transcript=transcript), tempfile.TemporaryDirectory() as tmp:
                project = Path(tmp) / "project"
                project.mkdir()
                source = Path(tmp) / "source.mp4"
                source.write_bytes(b"source")
                with patch("captions.transcription._load_model", return_value=object()), patch(
                    "captions.transcription._transcribe_with_model",
                    side_effect=[transcript, transcript],
                ) as transcribe:
                    with self.assertRaisesRegex(
                        RuntimeError, "Transcription produced no valid canonical word timestamps"
                    ):
                        self.call(project, source)
                    with self.assertRaisesRegex(
                        RuntimeError, "Transcription produced no valid canonical word timestamps"
                    ):
                        self.call(project, source)
                self.assertEqual(transcribe.call_count, 2)
                self.assertFalse((project / "captions" / "transcript.json").exists())

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
            ({}, {"model_size": "large-v3-turbo"}),
            ({}, {"language": "el"}),
            ({}, {"compute_type": "default"}),
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

    def test_malformed_canonical_fields_are_retranscribed_and_rejected_by_read_only_loader(self):
        cases = {
            "missing language": lambda transcript: transcript.pop("language"),
            "non-string language": lambda transcript: transcript.__setitem__("language", 7),
            "missing segment start": lambda transcript: transcript["segments"][0].pop("start"),
            "invalid segment start": lambda transcript: transcript["segments"][0].__setitem__("start", "bad"),
            "nonfinite segment start": lambda transcript: transcript["segments"][0].__setitem__("start", math.nan),
            "negative segment start": lambda transcript: transcript["segments"][0].__setitem__("start", -0.1),
            "missing segment end": lambda transcript: transcript["segments"][0].pop("end"),
            "invalid segment end": lambda transcript: transcript["segments"][0].__setitem__("end", "bad"),
            "nonfinite segment end": lambda transcript: transcript["segments"][0].__setitem__("end", math.inf),
            "negative segment end": lambda transcript: transcript["segments"][0].__setitem__("end", -0.1),
            "reversed segment times": lambda transcript: transcript["segments"][0].update(start=0.8, end=0.1),
            "non-string segment text": lambda transcript: transcript["segments"][0].__setitem__("text", 7),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
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
                    cached = json.loads(path.read_text(encoding="utf-8"))
                    mutate(cached)
                    path.write_text(json.dumps(cached), encoding="utf-8")
                    self.assertIsNone(load_cached_source_transcript(project, source, language="en"))
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


class ArollBeatTests(unittest.TestCase):
    @staticmethod
    def canonical_fixture():
        return {"language": "en", "segments": [{"words": [
            {"word": "Hello", "start": 0.25, "end": 0.7},
            {"word": "world.", "start": 0.8, "end": 1.2},
            {"word": "Again", "start": 2.0, "end": 2.5},
        ]}]}

    def test_canonical_word_field_and_genuine_times_drive_beats(self):
        words = asr_beats.words_from_transcript(self.canonical_fixture(), source_duration=3.0)
        beats = asr_beats.segment_words(words, min_dur=0.5, pause_gap=0.35)
        self.assertEqual(words[0], {"text": "Hello", "start": 0.25, "end": 0.7})
        self.assertEqual(beats[0]["start"], 0.25)
        self.assertEqual(beats[-1]["end"], 2.5)

    def test_invalid_words_are_ignored_and_source_bounds_are_clipped(self):
        transcript = {"segments": [{"words": [
            {"word": "negative", "start": -2.0, "end": -1.0},
            {"word": "nan", "start": math.nan, "end": 0.5},
            {"word": "reversed", "start": 1.0, "end": 0.5},
            {"word": "outside", "start": 3.1, "end": 4.0},
            {"word": "clipped", "start": 2.8, "end": 3.4},
        ]}]}
        self.assertEqual(
            asr_beats.words_from_transcript(transcript, 3.0),
            [{"text": "clipped", "start": 2.8, "end": 3.0}],
        )

    def test_malformed_transcript_shapes_are_ignored(self):
        self.assertEqual(asr_beats.words_from_transcript({"segments": None}, 3.0), [])
        self.assertEqual(asr_beats.words_from_transcript({"segments": [{"words": None}]}, 3.0), [])

    def test_legacy_text_words_preserve_segmentation(self):
        words = [
            {"text": "First", "start": 0.0, "end": 0.5},
            {"text": "sentence.", "start": 0.5, "end": 1.0},
            {"text": "Second", "start": 1.5, "end": 2.0},
        ]
        self.assertEqual(
            asr_beats.segment_words(words, min_dur=0.5, pause_gap=0.35),
            [
                {"start": 0.0, "end": 1.0, "text": "First sentence."},
                {"start": 1.5, "end": 2.0, "text": "Second"},
            ],
        )

    def test_run_transcribes_once_writes_word_mode_and_omits_remote_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            source = Path(tmp) / "source.mp4"
            source.write_bytes(b"source")
            with patch.object(
                asr_beats, "build_source_transcript", return_value=self.canonical_fixture()
            ) as build, patch.object(asr_beats, "probe_dims", return_value=(1080, 1920)), patch.object(
                asr_beats, "probe_dur", return_value=4.0
            ):
                asr_beats.run(str(project), str(source), language="en")
            doc = json.loads((project / "beats.json").read_text(encoding="utf-8"))
            self.assertNotIn("source_audio_url", doc)
            self.assertFalse((project / "_source_audio.mp3").exists())
        self.assertEqual(build.call_count, 1)
        self.assertEqual(build.call_args.args[1], source)
        self.assertEqual(build.call_args.kwargs["model_size"], "small")
        self.assertEqual(build.call_args.kwargs["device"], "cpu")
        self.assertEqual(build.call_args.kwargs["compute_type"], "int8")
        self.assertEqual(doc["caption_mode"], "word")
        self.assertEqual(doc["caption_whisper_model"], "small")
        self.assertEqual(doc["caption_whisper_device"], "cpu")
        self.assertEqual(doc["caption_whisper_compute_type"], "int8")

    def test_cli_model_default_uses_operational_small_profile(self):
        tree = ast.parse((ROOT / "scripts" / "asr_beats.py").read_text(encoding="utf-8"))
        model_options = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "--model"
        ]
        self.assertEqual(len(model_options), 1)
        defaults = [
            keyword.value
            for keyword in model_options[0].keywords
            if keyword.arg == "default"
        ]
        self.assertEqual(len(defaults), 1)
        self.assertIsInstance(defaults[0], ast.Name)
        self.assertEqual(defaults[0].id, "AROLL_DEFAULT_MODEL_SIZE")
        self.assertEqual(asr_beats.AROLL_DEFAULT_MODEL_SIZE, "small")

    def test_run_rejects_transcript_without_valid_timed_words(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            source = Path(tmp) / "source.mp4"
            source.write_bytes(b"source")
            with patch.object(asr_beats, "build_source_transcript", return_value={"segments": []}), patch.object(
                asr_beats, "probe_dur", return_value=4.0
            ):
                with self.assertRaisesRegex(RuntimeError, "A-roll transcription produced no valid timed words"):
                    asr_beats.run(str(project), str(source), language="en")


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

        self.assertEqual(mapped["language"], "en")
        self.assertEqual(mapped["segments"][0]["beat_id"], 1)
        self.assertEqual(mapped["segments"][0]["words"][0]["start"], 0.0)
        self.assertEqual(mapped["segments"][0]["words"][-1]["end"], 2.0)
        self.assertEqual(mapped["segments"][1]["words"][0]["start"], 2.2)

    def test_non_finite_and_non_positive_mapped_ranges_are_ignored(self):
        transcript = {"language": "en", "segments": [{"words": [
            {"word": "nan", "start": math.nan, "end": 0.5},
            {"word": "inf", "start": 0.5, "end": math.inf},
            {"word": "negative", "start": -0.2, "end": 0.5},
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

    def test_included_beat_without_valid_overlapping_words_keeps_empty_segment(self):
        transcript = {"language": "en", "segments": [{"words": [
            {"word": "outside", "start": 4.0, "end": 4.5},
            {"word": "invalid", "start": 1.0, "end": 1.0},
        ]}]}
        spans = [{
            "beat": {"id": "silent-cut", "start": 1.0, "end": 2.0},
            "output_start": 3.0,
            "dur": 1.0,
        }]

        mapped = aroll_assemble.remap_source_transcript(transcript, spans)

        self.assertEqual(mapped["segments"], [{
            "beat_id": "silent-cut",
            "start": 3.0,
            "end": 4.0,
            "text": "",
            "words": [],
        }])

    def test_source_transcript_is_not_mutated_or_persisted_as_a_second_cache(self):
        transcript = {"language": "el", "segments": [{"words": [
            {"word": " alpha ", "start": 1.0, "end": 1.5},
        ]}]}
        original = json.loads(json.dumps(transcript))
        spans = [{
            "beat": {"id": "cut-a", "start": 1.0, "end": 2.0},
            "output_start": 4.0,
            "dur": 1.0,
        }]

        mapped = aroll_assemble.remap_source_transcript(transcript, spans)

        self.assertEqual(transcript, original)
        self.assertEqual(mapped["language"], "el")
        self.assertEqual(mapped["segments"][0]["words"][0]["word"], " alpha ")
        self.assertEqual(mapped["segments"][0]["words"][0]["start"], 4.0)
        self.assertEqual(set(mapped), {"language", "segments"})

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


class ArollAssemblyTests(unittest.TestCase):
    def make_project(self, root, caption_mode="missing", *, caption_style=None):
        project = Path(root) / "project's"
        project.mkdir()
        source = project / "source.mp4"
        clip = project / "clip.mp4"
        source.write_bytes(b"source")
        clip.write_bytes(b"clip")
        doc = {
            "mode": "aroll",
            "source_video": str(source),
            "language": "en",
            "aspect": "9:16",
            "caption_whisper_model": "large-v3-turbo",
            "caption_whisper_device": "cpu",
            "caption_whisper_compute_type": "int8",
            "beats": [{
                "id": 1, "start": 1.0, "end": 2.0, "dur": 1.0,
                "narration": "hello", "clip_path": str(clip),
            }],
        }
        if caption_mode != "missing":
            doc["caption_mode"] = caption_mode
        if caption_style is not None:
            doc["caption_style"] = caption_style
        (project / "beats.json").write_text(json.dumps(doc), encoding="utf-8")
        return project

    def write_transcript(self, project, words=None, **overrides):
        caption_dir = project / "captions"
        caption_dir.mkdir(exist_ok=True)
        doc = json.loads((project / "beats.json").read_text(encoding="utf-8"))
        transcript = {
            "schema_version": 1,
            "source_fingerprint": hashlib.sha256(
                Path(doc["source_video"]).read_bytes()
            ).hexdigest(),
            "model": doc["caption_whisper_model"],
            "requested_language": doc["language"],
            "compute_type": doc["caption_whisper_compute_type"],
            "language": "en", "segments": [{
                "start": 1.0,
                "end": 2.0,
                "text": "hello",
                "words": words or [{"word": "hello", "start": 1.1, "end": 1.5}],
            }],
        }
        transcript.update(overrides)
        (caption_dir / "transcript.json").write_text(json.dumps(transcript), encoding="utf-8")

    def run_captured(self, project, generate_result=True, durations=None):
        calls = []
        durations = iter(durations or [1.0])

        def fake_probe(_path):
            try:
                return next(durations)
            except StopIteration:
                return 1.0

        def fake_generate(_transcript, output, **_kwargs):
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            Path(output).write_text("[Events]\n", encoding="utf-8")
            return generate_result

        with patch.object(aroll_assemble, "ff", side_effect=calls.append), patch.object(
            aroll_assemble, "probe_dur", side_effect=fake_probe
        ), patch.object(
            aroll_assemble, "generate_ass", side_effect=fake_generate
        ) as generate:
            aroll_assemble.run(str(project))
        return calls, generate

    def test_source_transcript_loader_uses_operational_defaults_when_doc_omits_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.mp4"
            source.write_bytes(b"source")
            doc = {"source_video": str(source), "language": "en"}
            with patch.object(aroll_assemble, "load_cached_source_transcript", return_value={}) as load:
                self.assertEqual(aroll_assemble._load_source_transcript(tmp, doc), {})
        self.assertEqual(load.call_args.kwargs["model_size"], "small")
        self.assertEqual(load.call_args.kwargs["compute_type"], "int8")

    def test_missing_caption_mode_preserves_legacy_off_behavior(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self.make_project(tmp)
            calls, generate = self.run_captured(project)
            final = str(project / "final.mp4")
        self.assertEqual(generate.call_count, 0)
        self.assertEqual(calls[-1][-2:], ["aac", final])
        self.assertEqual(calls[-1][calls[-1].index("-c:v") + 1], "copy")

    def test_explicit_off_skips_transcript_and_ass(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self.make_project(tmp, "off")
            calls, generate = self.run_captured(project)
            captions_exists = (project / "captions").exists()
        self.assertFalse(captions_exists)
        self.assertEqual(generate.call_count, 0)
        self.assertTrue(calls)

    def test_word_mode_reuses_cache_and_maps_original_audio_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self.make_project(tmp, "word")
            self.write_transcript(project)
            calls, generate = self.run_captured(project)
        mapped = generate.call_args.args[0]
        self.assertAlmostEqual(mapped["segments"][0]["words"][0]["start"], 0.1)
        self.assertFalse(hasattr(aroll_assemble, "build_source_transcript"))
        self.assertFalse(hasattr(aroll_assemble, "_transcribe_with_model"))
        video_call = next(call for call in calls if "-frames:v" in call)
        self.assertEqual(video_call.count("-map"), 1)
        self.assertIn("-an", video_call)
        self.assertEqual(sum("-vn" in call for call in calls), 1)
        final_call = calls[-1]
        self.assertIn("[a]", final_call)
        self.assertEqual(final_call.count("-map"), 2)

    def test_word_mode_uses_windows_safe_ass_filter_and_total_duration(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self.make_project(tmp, "word")
            self.write_transcript(project)
            calls, _generate = self.run_captured(project)
            expected = ffmpeg_filter_path(str(project / "captions" / "captions.ass"))
        final_call = calls[-1]
        filter_graph = final_call[final_call.index("-filter_complex") + 1]
        self.assertIn(expected, filter_graph)
        self.assertEqual(final_call[final_call.index("-t") + 1], "1.00")

    def test_ffconcat_line_normalizes_windows_paths_and_escapes_apostrophes(self):
        self.assertEqual(
            aroll_assemble.ffconcat_file_line(r"C:\caption project's\muxed 1.mp4"),
            "file 'C:/caption project'\\''s/muxed 1.mp4'\n",
        )

    def test_word_mode_passes_caption_style_and_defaults_position(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self.make_project(tmp, "word", caption_style="paper")
            self.write_transcript(project)
            _calls, generate = self.run_captured(project)
        self.assertEqual(generate.call_args.kwargs, {
            "caption_style": "paper",
            "caption_position": 10,
            "video_width": 1080,
            "video_height": 1920,
        })

    def test_word_mode_records_only_successfully_muxed_beat_spans(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self.make_project(tmp, "word")
            doc_path = project / "beats.json"
            doc = json.loads(doc_path.read_text(encoding="utf-8"))
            doc["beats"].append({
                "id": 2, "start": 3.0, "end": 4.0, "dur": 1.0,
                "narration": "second", "clip_path": doc["beats"][0]["clip_path"],
            })
            doc_path.write_text(json.dumps(doc), encoding="utf-8")
            self.write_transcript(project, [
                {"word": "discarded", "start": 1.1, "end": 1.5},
                {"word": "kept", "start": 3.1, "end": 3.5},
            ])
            _calls, generate = self.run_captured(project, durations=[0.0, 1.0, 1.0])
        mapped = generate.call_args.args[0]
        self.assertEqual(len(mapped["segments"]), 1)
        self.assertEqual(mapped["segments"][0]["beat_id"], 2)
        self.assertAlmostEqual(mapped["segments"][0]["words"][0]["start"], 0.1)

    def test_encoded_duration_is_used_for_mux_spans_and_final_total(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self.make_project(tmp, "word")
            doc_path = project / "beats.json"
            doc = json.loads(doc_path.read_text(encoding="utf-8"))
            doc["beats"].append({
                "id": 2, "start": 3.0, "end": 4.0, "dur": 1.0,
                "narration": "second", "clip_path": doc["beats"][0]["clip_path"],
            })
            doc_path.write_text(json.dumps(doc), encoding="utf-8")
            self.write_transcript(project, [
                {"word": "first", "start": 1.1, "end": 1.5},
                {"word": "second", "start": 3.1, "end": 3.5},
            ])
            calls, generate = self.run_captured(project, durations=[1.004] * 4)
        extraction_calls = [call for call in calls if "-vn" in call]
        video_calls = [call for call in calls if "-frames:v" in call]
        self.assertEqual([call[call.index("-t") + 1] for call in extraction_calls], ["1.00", "1.00"])
        self.assertEqual([call[call.index("-frames:v") + 1] for call in video_calls], ["24", "24"])
        self.assertEqual(
            [segment["start"] for segment in generate.call_args.args[0]["segments"]], [0.0, 1.0]
        )
        self.assertEqual(calls[-1][calls[-1].index("-t") + 1], "2.00")

    def test_requested_beat_cut_bounds_audio_mux_spans_and_final_total(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self.make_project(tmp, "word")
            doc_path = project / "beats.json"
            doc = json.loads(doc_path.read_text(encoding="utf-8"))
            doc["beats"].append({
                "id": 2, "start": 3.0, "end": 4.0, "dur": 1.0,
                "narration": "second", "clip_path": doc["beats"][0]["clip_path"],
            })
            doc_path.write_text(json.dumps(doc), encoding="utf-8")
            self.write_transcript(project, [
                {"word": "first", "start": 1.1, "end": 1.5},
                {"word": "second", "start": 3.1, "end": 3.5},
            ])
            calls, generate = self.run_captured(project, durations=[2.0] * 4)

        extraction_calls = [call for call in calls if "-vn" in call]
        self.assertEqual([call[call.index("-t") + 1] for call in extraction_calls], ["1.00", "1.00"])
        self.assertTrue(all(call[-1].endswith(".wav") for call in extraction_calls))
        video_calls = [call for call in calls if "-frames:v" in call]
        self.assertEqual([call[call.index("-frames:v") + 1] for call in video_calls], ["24", "24"])
        self.assertEqual(
            [segment["start"] for segment in generate.call_args.args[0]["segments"]], [0.0, 1.0]
        )
        self.assertEqual(calls[-1][calls[-1].index("-t") + 1], "2.00")

    def test_off_mode_final_is_bounded_by_requested_encoded_cut(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self.make_project(tmp, "off")
            calls, _generate = self.run_captured(project, durations=[2.0, 2.0])
        final_call = calls[-1]
        self.assertIn("-t", final_call)
        self.assertEqual(final_call[final_call.index("-t") + 1], "1.00")

    def test_video_segments_use_cumulative_audio_frame_allocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self.make_project(tmp, "off")
            doc_path = project / "beats.json"
            doc = json.loads(doc_path.read_text(encoding="utf-8"))
            clip = doc["beats"][0]["clip_path"]
            doc["beats"] = [
                {"id": index, "start": index * 3.0, "end": index * 3.0 + 2.03,
                 "dur": 2.03, "clip_path": clip}
                for index in range(8)
            ]
            doc_path.write_text(json.dumps(doc), encoding="utf-8")
            calls, _generate = self.run_captured(project, durations=[2.03] * 16)

        video_calls = [call for call in calls if "-frames:v" in call]
        self.assertEqual(
            [int(call[call.index("-frames:v") + 1]) for call in video_calls],
            [49, 48, 49, 49, 49, 48, 49, 49],
        )
        self.assertTrue(all("-an" in call for call in video_calls))
        self.assertTrue(any(call[-1].endswith("aroll_audio.wav") for call in calls))

    def test_pcm_concat_trims_each_beat_to_its_effective_timeline_duration(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self.make_project(tmp, "word")
            doc_path = project / "beats.json"
            doc = json.loads(doc_path.read_text(encoding="utf-8"))
            doc["beats"].append({
                "id": 2, "start": 3.0, "end": 5.0, "dur": 2.0,
                "clip_path": doc["beats"][0]["clip_path"],
            })
            doc["beats"][0].update({"start": 1.0, "end": 3.0, "dur": 2.0})
            doc_path.write_text(json.dumps(doc), encoding="utf-8")
            self.write_transcript(project, [
                {"word": "first", "start": 1.1, "end": 1.5},
                {"word": "second", "start": 3.1, "end": 3.5},
            ])
            calls, generate = self.run_captured(project, durations=[1.2, 2.0, 1.2, 2.0])

        concat_calls = [call for call in calls if call[-1].endswith("aroll_audio.wav")]
        self.assertEqual(len(concat_calls), 1)
        concat_call = concat_calls[0]
        self.assertIn("-filter_complex", concat_call)
        graph = concat_call[concat_call.index("-filter_complex") + 1]
        self.assertEqual(graph.count("atrim=duration=1.20"), 2)
        self.assertIn("concat=n=2:v=0:a=1", graph)
        self.assertEqual(
            [segment["start"] for segment in generate.call_args.args[0]["segments"]], [0.0, 1.2]
        )

    def test_centisecond_requested_cut_is_not_shortened_by_binary_float_rounding(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self.make_project(tmp, "off")
            doc_path = project / "beats.json"
            doc = json.loads(doc_path.read_text(encoding="utf-8"))
            doc["beats"][0].update(end=2.29, dur=1.29)
            doc_path.write_text(json.dumps(doc), encoding="utf-8")
            calls, _generate = self.run_captured(project, durations=[2.0, 2.0])
        extraction_call = next(call for call in calls if "-vn" in call)
        self.assertEqual(extraction_call[extraction_call.index("-t") + 1], "1.29")
        video_call = next(call for call in calls if "-frames:v" in call)
        self.assertEqual(video_call[video_call.index("-frames:v") + 1], "31")
        self.assertEqual(calls[-1][calls[-1].index("-t") + 1], "1.29")

    def test_source_cut_uses_exact_start_and_does_not_cross_beat_end(self):
        self.assertEqual(
            aroll_assemble._source_cut(1.3900000000000001, 2.68, 1.29), ("1.39", 1.29)
        )
        self.assertEqual(aroll_assemble._source_cut(1.006, 2.005, 1.0), ("1.01", 1.0))
        with tempfile.TemporaryDirectory() as tmp:
            project = self.make_project(tmp, "off")
            doc_path = project / "beats.json"
            doc = json.loads(doc_path.read_text(encoding="utf-8"))
            doc["beats"][0].update(start=1.006, end=2.006, dur=1.0)
            doc_path.write_text(json.dumps(doc), encoding="utf-8")
            calls, _generate = self.run_captured(project, durations=[2.0, 2.0])
        extraction_call = next(call for call in calls if "-vn" in call)
        self.assertEqual(extraction_call[extraction_call.index("-ss") + 1], "1.01")
        self.assertEqual(extraction_call[extraction_call.index("-t") + 1], "1.00")
        video_call = next(call for call in calls if "-frames:v" in call)
        self.assertEqual(video_call[video_call.index("-frames:v") + 1], "24")

    def test_caption_remap_uses_the_rounded_source_cut_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self.make_project(tmp, "word")
            self.write_transcript(project, words=[
                {"word": "before-cut", "start": 1.006, "end": 1.009},
            ])
            doc_path = project / "beats.json"
            doc = json.loads(doc_path.read_text(encoding="utf-8"))
            doc["beats"][0].update({"start": 1.006, "end": 2.005, "dur": 1.0})
            doc_path.write_text(json.dumps(doc), encoding="utf-8")
            _calls, generate = self.run_captured(project, durations=[2.0, 2.0])

        self.assertEqual(generate.call_args.args[0]["segments"][0]["words"], [])

    def test_exact_centisecond_source_bounds_keep_a_one_second_cut(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self.make_project(tmp, "off")
            doc_path = project / "beats.json"
            doc = json.loads(doc_path.read_text(encoding="utf-8"))
            doc["beats"][0].update(start=1.01, end=2.01, dur=1.0)
            doc_path.write_text(json.dumps(doc), encoding="utf-8")
            calls, _generate = self.run_captured(project, durations=[2.0, 2.0])
        extraction_call = next(call for call in calls if "-vn" in call)
        self.assertEqual(extraction_call[extraction_call.index("-ss") + 1], "1.01")
        self.assertEqual(extraction_call[extraction_call.index("-t") + 1], "1.00")
        video_call = next(call for call in calls if "-frames:v" in call)
        self.assertEqual(video_call[video_call.index("-frames:v") + 1], "24")

    def test_sub_frame_source_cut_is_skipped_before_ffmpeg(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self.make_project(tmp, "off")
            doc_path = project / "beats.json"
            doc = json.loads(doc_path.read_text(encoding="utf-8"))
            doc["beats"][0].update(start=0.10, end=0.11, dur=0.01)
            doc_path.write_text(json.dumps(doc), encoding="utf-8")
            with patch.object(aroll_assemble, "ff") as ff:
                with self.assertRaisesRegex(SystemExit, "minimum render duration"):
                    aroll_assemble.run(str(project))
        ff.assert_not_called()

    def test_just_below_minimum_render_duration_is_skipped_before_ffmpeg(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self.make_project(tmp, "off")
            doc_path = project / "beats.json"
            doc = json.loads(doc_path.read_text(encoding="utf-8"))
            doc["beats"][0].update(start=0.10, end=1.09, dur=0.99)
            doc_path.write_text(json.dumps(doc), encoding="utf-8")
            with patch.object(aroll_assemble, "ff") as ff:
                with self.assertRaisesRegex(SystemExit, "minimum render duration"):
                    aroll_assemble.run(str(project))
        ff.assert_not_called()

    def test_short_generated_clip_is_skipped_before_audio_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self.make_project(tmp, "off")
            with patch.object(aroll_assemble, "probe_dur", return_value=0.5), patch.object(
                aroll_assemble, "ff"
            ) as ff:
                with self.assertRaisesRegex(SystemExit, "minimum render duration"):
                    aroll_assemble.run(str(project))
        ff.assert_not_called()

    def test_short_extracted_audio_is_skipped_before_muxing(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self.make_project(tmp, "off")
            calls = []
            with patch.object(aroll_assemble, "probe_dur", side_effect=[2.0, 0.5]), patch.object(
                aroll_assemble, "ff", side_effect=calls.append
            ):
                with self.assertRaisesRegex(SystemExit, "minimum render duration"):
                    aroll_assemble.run(str(project))
        self.assertEqual(len(calls), 1)
        self.assertIn("-vn", calls[0])
        self.assertNotIn("1:a:0", calls[0])

    def test_exact_centisecond_source_bounds_are_not_lost_before_renderability_check(self):
        self.assertEqual(aroll_assemble._source_cut(0.10, 0.15, 0.05), ("0.10", 0.05))
        with tempfile.TemporaryDirectory() as tmp:
            project = self.make_project(tmp, "off")
            doc_path = project / "beats.json"
            doc = json.loads(doc_path.read_text(encoding="utf-8"))
            doc["beats"][0].update(start=0.10, end=0.15, dur=0.05)
            doc_path.write_text(json.dumps(doc), encoding="utf-8")
            with patch.object(aroll_assemble, "ff") as ff:
                with self.assertRaisesRegex(SystemExit, "minimum render duration"):
                    aroll_assemble.run(str(project))
        ff.assert_not_called()

    def test_unprobeable_clip_or_audio_skips_the_beat(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self.make_project(tmp, "off")
            with self.assertRaisesRegex(SystemExit, "No beats had a generated clip"):
                self.run_captured(project, durations=[2.0, 0.0])

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg and ffprobe required")
    def test_real_ffmpeg_nonround_and_multibeat_timelines_stay_within_media_quantum(self):
        cases = ((1.01,), (1.29,), (2.03,), (1.01, 1.29, 2.03), (2.03,) * 8)
        quantum = 1 / aroll_assemble.FPS + 1024 / 48000
        for durations in cases:
            for caption_mode in ("off", "word"):
                with self.subTest(caption_mode=caption_mode, durations=durations), tempfile.TemporaryDirectory() as tmp:
                    project = Path(tmp) / caption_mode
                    project.mkdir()
                    source = project / "source.mp4"
                    subprocess.run([
                        "ffmpeg", "-y", "-loglevel", "error",
                        "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=24",
                        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
                        "-t", "20", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(source),
                    ], check=True)
                    start = 0.1
                    beats = []
                    segments = []
                    for index, duration in enumerate(durations, start=1):
                        end = start + duration
                        beats.append({
                            "id": index, "start": start, "end": end, "dur": duration,
                            "clip_path": str(source),
                        })
                        word_end = min(end, start + 0.3)
                        segments.append({
                            "start": start, "end": word_end, "text": f"beat {index}",
                            "words": [{"word": f"beat{index}", "start": start, "end": word_end}],
                        })
                        start = end + 0.1
                    doc = {
                        "mode": "aroll",
                        "source_video": str(source),
                        "language": "en",
                        "aspect": "tiny",
                        "caption_mode": caption_mode,
                        "beats": beats,
                    }
                    (project / "beats.json").write_text(json.dumps(doc), encoding="utf-8")
                    if caption_mode == "word":
                        caption_dir = project / "captions"
                        caption_dir.mkdir()
                        (caption_dir / "transcript.json").write_text(json.dumps({
                            "schema_version": 1,
                            "source_fingerprint": hashlib.sha256(source.read_bytes()).hexdigest(),
                            "model": "small",
                            "requested_language": "en",
                            "compute_type": "int8",
                            "language": "en",
                            "segments": segments,
                        }), encoding="utf-8")
                    with patch.dict(aroll_assemble.RES, {"tiny": (64, 64)}):
                        aroll_assemble.run(str(project))
                    final = project / "final.mp4"
                    expected = sum(durations)
                    streams = json.loads(subprocess.run([
                        "ffprobe", "-v", "error",
                        "-show_entries", "stream=codec_type,start_time,duration",
                        "-of", "json", str(final),
                    ], capture_output=True, text=True, check=True).stdout)["streams"]
                    video_streams = [stream for stream in streams if stream["codec_type"] == "video"]
                    audio_streams = [stream for stream in streams if stream["codec_type"] == "audio"]
                    self.assertEqual(len(video_streams), 1)
                    self.assertEqual(len(audio_streams), 1)
                    self.assertLessEqual(abs(aroll_assemble.probe_dur(final) - expected), quantum)
                    for stream in [*video_streams, *audio_streams]:
                        self.assertLessEqual(abs(float(stream["start_time"])), 0.001)
                        self.assertLessEqual(abs(float(stream["duration"]) - expected), quantum)
                    volume = subprocess.run([
                        "ffmpeg", "-v", "info", "-i", str(final), "-map", "0:a:0",
                        "-af", "volumedetect", "-f", "null", "-",
                    ], capture_output=True, text=True, check=True).stderr
                    self.assertIn("max_volume:", volume)
                    self.assertNotIn("max_volume: -inf", volume)
                    if caption_mode == "word":
                        ass = (project / "captions" / "captions.ass").read_text(encoding="utf-8")
                        for index in range(1, len(durations) + 1):
                            self.assertIn(f"beat{index}", ass)
                        if len(durations) == 8:
                            later_event = next(line for line in ass.splitlines() if "beat8" in line)
                            self.assertIn(",0:00:14.21,", later_event)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg and ffprobe required")
    def test_real_ffmpeg_short_clips_trim_pcm_audio_to_caption_timeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "word"
            project.mkdir()
            source = project / "source.mp4"
            clip = project / "clip.mp4"
            subprocess.run([
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=24",
                "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
                "-t", "5", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(source),
            ], check=True)
            subprocess.run([
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=24",
                "-t", "1.2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(clip),
            ], check=True)
            doc = {
                "mode": "aroll", "source_video": str(source), "language": "en",
                "aspect": "tiny", "caption_mode": "word",
                "beats": [
                    {"id": 1, "start": 0.0, "end": 2.0, "dur": 2.0, "clip_path": str(clip)},
                    {"id": 2, "start": 2.1, "end": 4.1, "dur": 2.0, "clip_path": str(clip)},
                ],
            }
            (project / "beats.json").write_text(json.dumps(doc), encoding="utf-8")
            captions = project / "captions"
            captions.mkdir()
            (captions / "transcript.json").write_text(json.dumps({
                "schema_version": 1,
                "source_fingerprint": hashlib.sha256(source.read_bytes()).hexdigest(),
                "model": "small", "requested_language": "en", "compute_type": "int8", "language": "en",
                "segments": [
                    {"start": 0.1, "end": 0.4, "text": "first",
                     "words": [{"word": "first", "start": 0.1, "end": 0.4}]},
                    {"start": 2.1, "end": 2.4, "text": "second",
                     "words": [{"word": "second", "start": 2.1, "end": 2.4}]},
                ],
            }), encoding="utf-8")
            with patch.dict(aroll_assemble.RES, {"tiny": (64, 64)}):
                aroll_assemble.run(str(project))
            final = project / "final.mp4"
            streams = json.loads(subprocess.run([
                "ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration",
                "-of", "json", str(final),
            ], capture_output=True, text=True, check=True).stdout)["streams"]
            audio = next(stream for stream in streams if stream["codec_type"] == "audio")
            video = next(stream for stream in streams if stream["codec_type"] == "video")
            self.assertLessEqual(abs(float(audio["duration"]) - 2.4), 1024 / 48000)
            self.assertLessEqual(abs(float(video["duration"]) - 2.4), 1 / aroll_assemble.FPS)
            second_event = next(
                line for line in (captions / "captions.ass").read_text(encoding="utf-8").splitlines()
                if "second" in line
            )
            self.assertIn(",0:00:01.20,", second_event)

    def test_unknown_caption_mode_fails(self):
        with self.assertRaisesRegex(ValueError, "word, off"):
            aroll_assemble._caption_mode({"caption_mode": "static"})
        with self.assertRaisesRegex(ValueError, "word, off"):
            aroll_assemble._caption_mode({"caption_mode": ["word"]})

    def test_word_mode_missing_or_invalid_transcript_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self.make_project(tmp, "word")
            with self.assertRaisesRegex(RuntimeError, "rerun asr_beats.py"):
                self.run_captured(project)
            (project / "captions").mkdir()
            (project / "captions" / "transcript.json").write_text("{broken", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "rerun asr_beats.py"):
                self.run_captured(project)

    def test_word_mode_rejects_stale_or_noncanonical_source_cache_without_asr(self):
        cases = {
            "source": lambda project, doc: Path(doc["source_video"]).write_bytes(b"changed"),
            "model": lambda project, doc: self.write_transcript(project, model="small"),
            "language": lambda project, doc: self.write_transcript(project, requested_language="el"),
            "compute": lambda project, doc: self.write_transcript(project, compute_type="default"),
            "schema": lambda project, doc: self.write_transcript(project, schema_version=2),
            "words": lambda project, doc: self.write_transcript(project, segments=[{"words": [{"text": "bad"}]}]),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                project = self.make_project(tmp, "word")
                self.write_transcript(project)
                doc = json.loads((project / "beats.json").read_text(encoding="utf-8"))
                mutate(project, doc)
                with self.assertRaisesRegex(RuntimeError, "rerun asr_beats.py"):
                    self.run_captured(project)
        self.assertFalse(hasattr(aroll_assemble, "build_source_transcript"))
        self.assertFalse(hasattr(aroll_assemble, "_transcribe_with_model"))

    def test_word_mode_requires_rendered_ass_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self.make_project(tmp, "word")
            self.write_transcript(project)
            with self.assertRaisesRegex(RuntimeError, "no caption events"):
                self.run_captured(project, generate_result=False)

    def test_missing_generated_clips_keep_existing_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self.make_project(tmp, "off")
            doc_path = project / "beats.json"
            doc = json.loads(doc_path.read_text(encoding="utf-8"))
            del doc["beats"][0]["clip_path"]
            doc_path.write_text(json.dumps(doc), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "No beats had a generated clip"):
                aroll_assemble.run(str(project))


if __name__ == "__main__":
    unittest.main()
