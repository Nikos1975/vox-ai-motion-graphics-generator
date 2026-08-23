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
        return build_source_transcript(
            project, source, language=kwargs.get("language", "en"),
            model_size=kwargs.get("model_size", "large-v3-turbo"),
            device=kwargs.get("device", "cpu"),
            compute_type=kwargs.get("compute_type", "int8"),
        )

    def test_unconfigured_source_uses_accurate_cpu_profile(self):
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
            self.assertEqual(load_model.call_args.args, ("large-v3-turbo", "cpu", "int8"))

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
            ({}, {"model_size": "small"}),
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
        self.assertEqual(doc["caption_mode"], "word")
        self.assertEqual(doc["caption_whisper_model"], "large-v3-turbo")
        self.assertEqual(doc["caption_whisper_device"], "cpu")
        self.assertEqual(doc["caption_whisper_compute_type"], "int8")

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


if __name__ == "__main__":
    unittest.main()
