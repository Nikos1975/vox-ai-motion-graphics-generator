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
            ({}, {"model_size": "distil-small.en"}),
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


if __name__ == "__main__":
    unittest.main()
