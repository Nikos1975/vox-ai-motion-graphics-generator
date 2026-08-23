import sys
import tempfile
import unittest
import json
import math
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from captions.caption_styles import get_caption_style
from captions.subtitle_utils import escape_ass_text, ffmpeg_filter_path, format_ass_time
from captions.subtitles import generate_ass
from captions.transcription import build_timeline_transcript, merge_beat_transcripts


class CaptionStyleTests(unittest.TestCase):
    def test_legacy_aliases_are_supported(self):
        self.assertEqual(get_caption_style("white"), get_caption_style("classic"))
        self.assertEqual(get_caption_style("paper"), get_caption_style("editorial"))

    def test_unknown_style_fails(self):
        with self.assertRaises(ValueError):
            get_caption_style("unknown")


class SubtitleUtilsTests(unittest.TestCase):
    def test_ass_time(self):
        self.assertEqual(format_ass_time(65.432), "0:01:05.43")

    def test_ass_text_escape(self):
        escaped = escape_ass_text(r"O'Brien {hello} \ path")
        self.assertEqual(escaped, r"O'Brien \{hello\} \\ path")

    def test_windows_filter_path(self):
        self.assertEqual(ffmpeg_filter_path(r"C:\tmp\captions.ass"), r"C\:/tmp/captions.ass")

    def test_filter_path_with_apostrophe(self):
        self.assertEqual(
            ffmpeg_filter_path(r"C:\path's\captions.ass"),
            r"C\:/path'\\\''s/captions.ass",
        )


class SubtitleGenerationTests(unittest.TestCase):
    def test_generate_word_timed_ass(self):
        transcript = {
            "language": "en",
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.2,
                    "words": [
                        {"word": "Crete", "start": 0.0, "end": 0.4},
                        {"word": "has", "start": 0.42, "end": 0.65},
                        {"word": "history", "start": 0.7, "end": 1.2},
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "captions.ass"
            self.assertTrue(generate_ass(transcript, out, caption_style="paper"))
            text = out.read_text(encoding="utf-8")
        self.assertIn("[Events]", text)
        self.assertEqual(text.count("Dialogue: 0,"), 3)
        self.assertIn("Crete", text)
        self.assertIn("history", text)

    def test_empty_transcript_returns_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "captions.ass"
            self.assertFalse(generate_ass({"segments": []}, out))
            self.assertFalse(out.exists())

    def test_non_finite_word_times_are_ignored(self):
        transcript = {
            "segments": [{
                "words": [
                    {"word": "nan", "start": float("nan"), "end": 0.5},
                    {"word": "infinite", "start": 0.5, "end": float("inf")},
                    {"word": "valid", "start": 1.0, "end": 1.5},
                ]
            }]
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "captions.ass"
            try:
                created = generate_ass(transcript, out)
            except ValueError as exc:
                self.fail(f"non-finite word times should be ignored: {exc}")
            self.assertTrue(created)
            text = out.read_text(encoding="utf-8")
        self.assertEqual(text.count("Dialogue: 0,"), 1)
        self.assertIn("valid", text)

    def test_ass_uses_requested_landscape_and_portrait_dimensions(self):
        transcript = {
            "segments": [{"words": [{"word": "safe", "start": 0.0, "end": 0.5}]}]
        }
        with tempfile.TemporaryDirectory() as tmp:
            for width, height in ((1920, 1080), (1080, 1920)):
                out = Path(tmp) / f"captions-{width}x{height}.ass"
                self.assertTrue(
                    generate_ass(transcript, out, video_width=width, video_height=height)
                )
                text = out.read_text(encoding="utf-8")
                self.assertIn(f"PlayResX: {width}", text)
                self.assertIn(f"PlayResY: {height}", text)

    def test_caption_groups_do_not_cross_beat_boundaries(self):
        transcript = {
            "segments": [
                {
                    "beat_id": 1,
                    "words": [{"word": "first", "start": 0.0, "end": 0.5}],
                },
                {
                    "beat_id": 2,
                    "words": [{"word": "second", "start": 3.0, "end": 3.5}],
                },
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "captions.ass"
            self.assertTrue(generate_ass(transcript, out))
            events = [
                line for line in out.read_text(encoding="utf-8").splitlines()
                if line.startswith("Dialogue:")
            ]
        self.assertNotIn("second", events[0])
        self.assertNotIn("first", events[1])
        self.assertIn("0:00:00.50", events[0])


class TranscriptMergeTests(unittest.TestCase):
    def test_offsets_each_beat_into_final_timeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            a = tmp / "a.mp3"
            b = tmp / "b.mp3"
            a.write_bytes(b"a")
            b.write_bytes(b"b")
            spans = [
                {"start": 0.0, "dur": 2.0, "beat": {"id": 1, "audio_path": str(a)}},
                {"start": 3.0, "dur": 2.0, "beat": {"id": 2, "audio_path": str(b)}},
            ]

            def fake(_path):
                return {
                    "language": "en",
                    "segments": [{
                        "start": 0.1,
                        "end": 0.8,
                        "words": [
                            {"word": "one", "start": 0.1, "end": 0.3},
                            {"word": "two", "start": 0.35, "end": 0.8},
                        ],
                    }],
                }

            merged = merge_beat_transcripts(spans, fake, language="en")
        self.assertEqual(merged["segments"][0]["words"][0]["start"], 0.1)
        self.assertEqual(merged["segments"][1]["words"][0]["start"], 3.1)
        self.assertEqual(merged["segments"][1]["beat_id"], 2)

    def test_clips_word_times_to_beat_span(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "a.mp3"
            audio.write_bytes(b"a")
            spans = [{"start": 5.0, "dur": 1.0, "beat": {"id": 1, "audio_path": str(audio)}}]

            def fake(_path):
                return {
                    "segments": [{
                        "words": [
                            {"word": "keep", "start": 0.2, "end": 0.6},
                            {"word": "clip", "start": 0.7, "end": 1.4},
                            {"word": "drop", "start": 1.1, "end": 1.3},
                        ]
                    }]
                }

            merged = merge_beat_transcripts(spans, fake, language="en")
        words = merged["segments"][0]["words"]
        self.assertEqual([w["word"] for w in words], ["keep", "clip"])
        self.assertEqual(words[-1]["end"], 6.0)

    def test_clips_word_start_to_beat_span(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "a.mp3"
            audio.write_bytes(b"a")
            spans = [{"start": 5.0, "dur": 1.0, "beat": {"id": 1, "audio_path": str(audio)}}]

            def fake(_path):
                return {
                    "segments": [{
                        "words": [
                            {"word": "clip", "start": -0.2, "end": 0.3},
                        ]
                    }]
                }

            merged = merge_beat_transcripts(spans, fake, language="en")
        self.assertEqual(merged["segments"][0]["words"][0]["start"], 5.0)

    def test_missing_narration_audio_is_skipped_without_transcription(self):
        spans = [{
            "start": 0.0,
            "dur": 1.0,
            "beat": {"id": 1, "audio_path": "definitely-missing.mp3"},
        }]
        transcribe_calls = []

        def fake(path):
            transcribe_calls.append(path)
            return self.fail("missing audio should not be transcribed")

        merged = merge_beat_transcripts(spans, fake, language="en")
        self.assertEqual(merged["segments"], [])
        self.assertEqual(transcribe_calls, [])


class TranscriptCacheTests(unittest.TestCase):
    @staticmethod
    def _spans(audio: Path):
        return [{"start": 0.0, "dur": 2.0, "beat": {"id": 1, "audio_path": str(audio)}}]

    @staticmethod
    def _transcript():
        return {
            "language": "en",
            "segments": [{
                "start": 0.1,
                "end": 0.4,
                "text": "hello",
                "words": [{"word": "hello", "start": 0.1, "end": 0.4}],
            }],
        }

    def test_valid_json_with_missing_segments_is_retranscribed(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            audio = project / "a.mp3"
            audio.write_bytes(b"audio")
            spans = self._spans(audio)
            caption_dir = project / "captions"
            caption_dir.mkdir()

            with patch("captions.transcription._load_model", return_value=object()), patch(
                "captions.transcription._transcribe_with_model", return_value=self._transcript()
            ) as transcribe:
                first = build_timeline_transcript(project, spans, compute_type="default")
                corrupt = {
                    key: first[key]
                    for key in ("schema_version", "source_fingerprint", "model", "requested_language")
                }
                (caption_dir / "transcript.json").write_text(
                    json.dumps(corrupt), encoding="utf-8"
                )
                build_timeline_transcript(project, spans, compute_type="default")

            self.assertEqual(transcribe.call_count, 2)

    def test_valid_json_with_malformed_word_timing_is_retranscribed(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            audio = project / "a.mp3"
            audio.write_bytes(b"audio")
            spans = self._spans(audio)

            with patch("captions.transcription._load_model", return_value=object()), patch(
                "captions.transcription._transcribe_with_model", return_value=self._transcript()
            ) as transcribe:
                first = build_timeline_transcript(project, spans)
                first["segments"][0]["words"][0]["start"] = "not-a-number"
                (project / "captions" / "transcript.json").write_text(
                    json.dumps(first), encoding="utf-8"
                )
                build_timeline_transcript(project, spans)

            self.assertEqual(transcribe.call_count, 2)

    def test_malformed_canonical_segment_fields_are_recovered_for_broll_and_croll(self):
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
            "reversed segment times": lambda transcript: transcript["segments"][0].update(start=0.4, end=0.1),
            "non-string segment text": lambda transcript: transcript["segments"][0].__setitem__("text", 7),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                project = Path(tmp)
                audio = project / "a.mp3"
                audio.write_bytes(b"audio")
                spans = self._spans(audio)
                with patch("captions.transcription._load_model", return_value=object()), patch(
                    "captions.transcription._transcribe_with_model", return_value=self._transcript()
                ) as transcribe:
                    first = build_timeline_transcript(project, spans)
                    mutate(first)
                    (project / "captions" / "transcript.json").write_text(
                        json.dumps(first), encoding="utf-8"
                    )
                    build_timeline_transcript(project, spans)
                self.assertEqual(transcribe.call_count, 2)

    def test_compute_type_change_invalidates_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            audio = project / "a.mp3"
            audio.write_bytes(b"audio")
            spans = self._spans(audio)

            with patch("captions.transcription._load_model", return_value=object()), patch(
                "captions.transcription._transcribe_with_model", return_value=self._transcript()
            ) as transcribe:
                build_timeline_transcript(project, spans, compute_type="default")
                build_timeline_transcript(project, spans, compute_type="int8")

            self.assertEqual(transcribe.call_count, 2)

    def test_unchanged_inputs_reuse_cache_without_rewriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            audio = project / "a.mp3"
            audio.write_bytes(b"audio")
            spans = self._spans(audio)

            with patch("captions.transcription._load_model", return_value=object()), patch(
                "captions.transcription._transcribe_with_model", return_value=self._transcript()
            ) as transcribe:
                build_timeline_transcript(project, spans)
                transcript_path = project / "captions" / "transcript.json"
                original = transcript_path.read_bytes()
                build_timeline_transcript(project, spans)

            self.assertEqual(transcribe.call_count, 1)
            self.assertEqual(transcript_path.read_bytes(), original)

    def test_audio_digest_change_invalidates_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            audio = project / "a.mp3"
            audio.write_bytes(b"first")
            spans = self._spans(audio)

            with patch("captions.transcription._load_model", return_value=object()), patch(
                "captions.transcription._transcribe_with_model", return_value=self._transcript()
            ) as transcribe:
                build_timeline_transcript(project, spans)
                audio.write_bytes(b"second")
                build_timeline_transcript(project, spans)

            self.assertEqual(transcribe.call_count, 2)

    def test_beat_timeline_change_invalidates_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            audio = project / "a.mp3"
            audio.write_bytes(b"audio")
            spans = self._spans(audio)

            with patch("captions.transcription._load_model", return_value=object()), patch(
                "captions.transcription._transcribe_with_model", return_value=self._transcript()
            ) as transcribe:
                build_timeline_transcript(project, spans)
                changed = self._spans(audio)
                changed[0]["start"] = 0.25
                build_timeline_transcript(project, changed)

            self.assertEqual(transcribe.call_count, 2)

    def test_invalid_json_cache_is_recovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            audio = project / "a.mp3"
            audio.write_bytes(b"audio")
            spans = self._spans(audio)
            caption_dir = project / "captions"
            caption_dir.mkdir()
            (caption_dir / "transcript.json").write_text("{broken", encoding="utf-8")

            with patch("captions.transcription._load_model", return_value=object()), patch(
                "captions.transcription._transcribe_with_model", return_value=self._transcript()
            ) as transcribe:
                build_timeline_transcript(project, spans)

            self.assertEqual(transcribe.call_count, 1)
            cached = json.loads((caption_dir / "transcript.json").read_text(encoding="utf-8"))
            self.assertIn("segments", cached)

    def test_auto_device_retries_cpu_when_cuda_runtime_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            audio = project / "a.mp3"
            audio.write_bytes(b"audio")
            spans = self._spans(audio)
            auto_model = object()
            cpu_model = object()

            def transcribe(model, _path, _language):
                if model is auto_model:
                    raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
                return self._transcript()

            with patch(
                "captions.transcription._load_model", side_effect=[auto_model, cpu_model]
            ) as load_model, patch(
                "captions.transcription._transcribe_with_model", side_effect=transcribe
            ):
                try:
                    result = build_timeline_transcript(project, spans, device="auto")
                except RuntimeError as exc:
                    self.fail(f"auto device should retry CPU for a missing CUDA runtime: {exc}")

            self.assertEqual(len(result["segments"]), 1)
            self.assertEqual(load_model.call_args_list[0].args[1], "auto")
            self.assertEqual(load_model.call_args_list[1].args[1], "cpu")

    def test_auto_device_retries_cpu_when_model_construction_needs_cuda_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            audio = project / "a.mp3"
            audio.write_bytes(b"audio")
            spans = self._spans(audio)
            cpu_model = object()
            error = RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")

            with patch(
                "captions.transcription._load_model", side_effect=[error, cpu_model]
            ) as load_model, patch(
                "captions.transcription._transcribe_with_model", return_value=self._transcript()
            ):
                try:
                    result = build_timeline_transcript(project, spans, device="auto")
                except RuntimeError as exc:
                    self.fail(f"auto device should retry CPU after model construction fails: {exc}")

            self.assertEqual(len(result["segments"]), 1)
            self.assertEqual(load_model.call_args_list[1].args[1], "cpu")


class CaptionModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import assemble

        cls.assemble = assemble

    def test_word_static_and_off_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            ass = Path(tmp) / "captions.ass"
            with patch.object(self.assemble, "prepare_word_captions", return_value=ass):
                self.assertEqual(
                    self.assemble._prepare_captions(tmp, [], {"caption_mode": "word"}, 1920, 1080),
                    ("word", ass),
                )
            self.assertEqual(
                self.assemble._prepare_captions(tmp, [], {"caption_mode": "static"}, 1920, 1080),
                ("static", None),
            )
            self.assertEqual(
                self.assemble._prepare_captions(tmp, [], {"caption_mode": "off"}, 1920, 1080),
                ("off", None),
            )

    def test_missing_dependency_falls_back_unless_required(self):
        error = self.assemble.CaptionDependencyError("missing")
        with patch.object(self.assemble, "prepare_word_captions", side_effect=error):
            self.assertEqual(
                self.assemble._prepare_captions(".", [], {"caption_required": False}, 1920, 1080),
                ("static", None),
            )
            with self.assertRaises(self.assemble.CaptionDependencyError):
                self.assemble._prepare_captions(
                    ".", [], {"caption_required": True}, 1920, 1080
                )


class AssemblyAudioTests(unittest.TestCase):
    def test_mixer_counts_only_existing_narration_inputs(self):
        import assemble

        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            existing_audio = project / "narr-2.wav"
            existing_audio.write_bytes(b"audio")
            bgm_dir = project / "audio"
            bgm_dir.mkdir()
            (bgm_dir / "bgm.mp3").write_bytes(b"music")
            doc = {
                "caption_mode": "off",
                "beats": [
                    {
                        "id": 1,
                        "dur": 1.0,
                        "clip_path": str(project / "clip-1.mp4"),
                        "audio_path": str(project / "missing.wav"),
                    },
                    {
                        "id": 2,
                        "dur": 1.0,
                        "clip_path": str(project / "clip-2.mp4"),
                        "audio_path": str(existing_audio),
                    },
                ],
            }
            (project / "beats.json").write_text(json.dumps(doc), encoding="utf-8")
            ff_calls = []

            with patch.object(assemble, "ff", side_effect=ff_calls.append), patch.object(
                assemble, "probe_dur", return_value=1.0
            ), patch.object(assemble.text_overlay, "render_watermark"):
                assemble.run(str(project))

            audio_mix_call = next(call for call in ff_calls if call[-1].endswith("audio_mixed.m4a"))
            filter_graph = audio_mix_call[audio_mix_call.index("-filter_complex") + 1]
            self.assertIn("[a1]amix=inputs=1:duration=longest", filter_graph)
            self.assertNotIn("[a2]", filter_graph)
            self.assertIn("amix=inputs=2:duration=longest", filter_graph)
            self.assertIn("[aout]apad[aout_padded]", filter_graph)
            self.assertIn("[aout_padded]", audio_mix_call)

    def test_timeline_uses_the_rounded_encoded_segment_durations(self):
        import assemble

        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            doc = {
                "caption_mode": "off",
                "beats": [{
                    "id": 1,
                    "shots": [
                        {"dur": 1.004, "clip_path": str(project / "clip-1.mp4")},
                        {"dur": 1.004, "clip_path": str(project / "clip-2.mp4")},
                    ],
                }],
            }
            (project / "beats.json").write_text(json.dumps(doc), encoding="utf-8")
            captured_spans = []

            def capture_captions(_project, spans, _doc, _width, _height):
                captured_spans.extend(spans)
                return "off", None

            with patch.object(assemble, "ff"), patch.object(
                assemble, "probe_dur", return_value=1.0
            ), patch.object(
                assemble, "_prepare_captions", side_effect=capture_captions
            ), patch.object(assemble.text_overlay, "render_watermark"):
                assemble.run(str(project))

            self.assertEqual(captured_spans[0]["dur"], 2.5)


if __name__ == "__main__":
    unittest.main()
