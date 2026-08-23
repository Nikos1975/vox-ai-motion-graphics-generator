import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from captions.caption_styles import get_caption_style
from captions.subtitle_utils import escape_ass_text, ffmpeg_filter_path, format_ass_time
from captions.subtitles import generate_ass
from captions.transcription import merge_beat_transcripts


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
        self.assertIn(r"\{", escape_ass_text("{hello}"))

    def test_windows_filter_path(self):
        self.assertEqual(ffmpeg_filter_path(r"C:\tmp\captions.ass"), r"C\:/tmp/captions.ass")


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


if __name__ == "__main__":
    unittest.main()
