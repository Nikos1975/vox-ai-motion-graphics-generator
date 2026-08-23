#!/usr/bin/env python3
"""Build an A-roll beat skeleton from one canonical local transcript."""
import argparse
import json
import math
import os
import subprocess
from pathlib import Path

from captions.transcription import (
    AROLL_DEFAULT_COMPUTE_TYPE,
    AROLL_DEFAULT_DEVICE,
    AROLL_DEFAULT_MODEL_SIZE,
    build_source_transcript,
)

MAX_BEAT_DUR = 9.5
MIN_BEAT_DUR = 2.0
PAUSE_GAP_S = 0.35
SENTENCE_END = (".", "!", "?")


def probe_dims(path):
    out = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path),
    ], capture_output=True, text=True).stdout.strip()
    if not out:
        return None
    w, h = (int(x) for x in out.split(",")[:2])
    return w, h


def probe_dur(path):
    out = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "csv=p=0", str(path),
    ], capture_output=True, text=True).stdout.strip()
    try:
        duration = float(out)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Could not determine source duration: {path}") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError(f"Could not determine a positive source duration: {path}")
    return duration


def nearest_named_aspect(w, h):
    named = {"16:9": 16 / 9, "9:16": 9 / 16, "1:1": 1.0, "4:3": 4 / 3, "3:4": 3 / 4}
    ratio = w / h
    return min(named, key=lambda k: abs(named[k] - ratio))


def words_from_transcript(transcript, source_duration):
    """Flatten canonical transcript words into the legacy segmenter input shape."""
    try:
        duration = float(source_duration)
    except (TypeError, ValueError):
        return []
    if not math.isfinite(duration) or duration <= 0 or not isinstance(transcript, dict):
        return []

    segments = transcript.get("segments")
    if not isinstance(segments, list):
        return []

    words = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        segment_words = segment.get("words")
        if not isinstance(segment_words, list):
            continue
        for item in segment_words:
            if not isinstance(item, dict):
                continue
            # Canonical transcripts use ``word``. ``text`` is accepted only to
            # retain this segmenter's legacy input boundary.
            text = item.get("word", item.get("text", ""))
            if not isinstance(text, str) or not (text := text.strip()):
                continue
            try:
                start = float(item["start"])
                end = float(item["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if (
                not math.isfinite(start)
                or not math.isfinite(end)
                or start < 0
                or end < 0
                or end <= start
                or start >= duration
                or end <= 0
            ):
                continue
            end = min(end, duration)
            if end <= start:
                continue
            words.append({"text": text, "start": start, "end": end})
    return sorted(words, key=lambda item: (item["start"], item["end"]))


def segment_words(words, max_dur=MAX_BEAT_DUR, min_dur=MIN_BEAT_DUR, pause_gap=PAUSE_GAP_S):
    beats, cur = [], []

    def flush():
        if not cur:
            return
        beats.append({
            "start": cur[0]["start"],
            "end": cur[-1]["end"],
            "text": " ".join(w["text"] for w in cur),
        })
        cur.clear()

    for i, word in enumerate(words):
        cur.append(word)
        dur = word["end"] - cur[0]["start"]
        is_last = i == len(words) - 1
        gap_next = (words[i + 1]["start"] - word["end"]) if not is_last else None
        sentence_end = word["text"].rstrip().endswith(SENTENCE_END)
        if is_last:
            flush()
        elif dur >= max_dur:
            flush()
        elif dur >= min_dur and (sentence_end or (gap_next is not None and gap_next >= pause_gap)):
            flush()

    merged = []
    for beat in beats:
        if merged and (beat["end"] - beat["start"]) < min_dur:
            merged[-1]["end"] = beat["end"]
            merged[-1]["text"] += " " + beat["text"]
        else:
            merged.append(beat)
    return merged


def run(
    project_dir,
    src_path,
    language="en",
    keyterms=None,
    max_beat_dur=MAX_BEAT_DUR,
    model_size=AROLL_DEFAULT_MODEL_SIZE,
    device=AROLL_DEFAULT_DEVICE,
    compute_type=AROLL_DEFAULT_COMPUTE_TYPE,
):
    del keyterms
    project = Path(project_dir)
    source = Path(src_path).resolve()
    project.mkdir(parents=True, exist_ok=True)

    transcript = build_source_transcript(
        project,
        source,
        language=language,
        model_size=model_size,
        device=device,
        compute_type=compute_type,
    )
    source_duration = probe_dur(source)
    words = words_from_transcript(transcript, source_duration)
    beats = segment_words(words, max_dur=max_beat_dur)
    if not beats:
        raise RuntimeError("A-roll transcription produced no valid timed words")

    dims = probe_dims(source)
    aspect = nearest_named_aspect(*dims) if dims else "9:16"
    doc = {
        "mode": "aroll",
        "project": project.resolve().name,
        "source_video": str(source),
        "language": language,
        "aspect": aspect,
        "theme": "american-retro",
        "video_model": "veo31-image-to-video",
        "caption_mode": "word",
        "caption_whisper_model": model_size,
        "caption_whisper_device": device,
        "caption_whisper_compute_type": compute_type,
        "beats": [
            {
                "id": i + 1,
                "start": beat["start"],
                "end": beat["end"],
                "dur": round(beat["end"] - beat["start"], 2),
                "narration": beat["text"],
                "title_en": f"BEAT {i + 1}",
                "content_beats": "",
            }
            for i, beat in enumerate(beats)
        ],
    }

    out_file = project / "beats.json"
    out_file.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"ASR beats skeleton written to {out_file}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="ASR bridge for A-roll mode")
    parser.add_argument("project_dir", help="Project directory")
    parser.add_argument("source", help="Source video or audio file")
    parser.add_argument("--language", default="en")
    parser.add_argument("--model", default=AROLL_DEFAULT_MODEL_SIZE, dest="model_size")
    parser.add_argument("--device", default=AROLL_DEFAULT_DEVICE)
    parser.add_argument("--compute-type", default=AROLL_DEFAULT_COMPUTE_TYPE)
    args = parser.parse_args(argv)
    run(
        os.path.abspath(args.project_dir),
        os.path.abspath(args.source),
        language=args.language,
        model_size=args.model_size,
        device=args.device,
        compute_type=args.compute_type,
    )


if __name__ == "__main__":
    main()
