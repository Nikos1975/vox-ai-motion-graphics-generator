#!/usr/bin/env python3
"""
A-roll assembly: muxes each beat's generated visual clip with the ORIGINAL beat segment's audio.
"""
import json
import math
import os
import subprocess
import sys
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from pathlib import Path

from captions.transcription import (
    AROLL_DEFAULT_COMPUTE_TYPE,
    AROLL_DEFAULT_MODEL_SIZE,
    load_cached_source_transcript,
)
from captions.subtitles import generate_ass
from captions.subtitle_utils import ffmpeg_filter_path

RES = {"16:9": (1920, 1080), "9:16": (1080, 1920), "1:1": (1080, 1080),
       "4:3": (1440, 1080), "3:4": (1080, 1440)}


def ff(args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def probe_dur(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", path], capture_output=True, text=True).stdout
    try:
        return float(out.strip())
    except ValueError:
        return 0.0


def _finite_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _encoded_cut_duration(value):
    duration = _finite_number(value)
    if duration is None or duration <= 0:
        return 0.0
    return float(Decimal(str(duration)).quantize(Decimal("0.01"), rounding=ROUND_DOWN))


def _source_cut(start, end, requested_duration):
    try:
        source_start = Decimal(str(start))
        source_end = Decimal(str(end))
    except (InvalidOperation, ValueError):
        return None, 0.0
    if (
        not source_start.is_finite()
        or not source_end.is_finite()
        or source_start < 0
        or source_end <= source_start
    ):
        return None, 0.0
    return format(source_start, "f"), _encoded_cut_duration(
        min(Decimal(str(requested_duration)), source_end - source_start)
    )


def remap_source_transcript(transcript, edit_spans):
    """Map canonical source words onto the successfully assembled edit timeline."""
    language = transcript.get("language") if isinstance(transcript, dict) else None
    source_words = []
    if isinstance(transcript, dict):
        segments = transcript.get("segments")
        if isinstance(segments, list):
            for segment in segments:
                if not isinstance(segment, dict):
                    continue
                words = segment.get("words")
                if not isinstance(words, list):
                    continue
                for index, word in enumerate(words):
                    if not isinstance(word, dict):
                        continue
                    text = word.get("word")
                    start = _finite_number(word.get("start"))
                    end = _finite_number(word.get("end"))
                    if (
                        not isinstance(text, str)
                        or not text.strip()
                        or start is None
                        or end is None
                        or start < 0
                        or end <= start
                    ):
                        continue
                    source_words.append((start, end, index, text))

    source_words.sort(key=lambda item: (item[0], item[1], item[2]))
    mapped_segments = []
    if not isinstance(edit_spans, list):
        return {"language": language, "segments": mapped_segments}

    for span in edit_spans:
        if not isinstance(span, dict) or not isinstance(span.get("beat"), dict):
            continue
        beat = span["beat"]
        source_start = _finite_number(beat.get("start"))
        beat_end = _finite_number(beat.get("end"))
        output_start = _finite_number(span.get("output_start"))
        duration = _finite_number(span.get("dur"))
        if (
            source_start is None
            or beat_end is None
            or output_start is None
            or duration is None
            or source_start < 0
            or output_start < 0
            or duration <= 0
        ):
            continue

        source_end = min(beat_end, source_start + duration)
        output_end = output_start + duration
        if source_end <= source_start:
            continue

        words = []
        for word_start, word_end, _index, text in source_words:
            if word_end <= source_start or word_start >= source_end:
                continue
            clipped_start = max(word_start, source_start)
            clipped_end = min(word_end, source_end)
            mapped_start = max(output_start, min(output_end, output_start + clipped_start - source_start))
            mapped_end = max(output_start, min(output_end, output_start + clipped_end - source_start))
            if mapped_end <= mapped_start:
                continue
            words.append({"word": text, "start": mapped_start, "end": mapped_end})

        mapped_segments.append({
            "beat_id": beat.get("id"),
            "start": output_start,
            "end": output_end,
            "text": " ".join(word["word"] for word in words),
            "words": words,
        })

    return {"language": language, "segments": mapped_segments}


def _caption_mode(doc):
    mode = doc.get("caption_mode", "off")
    if mode not in ("word", "off"):
        raise ValueError("A-roll caption_mode must be one of: word, off")
    return mode


def _load_source_transcript(project_dir, doc):
    transcript = load_cached_source_transcript(
        project_dir,
        doc["source_video"],
        language=doc.get("language"),
        model_size=doc.get("caption_whisper_model", AROLL_DEFAULT_MODEL_SIZE),
        compute_type=doc.get("caption_whisper_compute_type", AROLL_DEFAULT_COMPUTE_TYPE),
    )
    if transcript is None:
        raise RuntimeError(
            "A-roll caption transcript is missing, stale, or invalid; "
            "rerun asr_beats.py to regenerate it."
        )
    return transcript


def ffconcat_file_line(path):
    """Return one ffconcat entry safe for Windows paths, spaces, and apostrophes."""
    normalized = os.path.abspath(path).replace("\\", "/")
    escaped = normalized.replace("'", "'\\''")
    return f"file '{escaped}'\n"


def run(project_dir):
    bpath = os.path.join(project_dir, "beats.json")
    with open(bpath) as f:
        doc = json.load(f)
    src = doc["source_video"]
    W, H = RES.get(doc.get("aspect", "9:16"), (1080, 1920))
    caption_mode = _caption_mode(doc)
    tmp = os.path.join(project_dir, "_seg")
    os.makedirs(tmp, exist_ok=True)

    muxed = []
    edit_spans = []
    output_start = 0.0
    for beat in doc["beats"]:
        clip = beat.get("clip_path")
        if not clip or not os.path.exists(clip):
            print(f"[{beat['id']}] no generated clip -- skipped")
            continue
        requested_dur = _encoded_cut_duration(beat.get("dur"))
        source_start, source_cut_dur = _source_cut(
            beat.get("start"), beat.get("end"), requested_dur
        )
        if requested_dur <= 0 or source_start is None:
            print(f"[{beat['id']}] requested duration rounded to zero -- skipped")
            continue
        if source_cut_dur <= 0:
            print(f"[{beat['id']}] source cut duration rounded to zero -- skipped")
            continue
        audio_path = os.path.join(tmp, f"audio_{beat['id']}.m4a")
        ff(["-ss", str(source_start), "-i", src, "-t", f"{source_cut_dur:.2f}",
            "-vn", "-c:a", "aac", audio_path])
        vd = _finite_number(probe_dur(clip))
        ad = _finite_number(probe_dur(audio_path))
        if vd is None or ad is None or vd <= 0 or ad <= 0:
            print(f"[{beat['id']}] couldn't probe duration -- skipped")
            continue
        encoded_dur = _encoded_cut_duration(min(source_cut_dur, vd, ad))
        if encoded_dur <= 0:
            print(f"[{beat['id']}] encoded duration rounded to zero -- skipped")
            continue
        out = os.path.join(tmp, f"muxed_{beat['id']}.mp4")
        fc = (f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,"
              f"crop={W}:{H},setsar=1,fps=24[v]")
        ff(["-i", clip, "-i", audio_path, "-filter_complex", fc, "-map", "[v]", "-map", "1:a:0",
            "-t", f"{encoded_dur:.2f}", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "copy", out])
        muxed.append(out)
        edit_spans.append({"beat": beat, "output_start": output_start, "dur": encoded_dur})
        output_start += encoded_dur
        print(f"[{beat['id']}] muxed ({encoded_dur:.2f}s)")

    if not muxed:
        raise SystemExit("No beats had a generated clip -- run aroll_clips.py first")

    listf = os.path.join(tmp, "concat_list.txt")
    with open(listf, "w") as f:
        for m in muxed:
            f.write(ffconcat_file_line(m))
    final = os.path.join(project_dir, "final.mp4")
    total = sum(span["dur"] for span in edit_spans)
    if caption_mode == "off":
        ff([
            "-f", "concat", "-safe", "0", "-i", listf, "-t", f"{total:.2f}",
            "-map", "0:v:0", "-map", "0:a:0",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "copy", final,
        ])
        print("FINAL:", final, f"({len(muxed)}/{len(doc['beats'])} beats)")
        return

    concat_path = os.path.join(tmp, "aroll_concat.mp4")
    ff(["-f", "concat", "-safe", "0", "-i", listf, "-c", "copy", concat_path])
    transcript = _load_source_transcript(project_dir, doc)
    mapped_transcript = remap_source_transcript(transcript, edit_spans)
    ass_path = Path(project_dir) / "captions" / "captions.ass"
    created = generate_ass(
        mapped_transcript,
        ass_path,
        caption_style=doc.get("caption_style") or "editorial",
        caption_position=doc.get("caption_position", 10),
        video_width=W,
        video_height=H,
    )
    if not created:
        raise RuntimeError("A-roll caption transcript produced no caption events")
    ass_filter_path = ffmpeg_filter_path(str(ass_path))
    ff([
        "-i", concat_path,
        "-filter_complex", f"[0:v]subtitles=filename='{ass_filter_path}'[v]",
        "-map", "[v]", "-map", "0:a:0",
        "-t", f"{total:.2f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-c:a", "copy", final,
    ])
    print("FINAL:", final, f"({len(muxed)}/{len(doc['beats'])} beats)")


if __name__ == "__main__":
    proj = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "..", "out", "aroll-demo")
    run(os.path.abspath(proj))
