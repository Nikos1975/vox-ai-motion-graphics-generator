#!/usr/bin/env python3
"""
A-roll assembly: muxes each beat's generated visual clip with the ORIGINAL beat segment's audio.
"""
import json
import math
import os
import subprocess
import sys
from decimal import Decimal, InvalidOperation, ROUND_DOWN, ROUND_HALF_UP
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
FPS = 24
# Empirical 24 fps/AAC timestamp floor: 0.10-0.50s cuts exceed the 5ms bound; 1.00s does not.
MIN_RENDER_DUR = 1.0
CENTISECOND = Decimal("0.01")


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
    return float(Decimal(str(duration)).quantize(CENTISECOND, rounding=ROUND_DOWN))


def _timeline_decimal(value):
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    if not number.is_finite():
        return None
    return number.quantize(CENTISECOND, rounding=ROUND_HALF_UP)


def _source_cut(start, end, requested_duration):
    source_start = _timeline_decimal(start)
    source_end = _timeline_decimal(end)
    requested = _timeline_decimal(requested_duration)
    if (
        source_start is None
        or source_end is None
        or requested is None
        or source_start < 0
        or source_end <= source_start
    ):
        return None, 0.0
    return format(source_start, "f"), _encoded_cut_duration(
        min(requested, source_end - source_start)
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

    video_segments = []
    audio_segments = []
    edit_spans = []
    audio_timeline_end = Decimal("0")
    previous_frame_end = 0
    for beat in doc["beats"]:
        clip = beat.get("clip_path")
        if not clip or not os.path.exists(clip):
            print(f"[{beat['id']}] no generated clip -- skipped")
            continue
        requested = _timeline_decimal(beat.get("dur"))
        requested_dur = float(requested) if requested is not None and requested > 0 else 0.0
        source_start, source_cut_dur = _source_cut(
            beat.get("start"), beat.get("end"), requested_dur
        )
        if requested_dur <= 0 or source_start is None:
            print(f"[{beat['id']}] requested duration rounded to zero -- skipped")
            continue
        if source_cut_dur <= 0:
            print(f"[{beat['id']}] source cut duration rounded to zero -- skipped")
            continue
        if source_cut_dur < MIN_RENDER_DUR:
            print(
                f"[{beat['id']}] source cut is below the empirical minimum render duration "
                f"({MIN_RENDER_DUR:.2f}s at {FPS} fps/AAC) -- skipped"
            )
            continue
        vd = _finite_number(probe_dur(clip))
        if vd is None or vd <= 0:
            print(f"[{beat['id']}] couldn't probe clip duration -- skipped")
            continue
        if _encoded_cut_duration(min(source_cut_dur, vd)) < MIN_RENDER_DUR:
            print(f"[{beat['id']}] clip is below the empirical minimum render duration -- skipped")
            continue
        audio_path = os.path.join(tmp, f"audio_{beat['id']}.wav")
        ff(["-ss", str(source_start), "-i", src, "-t", f"{source_cut_dur:.2f}",
            "-vn", "-c:a", "pcm_s16le", audio_path])
        ad = _finite_number(probe_dur(audio_path))
        if ad is None or ad <= 0:
            print(f"[{beat['id']}] couldn't probe audio duration -- skipped")
            continue
        encoded_dur = _encoded_cut_duration(min(source_cut_dur, vd, ad))
        if encoded_dur < MIN_RENDER_DUR:
            print(f"[{beat['id']}] effective duration is below the empirical minimum render duration -- skipped")
            continue
        encoded_duration = Decimal(str(encoded_dur))
        audio_timeline_start = audio_timeline_end
        audio_timeline_end += encoded_duration
        frame_end = int(
            (audio_timeline_end * FPS).to_integral_value(rounding=ROUND_HALF_UP)
        )
        frames_this = max(1, frame_end - previous_frame_end)
        previous_frame_end = frame_end
        video_path = os.path.join(tmp, f"video_{beat['id']}.mp4")
        fc = (f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,"
              f"crop={W}:{H},setsar=1,fps={FPS},setpts=PTS-STARTPTS[v]")
        ff(["-i", clip, "-filter_complex", fc, "-map", "[v]", "-frames:v", str(frames_this),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", video_path])
        video_segments.append(video_path)
        audio_segments.append((audio_path, encoded_duration))
        effective_beat = dict(beat)
        effective_beat["start"] = float(source_start)
        effective_beat["end"] = float(Decimal(source_start) + encoded_duration)
        edit_spans.append({
            "beat": effective_beat,
            "output_start": float(audio_timeline_start),
            "dur": encoded_dur,
        })
        print(f"[{beat['id']}] rendered ({encoded_dur:.2f}s)")

    if not video_segments:
        raise SystemExit(
            "No beats had a generated clip or minimum render duration -- "
            "run aroll_clips.py and check beat durations."
        )

    video_list = os.path.join(tmp, "video_concat_list.txt")
    with open(video_list, "w") as f:
        for path in video_segments:
            f.write(ffconcat_file_line(path))
    video_concat = os.path.join(tmp, "aroll_video.mp4")
    ff([
        "-f", "concat", "-safe", "0", "-i", video_list,
        "-map", "0:v:0", "-c:v", "copy", "-an", video_concat,
    ])
    audio_concat = os.path.join(tmp, "aroll_audio.wav")
    audio_args = []
    trim_filters = []
    for index, (audio_path, duration) in enumerate(audio_segments):
        audio_args.extend(["-i", audio_path])
        label = "a" if len(audio_segments) == 1 else f"a{index}"
        trim_filters.append(
            f"[{index}:a]atrim=duration={duration:.2f},asetpts=PTS-STARTPTS[{label}]"
        )
    if len(audio_segments) == 1:
        audio_graph = trim_filters[0]
    else:
        concat_inputs = "".join(f"[a{index}]" for index in range(len(audio_segments)))
        audio_graph = ";".join([
            *trim_filters,
            f"{concat_inputs}concat=n={len(audio_segments)}:v=0:a=1[a]",
        ])
    ff([
        *audio_args, "-filter_complex", audio_graph,
        "-map", "[a]", "-c:a", "pcm_s16le", audio_concat,
    ])
    final = os.path.join(project_dir, "final.mp4")
    total_arg = format(audio_timeline_end, ".2f")
    audio_filter = f"[1:a]atrim=duration={total_arg},asetpts=PTS-STARTPTS[a]"
    if caption_mode == "off":
        ff([
            "-i", video_concat, "-i", audio_concat, "-t", total_arg,
            "-filter_complex", audio_filter, "-map", "0:v:0", "-map", "[a]",
            "-c:v", "copy", "-c:a", "aac", final,
        ])
        print("FINAL:", final, f"({len(video_segments)}/{len(doc['beats'])} beats)")
        return

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
        "-i", video_concat, "-i", audio_concat,
        "-filter_complex", (
            f"[0:v]subtitles=filename='{ass_filter_path}',setpts=PTS-STARTPTS[v];"
            f"[1:a]atrim=duration={total_arg},asetpts=PTS-STARTPTS[a]"
        ),
        "-map", "[v]", "-map", "[a]",
        "-t", total_arg,
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-c:a", "aac", final,
    ])
    print("FINAL:", final, f"({len(video_segments)}/{len(doc['beats'])} beats)")


if __name__ == "__main__":
    proj = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "..", "out", "aroll-demo")
    run(os.path.abspath(proj))
