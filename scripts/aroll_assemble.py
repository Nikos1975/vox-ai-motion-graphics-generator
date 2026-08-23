#!/usr/bin/env python3
"""
A-roll assembly: muxes each beat's generated visual clip with the ORIGINAL beat segment's audio.
"""
import json
import math
import os
import subprocess
import sys

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

        if words:
            mapped_segments.append({
                "beat_id": beat.get("id"),
                "start": output_start,
                "end": output_end,
                "text": " ".join(word["word"] for word in words),
                "words": words,
            })

    return {"language": language, "segments": mapped_segments}


def run(project_dir):
    bpath = os.path.join(project_dir, "beats.json")
    with open(bpath) as f:
        doc = json.load(f)
    src = doc["source_video"]
    W, H = RES.get(doc.get("aspect", "9:16"), (1080, 1920))
    tmp = os.path.join(project_dir, "_seg")
    os.makedirs(tmp, exist_ok=True)

    muxed = []
    for beat in doc["beats"]:
        clip = beat.get("clip_path")
        if not clip or not os.path.exists(clip):
            print(f"[{beat['id']}] no generated clip -- skipped")
            continue
        audio_path = os.path.join(tmp, f"audio_{beat['id']}.aac")
        ff(["-ss", f"{beat['start']:.2f}", "-i", src, "-t", f"{beat['dur']:.2f}",
            "-vn", "-c:a", "aac", audio_path])
        vd, ad = probe_dur(clip), probe_dur(audio_path)
        d = min(vd, ad) if vd and ad else (vd or ad)
        if not d:
            print(f"[{beat['id']}] couldn't probe duration -- skipped")
            continue
        out = os.path.join(tmp, f"muxed_{beat['id']}.mp4")
        fc = (f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,"
              f"crop={W}:{H},setsar=1,fps=24[v]")
        ff(["-i", clip, "-i", audio_path, "-filter_complex", fc, "-map", "[v]", "-map", "1:a:0",
            "-t", f"{d:.2f}", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", out])
        muxed.append(out)
        print(f"[{beat['id']}] muxed ({d:.2f}s)")

    if not muxed:
        raise SystemExit("No beats had a generated clip -- run aroll_clips.py first")

    listf = os.path.join(tmp, "concat_list.txt")
    with open(listf, "w") as f:
        for m in muxed:
            f.write(f"file '{os.path.abspath(m)}'\n")
    final = os.path.join(project_dir, "final.mp4")
    ff(["-f", "concat", "-safe", "0", "-i", listf, "-c", "copy", final])
    print("FINAL:", final, f"({len(muxed)}/{len(doc['beats'])} beats)")


if __name__ == "__main__":
    proj = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "..", "out", "aroll-demo")
    run(os.path.abspath(proj))
