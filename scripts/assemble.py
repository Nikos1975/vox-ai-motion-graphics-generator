#!/usr/bin/env python3
"""
Assembly stage (ffmpeg): multi-shot clips + per-beat narration + music -> final.mp4
"""
import json
import os
import subprocess
import sys

import text_overlay
from captions import CaptionDependencyError, prepare_word_captions
from captions.subtitle_utils import ffmpeg_filter_path

FPS, TAIL = 24, 0.5
WATERMARK = "Made with MuAPI · muapi-director"
RES = {"16:9": (1920, 1080), "9:16": (1080, 1920), "1:1": (1080, 1080)}


def ff(args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def probe_dur(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", path], capture_output=True, text=True).stdout
    try:
        return float(out.strip())
    except ValueError:
        return 0.0


def shots_of(beat):
    if beat.get("shots"):
        for s in beat["shots"]:
            yield s
    else:
        yield beat


def _prepare_captions(project_dir, beat_spans, doc, width, height):
    mode = str(doc.get("caption_mode", "word")).strip().lower()
    if mode not in {"word", "static", "off"}:
        raise ValueError("caption_mode must be one of: word, static, off")

    if mode == "word":
        try:
            ass_path = prepare_word_captions(
                project_dir,
                beat_spans,
                doc,
                video_width=width,
                video_height=height,
            )
            print(f"Word-timed captions -> {ass_path}")
            return "word", ass_path
        except CaptionDependencyError as exc:
            if doc.get("caption_required", False):
                raise
            print(f"Word-timed captions unavailable ({exc}); falling back to static captions.")
            return "static", None

    return mode, None


def _static_caption_inputs(project_dir, beat_spans, width, height, caption_style):
    overlays = []
    tmp = os.path.join(project_dir, "_seg")
    for i, span in enumerate(beat_spans):
        beat = span["beat"]
        text = beat.get("narration", "")
        if not text:
            continue
        caption_png = os.path.join(tmp, f"cap_{i:02d}.png")
        keyframe = beat.get("keyframe_path") or (
            beat.get("shots") and beat["shots"][0].get("keyframe_path")
        )
        accent = text_overlay.accent_color(keyframe) if keyframe else None
        text_overlay.render_caption(
            text,
            caption_png,
            W=width,
            H=height,
            accent=accent,
            style=caption_style,
        )
        overlays.append((caption_png, span["start"], span["start"] + span["dur"]))
    return overlays


def run(project_dir):
    with open(os.path.join(project_dir, "beats.json")) as f:
        doc = json.load(f)
    beats = doc["beats"]
    W, H = RES.get(doc.get("aspect", "16:9"), (1920, 1080))
    wm_text = doc.get("watermark", WATERMARK)
    mix = doc.get("mix", {})
    music_vol = float(mix.get("music", 0.6))
    voice_vol = float(mix.get("voice", 1.25))
    cap_style = doc.get("caption_style", "white")
    tmp = os.path.join(project_dir, "_seg")
    os.makedirs(tmp, exist_ok=True)

    segs = []
    beat_spans = []
    t = 0.0
    for beat in beats:
        beat_start = t
        shot_list = list(shots_of(beat))
        durs = [float(s.get("dur", 10)) for s in shot_list]
        need = float(beat.get("narration_dur", sum(durs))) + TAIL
        if sum(durs) < need:
            durs[-1] += need - sum(durs)
        for s, d in zip(shot_list, durs):
            encoded_dur = round(d, 2)
            segs.append({"clip": s["clip_path"], "dur": encoded_dur})
            t += encoded_dur
        beat_spans.append({"start": beat_start, "dur": round(t - beat_start, 2), "beat": beat})
    total = round(t, 2)

    seg_files = []
    for i, s in enumerate(segs):
        out = os.path.join(tmp, f"seg_{i:02d}.mp4")
        cd = probe_dur(s["clip"])
        factor = s["dur"] / cd if cd > 0 else 1.0
        pre = f"setpts={factor:.4f}*PTS," if factor > 1.02 else ""
        fc = (f"[0:v]{pre}split[s0][s1];"
              f"[s0]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
              f"boxblur=26:2,eq=brightness=-0.05[bg];"
              f"[s1]scale={W}:{H}:force_original_aspect_ratio=decrease[fg];"
              f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1,fps={FPS},"
              f"tpad=stop_mode=clone:stop_duration=1[v]")
        ff(["-i", s["clip"], "-an", "-filter_complex", fc, "-map", "[v]", "-t", f"{s['dur']}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", out])
        seg_files.append(out)

    listf = os.path.join(tmp, "list.txt")
    with open(listf, "w") as f:
        for s in seg_files:
            f.write(f"file '{s}'\n")

    v_concat = os.path.join(tmp, "v_concat.mp4")
    ff(["-f", "concat", "-safe", "0", "-i", listf, "-c", "copy", v_concat])

    narr_inputs, amix_filters = [], []
    for i, span in enumerate(beat_spans):
        b = span["beat"]
        apath = b.get("audio_path")
        if apath and os.path.exists(apath):
            idx = (len(narr_inputs) // 2) + 1
            narr_inputs.extend(["-i", apath])
            delay_ms = int(span["start"] * 1000)
            amix_filters.append(f"[{idx}:a]volume={voice_vol},adelay={delay_ms}|{delay_ms}[a{idx}];")

    bgm_path = os.path.join(project_dir, "audio", "bgm.mp3")
    has_bgm = os.path.exists(bgm_path)
    if has_bgm:
        bgm_idx = (len(narr_inputs) // 2) + 1
        narr_inputs.extend(["-stream_loop", "-1", "-i", bgm_path])

    filter_lines = []
    if amix_filters:
        filter_lines.extend(amix_filters)
        ins = "".join(f"[a{i+1}]" for i in range(len(amix_filters)))
        filter_lines.append(f"{ins}amix=inputs={len(amix_filters)}:duration=longest:dropout_transition=0.5[vo];")
        if has_bgm:
            filter_lines.append(
                f"[{bgm_idx}:a]volume={music_vol}[bgm0];"
                f"[vo]asplit=2[vo1][vo2];"
                f"[bgm0][vo1]sidechaincompress=threshold=0.08:ratio=6:attack=15:release=250[bgm_ducked];"
                f"[vo2][bgm_ducked]amix=inputs=2:duration=longest:weights=1.2 0.7[aout]"
            )
        else:
            filter_lines.append(f"[vo]anull[aout]")
    elif has_bgm:
        filter_lines.append(f"[{bgm_idx}:a]volume={music_vol}[aout]")
    else:
        filter_lines.append("anullsrc=channel_layout=stereo:sample_rate=44100[aout]")

    full_filter = "".join(filter_lines) + ";[aout]apad[aout_padded]"
    audio_full = os.path.join(tmp, "audio_mixed.m4a")
    ff(["-i", v_concat, *narr_inputs, "-filter_complex", full_filter,
        "-map", "[aout_padded]", "-t", f"{total}", "-c:a", "aac", "-b:a", "192k", audio_full])

    caption_mode, ass_path = _prepare_captions(project_dir, beat_spans, doc, W, H)
    cap_overlays = []
    if caption_mode == "static":
        cap_overlays = _static_caption_inputs(project_dir, beat_spans, W, H, cap_style)

    wm_png = os.path.join(tmp, "watermark.png")
    text_overlay.render_watermark(wm_text, wm_png, W=W, H=H)

    inputs = ["-i", v_concat]
    filter_parts = []
    last_video = "[0:v]"
    next_input_idx = 1

    if caption_mode == "word" and ass_path is not None:
        ass_filter_path = ffmpeg_filter_path(str(ass_path))
        filter_parts.append(f"{last_video}subtitles=filename='{ass_filter_path}'[vword]")
        last_video = "[vword]"

    for i, (caption_png, start, end) in enumerate(cap_overlays):
        inputs.extend(["-i", caption_png])
        out_label = f"[vcap{i + 1}]"
        filter_parts.append(
            f"{last_video}[{next_input_idx}:v]overlay=0:0:enable='between(t,{start},{end})'{out_label}"
        )
        last_video = out_label
        next_input_idx += 1

    inputs.extend(["-i", wm_png])
    watermark_idx = next_input_idx
    filter_parts.append(f"{last_video}[{watermark_idx}:v]overlay=0:0[vfinal]")
    next_input_idx += 1

    inputs.extend(["-i", audio_full])
    audio_idx = next_input_idx
    final_mp4 = os.path.join(project_dir, "final.mp4")
    ff([
        *inputs,
        "-filter_complex", ";".join(filter_parts),
        "-map", "[vfinal]", "-map", f"{audio_idx}:a", "-t", f"{total}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "copy", final_mp4,
    ])

    print(f"Assembly finished -> {final_mp4}")


if __name__ == "__main__":
    proj = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "..", "out", "demo")
    run(os.path.abspath(proj))
