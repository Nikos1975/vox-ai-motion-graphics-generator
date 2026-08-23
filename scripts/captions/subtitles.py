from __future__ import annotations

from pathlib import Path
from typing import Any

from .caption_styles import CaptionStyle, get_caption_style
from .subtitle_utils import escape_ass_text, format_ass_time


def _flatten_words(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for segment in transcript.get("segments", []):
        for item in segment.get("words", []):
            text = str(item.get("word", "")).strip()
            start = item.get("start")
            end = item.get("end")
            if not text or start is None or end is None:
                continue
            start_f = float(start)
            end_f = float(end)
            if start_f < 0 or end_f <= start_f:
                continue
            words.append({"word": text, "start": start_f, "end": end_f})
    words.sort(key=lambda item: (item["start"], item["end"]))
    return words


def _group_words(words: list[dict[str, Any]], style: CaptionStyle) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_line = 0
    line_chars = 0

    for item in words:
        word = item["word"]
        required = len(word) + (1 if line_chars else 0)
        if current and line_chars + required > style.max_chars_per_line:
            if current_line + 1 < style.max_lines:
                current_line += 1
                line_chars = 0
                required = len(word)
            else:
                groups.append(current)
                current = []
                current_line = 0
                line_chars = 0
                required = len(word)

        copied = dict(item)
        copied["line"] = current_line
        current.append(copied)
        line_chars += required

    if current:
        groups.append(current)
    return groups


def _render_group_text(group: list[dict[str, Any]], current_idx: int, style: CaptionStyle) -> str:
    parts: list[str] = []
    previous_line: int | None = None

    for idx, item in enumerate(group):
        line = int(item["line"])
        if previous_line is not None:
            parts.append(r"\N" if line != previous_line else " ")

        word = item["word"].upper() if style.uppercase else item["word"]
        word = escape_ass_text(word)

        if idx == current_idx:
            if style.animation_type == "scale":
                word = rf"{{\fscx112\fscy112\c{style.highlight_color}}}{word}{{\r}}"
            elif style.animation_type == "bounce":
                word = rf"{{\t(0,70,\fscx120\fscy120)\t(70,140,\fscx100\fscy100)\c{style.highlight_color}}}{word}{{\r}}"
            elif style.animation_type == "karaoke":
                dur_cs = max(1, int(round((item["end"] - item["start"]) * 100)))
                word = rf"{{\kf{dur_cs}\c{style.highlight_color}}}{word}{{\r}}"
            else:
                word = rf"{{\c{style.highlight_color}}}{word}{{\r}}"
        parts.append(word)
        previous_line = line

    return "".join(parts)


def generate_ass(
    transcript: dict[str, Any],
    output_path: str | Path,
    *,
    caption_style: str = "editorial",
    caption_position: int = 10,
    video_width: int = 1080,
    video_height: int = 1920,
) -> bool:
    """Generate word-timed ASS subtitles from a canonical word transcript."""
    if not 0 <= int(caption_position) <= 50:
        raise ValueError("caption_position must be between 0 and 50")
    if video_width <= 0 or video_height <= 0:
        raise ValueError("video dimensions must be positive")

    style = get_caption_style(caption_style)
    words = _flatten_words(transcript)
    if not words:
        return False

    groups = _group_words(words, style)
    scale = max(video_height / 1920.0, 0.35)
    font_size = max(18, int(round(style.font_size * scale)))
    outline = round(style.outline_size * scale, 1)
    shadow = round(style.shadow_depth * scale, 1)
    margin_v = int(video_height * int(caption_position) / 100)
    bold = -1 if style.bold else 0
    italic = -1 if style.italic else 0

    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {video_width}",
        f"PlayResY: {video_height}",
        "ScaledBorderAndShadow: yes",
        "WrapStyle: 3",
        "",
        "[V4+ Styles]",
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding",
        (
            "Style: Default,"
            f"{style.font_name},{font_size},{style.primary_color},{style.highlight_color},"
            f"{style.outline_color},{style.back_color},{bold},{italic},0,0,100,100,0,0,1,"
            f"{outline},{shadow},2,40,40,{margin_v},1"
        ),
        "",
        "[Events]",
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
    ]

    for group in groups:
        for idx, item in enumerate(group):
            start = float(item["start"])
            if idx + 1 < len(group):
                end = max(start + 0.01, float(group[idx + 1]["start"]))
            else:
                end = max(start + 0.01, float(item["end"]))
            text = _render_group_text(group, idx, style)
            lines.append(
                "Dialogue: 0,"
                f"{format_ass_time(start)},{format_ass_time(end)},Default,,0,0,0,,{text}"
            )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True
