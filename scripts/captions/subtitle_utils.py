from __future__ import annotations

import math


def escape_ass_text(text: str) -> str:
    """Escape user text for ASS dialogue payloads."""
    return (
        str(text)
        .replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("\r", " ")
        .replace("\n", r"\N")
    )


def format_ass_time(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("ASS timestamp must be a finite non-negative number")
    centiseconds = int(round(seconds * 100))
    hours, rem = divmod(centiseconds, 360000)
    minutes, rem = divmod(rem, 6000)
    secs, cs = divmod(rem, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"


def ffmpeg_filter_path(path: str) -> str:
    """Escape a path for use inside an ffmpeg filter argument on Windows/POSIX."""
    normalized = str(path).replace("\\", "/")
    normalized = normalized.replace(":", r"\:")
    normalized = normalized.replace("'", r"\'")
    return normalized
