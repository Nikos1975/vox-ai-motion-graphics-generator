from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CaptionStyle:
    font_name: str
    font_size: int
    primary_color: str
    highlight_color: str
    outline_color: str
    back_color: str
    bold: bool
    italic: bool
    outline_size: float
    shadow_depth: float
    animation_type: str
    uppercase: bool
    max_chars_per_line: int
    max_lines: int = 2


_STYLES = {
    "hormozi": CaptionStyle(
        font_name="Arial",
        font_size=76,
        primary_color="&H00FFFFFF",
        highlight_color="&H0000D7FF",
        outline_color="&H00000000",
        back_color="&H80000000",
        bold=True,
        italic=False,
        outline_size=5.0,
        shadow_depth=1.5,
        animation_type="scale",
        uppercase=True,
        max_chars_per_line=22,
    ),
    "mrbeast": CaptionStyle(
        font_name="Arial",
        font_size=72,
        primary_color="&H00FFFFFF",
        highlight_color="&H0000FFFF",
        outline_color="&H00000000",
        back_color="&H80000000",
        bold=True,
        italic=False,
        outline_size=5.0,
        shadow_depth=2.0,
        animation_type="bounce",
        uppercase=True,
        max_chars_per_line=24,
    ),
    "karaoke": CaptionStyle(
        font_name="Arial",
        font_size=66,
        primary_color="&H00FFFFFF",
        highlight_color="&H0000FF00",
        outline_color="&H00000000",
        back_color="&H80000000",
        bold=True,
        italic=False,
        outline_size=4.0,
        shadow_depth=1.0,
        animation_type="karaoke",
        uppercase=False,
        max_chars_per_line=28,
    ),
    "minimal": CaptionStyle(
        font_name="Arial",
        font_size=58,
        primary_color="&H00FFFFFF",
        highlight_color="&H0000D7FF",
        outline_color="&H00000000",
        back_color="&H70000000",
        bold=False,
        italic=False,
        outline_size=2.0,
        shadow_depth=0.5,
        animation_type="highlight",
        uppercase=False,
        max_chars_per_line=32,
    ),
    "bounce": CaptionStyle(
        font_name="Arial",
        font_size=70,
        primary_color="&H00FFFFFF",
        highlight_color="&H00FF9A30",
        outline_color="&H00000000",
        back_color="&H80000000",
        bold=True,
        italic=False,
        outline_size=4.0,
        shadow_depth=1.5,
        animation_type="bounce",
        uppercase=True,
        max_chars_per_line=24,
    ),
    "classic": CaptionStyle(
        font_name="Arial",
        font_size=62,
        primary_color="&H00FFFFFF",
        highlight_color="&H0000D7FF",
        outline_color="&H00000000",
        back_color="&H80000000",
        bold=True,
        italic=False,
        outline_size=4.0,
        shadow_depth=1.0,
        animation_type="highlight",
        uppercase=False,
        max_chars_per_line=30,
    ),
    "editorial": CaptionStyle(
        font_name="Arial",
        font_size=60,
        primary_color="&H00F6F8F8",
        highlight_color="&H002A4BB4",
        outline_color="&H00191612",
        back_color="&H70251F1B",
        bold=True,
        italic=False,
        outline_size=3.0,
        shadow_depth=1.0,
        animation_type="highlight",
        uppercase=False,
        max_chars_per_line=30,
    ),
}

_ALIASES = {
    "white": "classic",
    "paper": "editorial",
}


def get_caption_style(name: str | None) -> CaptionStyle:
    key = (name or "editorial").strip().lower()
    key = _ALIASES.get(key, key)
    if key not in _STYLES:
        supported = ", ".join(sorted({*_STYLES, *_ALIASES}))
        raise ValueError(f"Unknown caption style {name!r}. Supported: {supported}")
    return _STYLES[key]


def supported_caption_styles() -> tuple[str, ...]:
    return tuple(sorted({*_STYLES, *_ALIASES}))
