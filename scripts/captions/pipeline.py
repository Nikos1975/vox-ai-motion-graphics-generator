from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .subtitles import generate_ass
from .transcription import CaptionDependencyError, build_timeline_transcript


def prepare_word_captions(
    project_dir: str | Path,
    beat_spans: list[dict[str, Any]],
    doc: dict[str, Any],
    *,
    video_width: int,
    video_height: int,
) -> Path:
    project = Path(project_dir)
    caption_dir = project / "captions"
    caption_dir.mkdir(parents=True, exist_ok=True)

    model_size = str(
        doc.get("caption_whisper_model")
        or os.environ.get("VOX_CAPTION_WHISPER_MODEL")
        or "base"
    )
    device = str(doc.get("caption_whisper_device") or "auto")
    compute_type = str(doc.get("caption_whisper_compute_type") or "default")
    language = doc.get("language") or doc.get("voice", {}).get("language")

    transcript = build_timeline_transcript(
        project,
        beat_spans,
        language=language,
        model_size=model_size,
        device=device,
        compute_type=compute_type,
    )
    output = caption_dir / "captions.ass"
    created = generate_ass(
        transcript,
        output,
        caption_style=str(doc.get("caption_style", "editorial")),
        caption_position=int(doc.get("caption_position", 10)),
        video_width=video_width,
        video_height=video_height,
    )
    if not created:
        raise RuntimeError("No word-timed caption events were produced from narration audio")
    return output
