from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable


class CaptionDependencyError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_model(model_size: str, device: str, compute_type: str):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise CaptionDependencyError(
            "Word-timed captions require faster-whisper. "
            "Install with: pip install -r requirements-captions.txt"
        ) from exc
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def _transcribe_with_model(model: Any, audio_path: Path, language: str | None) -> dict[str, Any]:
    segments_iter, info = model.transcribe(
        str(audio_path),
        language=language or None,
        word_timestamps=True,
        vad_filter=True,
    )
    segments = []
    for segment in segments_iter:
        words = []
        for word in getattr(segment, "words", None) or []:
            if word.start is None or word.end is None:
                continue
            words.append(
                {
                    "word": str(word.word).strip(),
                    "start": float(word.start),
                    "end": float(word.end),
                }
            )
        segments.append(
            {
                "start": float(segment.start),
                "end": float(segment.end),
                "text": str(segment.text).strip(),
                "words": words,
            }
        )
    detected = getattr(info, "language", None) or language or "unknown"
    return {"language": detected, "segments": segments}


def merge_beat_transcripts(
    beat_spans: Iterable[dict[str, Any]],
    transcribe_one: Callable[[Path], dict[str, Any]],
    *,
    language: str | None,
) -> dict[str, Any]:
    merged_segments: list[dict[str, Any]] = []
    detected_language = language

    for span in beat_spans:
        beat = span.get("beat", {})
        audio_path_raw = beat.get("audio_path")
        if not audio_path_raw:
            continue
        audio_path = Path(audio_path_raw)
        if not audio_path.exists():
            continue

        base = float(span["start"])
        limit = base + float(span["dur"])
        local = transcribe_one(audio_path)
        detected_language = detected_language or local.get("language")

        for segment in local.get("segments", []):
            words = []
            for item in segment.get("words", []):
                start = base + float(item["start"])
                end = base + float(item["end"])
                if start >= limit:
                    continue
                end = min(end, limit)
                if end <= start:
                    continue
                words.append(
                    {
                        "word": str(item.get("word", "")).strip(),
                        "start": round(start, 4),
                        "end": round(end, 4),
                    }
                )
            if not words:
                continue
            merged_segments.append(
                {
                    "beat_id": beat.get("id"),
                    "start": words[0]["start"],
                    "end": words[-1]["end"],
                    "text": " ".join(item["word"] for item in words).strip(),
                    "words": words,
                }
            )

    merged_segments.sort(key=lambda item: (item["start"], item["end"]))
    return {"language": detected_language or "unknown", "segments": merged_segments}


def _source_fingerprint(beat_spans: Iterable[dict[str, Any]]) -> str:
    payload = []
    for span in beat_spans:
        beat = span.get("beat", {})
        audio_raw = beat.get("audio_path")
        if not audio_raw:
            continue
        audio = Path(audio_raw)
        if not audio.exists():
            continue
        payload.append(
            {
                "beat_id": beat.get("id"),
                "audio_sha256": _sha256_file(audio),
                "start": round(float(span["start"]), 4),
                "dur": round(float(span["dur"]), 4),
            }
        )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_timeline_transcript(
    project_dir: str | Path,
    beat_spans: list[dict[str, Any]],
    *,
    language: str | None = None,
    model_size: str = "base",
    device: str = "auto",
    compute_type: str = "default",
) -> dict[str, Any]:
    project = Path(project_dir)
    caption_dir = project / "captions"
    caption_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = caption_dir / "transcript.json"
    fingerprint = _source_fingerprint(beat_spans)

    if transcript_path.exists():
        try:
            cached = json.loads(transcript_path.read_text(encoding="utf-8"))
            if (
                cached.get("source_fingerprint") == fingerprint
                and cached.get("model") == model_size
                and cached.get("requested_language") == language
            ):
                return cached
        except (OSError, json.JSONDecodeError):
            pass

    model = _load_model(model_size, device, compute_type)

    def transcribe_one(audio_path: Path) -> dict[str, Any]:
        return _transcribe_with_model(model, audio_path, language)

    merged = merge_beat_transcripts(beat_spans, transcribe_one, language=language)
    merged.update(
        {
            "schema_version": 1,
            "source_fingerprint": fingerprint,
            "model": model_size,
            "requested_language": language,
        }
    )
    transcript_path.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return merged
