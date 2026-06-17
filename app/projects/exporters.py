"""Derived export views of a :class:`TranscriptDocument`.

These regenerate plain text / subtitle files from the canonical document. They
never read or mutate the document; they only format it.
"""
from typing import Optional, Tuple

from app.projects.transcript_document import TranscriptDocument


def to_txt(document: TranscriptDocument) -> str:
    text = document.full_text()
    return text + "\n" if text else ""


def _segment_span(segment) -> Tuple[Optional[float], Optional[float]]:
    starts = [w.start for w in segment.words if w.start is not None]
    ends = [w.end for w in segment.words if w.end is not None]
    if not starts or not ends:
        return None, None
    return min(starts), max(ends)


def _format_timestamp(seconds: Optional[float], sep: str = ",") -> str:
    if seconds is None or seconds < 0:
        seconds = 0.0
    ms_total = int(round(seconds * 1000))
    hours, ms_total = divmod(ms_total, 3_600_000)
    minutes, ms_total = divmod(ms_total, 60_000)
    secs, ms = divmod(ms_total, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{ms:03d}"


def to_srt(document: TranscriptDocument) -> str:
    blocks = []
    index = 1
    for segment in document.segments:
        start, end = _segment_span(segment)
        if start is None:
            continue
        blocks.append(
            f"{index}\n"
            f"{_format_timestamp(start, ',')} --> {_format_timestamp(end, ',')}\n"
            f"{segment.text.strip()}\n"
        )
        index += 1
    return "\n".join(blocks)


def to_vtt(document: TranscriptDocument) -> str:
    blocks = ["WEBVTT\n"]
    for segment in document.segments:
        start, end = _segment_span(segment)
        if start is None:
            continue
        blocks.append(
            f"{_format_timestamp(start, '.')} --> {_format_timestamp(end, '.')}\n"
            f"{segment.text.strip()}\n"
        )
    return "\n".join(blocks)
