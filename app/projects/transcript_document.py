"""Canonical transcript document and its JSON serialization.

``TranscriptDocument`` wraps the same ``TranscriptionSegment`` / ``Word``
objects the engine already produces, so nothing upstream has to change. It is
the single source of truth for a transcription project — TXT/SRT/VTT are
derived from it (see :mod:`app.projects.exporters`).
"""
import json
from dataclasses import dataclass, field
from typing import List, Optional

from app.models import Word, TranscriptionSegment

SCHEMA_VERSION = 1


@dataclass
class TranscriptDocument:
    segments: List[TranscriptionSegment] = field(default_factory=list)
    version: int = SCHEMA_VERSION

    @classmethod
    def from_segments(cls, segments) -> "TranscriptDocument":
        return cls(segments=list(segments))

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "segments": [
                {
                    "text": seg.text,
                    "no_speech_prob": seg.no_speech_prob,
                    "words": [
                        {
                            "text": w.text,
                            "start": w.start,
                            "end": w.end,
                            "probability": w.probability,
                        }
                        for w in seg.words
                    ],
                }
                for seg in self.segments
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TranscriptDocument":
        segments = []
        for seg in data.get("segments", []):
            words = [
                Word(
                    text=w["text"],
                    start=w.get("start"),
                    end=w.get("end"),
                    probability=w.get("probability"),
                )
                for w in seg.get("words", [])
            ]
            segments.append(
                TranscriptionSegment(
                    text=seg["text"],
                    words=words,
                    no_speech_prob=seg.get("no_speech_prob", 0.0),
                )
            )
        return cls(segments=segments, version=data.get("version", SCHEMA_VERSION))

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "TranscriptDocument":
        return cls.from_dict(json.loads(text))

    def full_text(self) -> str:
        """Joined segment text, one segment per line."""
        return "\n".join(seg.text.strip() for seg in self.segments)

    def duration(self) -> float:
        """Best-effort media duration: the latest word end-time seen."""
        end = 0.0
        for seg in self.segments:
            for w in seg.words:
                if w.end is not None and w.end > end:
                    end = w.end
        return end
