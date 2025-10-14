from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float
    probability: float


@dataclass(frozen=True)
class TranscriptionSegment:
    words: List[Word]
    no_speech_probability: float


@dataclass(frozen=True)
class AudioChunk:
    data: bytes
    sample_rate: int
    offset: float


@dataclass(frozen=True)
class TranscriptionConfig:
    model_size: str
    device: str
    compute_type: str
    language: str
    sample_rate: int
    chunk_duration: float
    overlap_duration: float
    no_speech_threshold: float
    vad_filter: bool = True
    word_timestamps: bool = True

