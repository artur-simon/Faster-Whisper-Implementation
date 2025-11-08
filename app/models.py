from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float
    probability: float


@dataclass(frozen=True)
class TranscriptionSegment:
    text: str
    words: List[Word]
    no_speech_prob: float


@dataclass(frozen=True)
class AudioChunk:
    data: bytes
    sample_rate: int
    offset: float


@dataclass()
class LocalAgreementConfig:
    agreement_count: int = 2
    edit_threshold: float = 0.2
    confidence_threshold: float = 0.8
    min_words: int = 3
    context_size: int = 100


@dataclass()
class VADParams:
    threshold: float = 0.5
    min_speech_duration_ms: int = 400
    min_silence_duration_ms: int = 400


@dataclass()
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
    mic_id: int = 0
    should_paste_content: bool = False
    use_previous_context: bool = True
    transcription_algorithm: str = 'simple_overlap_resolve'
    local_agreement_config: LocalAgreementConfig = field(default_factory=LocalAgreementConfig)
    vad_params: VADParams = field(default_factory=VADParams)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'TranscriptionConfig':
        config_copy = config_dict.copy()
        if 'local_agreement_config' in config_copy and isinstance(config_copy['local_agreement_config'], dict):
            config_copy['local_agreement_config'] = LocalAgreementConfig(**config_copy['local_agreement_config'])
        if 'vad_params' in config_copy and isinstance(config_copy['vad_params'], dict):
            config_copy['vad_params'] = VADParams(**config_copy['vad_params'])
        return cls(**config_copy)

