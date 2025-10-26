import gc
from faster_whisper import WhisperModel
from typing import Iterator, List, Optional
from app.models import Word, TranscriptionSegment, TranscriptionConfig
import numpy as np
import logging

logger = logging.getLogger("app.transcription.engine")


class TranscriptionEngine:
    def __init__(self, config: TranscriptionConfig):
        logger.info(f"Initializing Whisper model: {config.model_size} on {config.device} with {config.compute_type}")
        self._model = WhisperModel(
            model_size_or_path=config.model_size,
            device=config.device,
            compute_type=config.compute_type
        )
        self._config = config
        logger.info("Whisper model initialized successfully")
    
    def transcribe_audio(
        self, 
        audio_source: str | np.ndarray, 
        context_words: Optional[List[Word]] = None
    ) -> Iterator[TranscriptionSegment]:
        
        transcribe_kwargs = {
            'vad_filter': self._config.vad_filter,
            'language': self._config.language if self._config.language != "auto" else None,
            'word_timestamps': True #Don't need this if transcribing files
        }
        
        if context_words:
            initial_prompt = "".join([w.text for w in context_words]).strip()
            if initial_prompt:
                logger.debug(f"Using context prompt: '{initial_prompt[:50]}'")
                transcribe_kwargs['initial_prompt'] = initial_prompt
                transcribe_kwargs['condition_on_previous_text'] = True
        
        logger.debug(f"Transcribing with VAD={self._config.vad_filter}, lang={self._config.language}")
        segments, _ = self._model.transcribe(audio_source, **transcribe_kwargs)
        
        for segment in segments:
            yield TranscriptionSegment(
                text=segment.text,
                words=[
                    Word(
                        text=word.word,
                        start=word.start,
                        end=word.end,
                        probability=word.probability
                    )
                    for word in segment.words
                ] if segment.words else [],
                no_speech_prob=segment.no_speech_prob
            )
    
    def release(self) -> None:
        logger.info("Releasing transcription engine resources")
        if hasattr(self, '_model') and self._model is not None:
            del self._model
            self._model = None
        logger.info("Transcription engine resources released")

