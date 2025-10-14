from faster_whisper import WhisperModel
from typing import List, Optional
from app.models import Word, TranscriptionSegment, TranscriptionConfig


class TranscriptionEngine:
    def __init__(self, config: TranscriptionConfig):
        self._model = WhisperModel(
            model_size_or_path=config.model_size,
            device=config.device,
            compute_type=config.compute_type
        )
        self._config = config
    
    def transcribe_file(
        self, 
        audio_path: str, 
        context_words: Optional[List[Word]] = None
    ) -> List[TranscriptionSegment]:
        
        transcribe_kwargs = {
            'vad_filter': self._config.vad_filter,
            'language': self._config.language if self._config.language != "auto" else None,
            'word_timestamps': self._config.word_timestamps,
        }
        
        if context_words:
            initial_prompt = "".join([w.text for w in context_words]).strip()
            if initial_prompt:
                transcribe_kwargs['initial_prompt'] = initial_prompt
                transcribe_kwargs['condition_on_previous_text'] = True
        
        segments, _ = self._model.transcribe(audio_path, **transcribe_kwargs)
        
        return [
            TranscriptionSegment(
                words=[
                    Word(
                        text=word.word,
                        start=word.start,
                        end=word.end,
                        probability=word.probability
                    )
                    for word in segment.words
                ] if segment.words else [],
                no_speech_probability=segment.no_speech_prob
            )
            for segment in segments
        ]
    
    def release(self) -> None:
        if hasattr(self, '_model') and self._model is not None:
            del self._model
            self._model = None

