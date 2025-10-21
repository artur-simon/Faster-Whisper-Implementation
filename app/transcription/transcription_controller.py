from typing import Optional
import logging
from app.models import TranscriptionConfig
from app.transcription.orchestrator import TranscriptionOrchestrator
from app.transcription.transcription_engine import TranscriptionEngine
from app.utils.document_writer import TranscriptionWriter

logger = logging.getLogger("app.transcription.controller")


class TranscriptionController:
    def __init__(
        self,
        model_size: str = "large-v3",
        sample_rate: int = 44100,
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = "auto",
        chunk_duration: float = 5.0,
        overlap_duration: float = 1.0,
        no_speech_threshold: float = 0.6,
        mic_id: Optional[int] = None,
        should_paste_content: bool = False
    ):
        self._config = TranscriptionConfig(
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            language=language,
            sample_rate=sample_rate,
            chunk_duration=chunk_duration,
            overlap_duration=overlap_duration,
            no_speech_threshold=no_speech_threshold,
            mic_id=mic_id,
            should_paste_content=should_paste_content,
        )
        self._orchestrator = None
        self._engine = None

    def run(self, output_file: str = "transcription.txt") -> None:
        logger.info(f"Starting transcription to file: {output_file}")
        self._orchestrator = TranscriptionOrchestrator(self._config, output_file)
        self._orchestrator.start()

    def stop(self) -> None:
        logger.info("Stopping transcription")
        if self._orchestrator:
            self._orchestrator.stop()

    def transcribe_audio_file(self, audio_path: str, output_file: str) -> None:
        logger.info(f"Transcribing audio file: {audio_path}")
        if self._engine is None:
            self._engine = TranscriptionEngine(self._config)

        writer = TranscriptionWriter(output_file, self._config)

        try:
            segments = self._engine.transcribe_audio(audio_path)
            segment_count = 0
            for segment in segments:
                if segment.no_speech_prob < self._config.no_speech_threshold:
                    writer.write_string_to_file(segment.text)
                    writer.write_string_to_file("\n")
                    segment_count += 1
            logger.info(f"Transcription completed: {segment_count} segments processed")
        except Exception as e:
            logger.error(f"Error transcribing audio file: {e}", exc_info=True)
            raise Exception(f"Error transcribing audio file: {e}")

    def update_input_config(self, **kwargs):
        logger.debug(f"Updating config: {kwargs}")
        for key, value in kwargs.items():
            if value is not None and hasattr(self._config, key):
                setattr(self._config, key, value)

    def shutdown(self) -> None:
        logger.info("Shutting down transcription controller")
        if self._orchestrator:
            self._orchestrator.release()
            self._orchestrator = None
        if self._engine:
            self._engine.release()
            self._engine = None
        logger.info("Transcription controller shutdown complete")
