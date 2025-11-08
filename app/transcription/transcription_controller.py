import logging
from app.models import TranscriptionConfig
from app.transcription.orchestrator import TranscriptionOrchestrator
from app.transcription.transcription_engine import TranscriptionEngine
from app.utils.document_writer import TranscriptionWriter
from app.audio.audio_data_provider import AudioDataProvider

logger = logging.getLogger("app.transcription.controller")


class TranscriptionController:
    def __init__(self, config: TranscriptionConfig):
        self._config = config
        self._audio_data_provider = AudioDataProvider(config.sample_rate, device_index=config.mic_id)
        self._orchestrator = None
        self._engine = None

    def run(self, output_file: str = "transcription.txt") -> None:
        logger.info(f"Starting transcription to file: {output_file}")
        self._audio_data_provider.start()
        self._orchestrator = TranscriptionOrchestrator(self._config, output_file, self._audio_data_provider)
        self._orchestrator.start()

    def stop(self) -> None:
        logger.info("Stopping transcription")
        if self._orchestrator:
            self._orchestrator.stop()
        self._audio_data_provider.stop()

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
                if isinstance(value, dict):
                    nested_obj = getattr(self._config, key)
                    for sub_key, sub_value in value.items():
                        if hasattr(nested_obj, sub_key):
                            setattr(nested_obj, sub_key, sub_value)
                else:
                    setattr(self._config, key, value)

    def shutdown(self) -> None:
        logger.info("Shutting down transcription controller")
        if self._orchestrator:
            self._orchestrator.release()
            self._orchestrator = None
        self._audio_data_provider.release()
        if self._engine:
            self._engine.release()
            self._engine = None
        logger.info("Transcription controller shutdown complete")

    def get_audio_chunk(self, chunk_size: int = 1024):
        return self._audio_data_provider.get_audio_chunk(chunk_size)
