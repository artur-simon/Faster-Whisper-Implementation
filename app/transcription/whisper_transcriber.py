from typing import Optional
from app.models import TranscriptionConfig
from app.transcription.orchestrator import TranscriptionOrchestrator
from app.transcription.transcription_engine import TranscriptionEngine
from app.utils.document_writer import TranscriptionWriter


class LiveWhisperTranscriber:
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
        microphone_index: Optional[int] = None,
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
            microphone_index=microphone_index,
            should_paste_content=should_paste_content,
        )
        self._orchestrator = None
        self._engine = None

    def run(self, output_file: str = "transcription.txt") -> None:
        self._orchestrator = TranscriptionOrchestrator(self._config, output_file)
        self._orchestrator.start()

    def stop(self) -> None:
        if self._orchestrator:
            self._orchestrator.stop()

    def transcribe_audio_file(self, audio_path: str, output_file: str) -> None:
        if self._engine is None:
            self._engine = TranscriptionEngine(self._config)

        writer = TranscriptionWriter(output_file)

        try:
            segments = self._engine.transcribe_file(audio_path)
            for segment in segments:
                if segment.no_speech_probability < self._config.no_speech_threshold:
                    writer.write_string_to_file(segment.text, False)
                    writer.write_string_to_file("\n")
        except Exception as e:
            raise Exception(f"Error transcribing audio file: {e}")

    def update_input_config(
        self,
        language: Optional[str] = None,
        mic_id: Optional[int] = None,
        should_paste_content: Optional[bool] = None,
    ):
        if mic_id is not None:
            self._config.microphone_index = mic_id
        if language is not None:
            self._config.language = language
        if should_paste_content is not None:
            self._config.should_paste_content = should_paste_content

    def shutdown(self) -> None:
        if self._orchestrator:
            self._orchestrator.release()
            self._orchestrator = None
        if self._engine:
            self._engine.release()
            self._engine = None
