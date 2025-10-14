from app.models import TranscriptionConfig
from app.transcription.orchestrator import TranscriptionOrchestrator


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
        )
        self._orchestrator = None

    def run(self, output_file: str = "transcription.txt") -> None:
        self._orchestrator = TranscriptionOrchestrator(self._config, output_file)
        self._orchestrator.start()

    def shutdown(self) -> None:
        if self._orchestrator:
            self._orchestrator.release()
            self._orchestrator = None
