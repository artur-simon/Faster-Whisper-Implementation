import threading
import logging
import scipy.signal
import numpy as np
from typing import Optional, List, Callable
from app.audio.audio_capture import AudioCapture

logger = logging.getLogger("app.audio.data_provider")


class AudioDataProvider:

    def __init__(self, sample_rate: int, device_index: int = 0):
        self._audio_capture = AudioCapture(sample_rate, device_index=device_index)
        self._sample_rate = sample_rate
        self._device_index = device_index
        self._listeners: List[Callable[[], None]] = []
        self._lock = threading.Lock()

    def start(self) -> None:
        self._audio_capture.start()

    def stop(self) -> None:
        self._audio_capture.stop()

    def release(self) -> None:
        self._audio_capture.release()

    def get_audio_chunk(self, chunk_size: int = 1024) -> Optional['np.ndarray']:
        return self._audio_capture.get_chunk(chunk_size, overlap_size=0)

    def get_transcription_chunk(self, chunk_size: int, overlap_size: int) -> Optional['np.ndarray']:
        return self._audio_capture.get_chunk(chunk_size, overlap_size)

    def has_data(self, minimum_samples: int = 512) -> bool:
        return self._audio_capture.has_data(minimum_samples)

    def get_sample_rate(self) -> int:
        return self._sample_rate

    def get_buffer_size(self) -> int:
        return self._audio_capture.get_buffer_size()

    def resample_chunk_to_16k(self, chunk, sample_rate):
        return self._audio_capture.resample_chunk_to_16k(chunk, sample_rate)
