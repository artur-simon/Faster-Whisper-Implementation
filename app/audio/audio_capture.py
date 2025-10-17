import sounddevice as sd
import numpy as np
import threading
from typing import Optional, List, Dict


class AudioBuffer:
    def __init__(self, sample_rate: int, dtype: str = "float32"):
        self._buffer: np.ndarray = np.zeros((0, 1), dtype=dtype)
        self._lock = threading.Lock()
        self._sample_rate = sample_rate

    def append(self, data: np.ndarray) -> None:
        with self._lock:
            self._buffer = np.concatenate((self._buffer, data.copy()))

    def extract_chunk(self, chunk_size: int, overlap_size: int) -> Optional[np.ndarray]:
        with self._lock:
            if len(self._buffer) >= chunk_size:
                chunk = self._buffer[:chunk_size].copy()
                self._buffer = self._buffer[chunk_size - overlap_size :]
                return chunk
            return None

    def clear(self) -> None:
        with self._lock:
            self._buffer = np.zeros((0, 1), dtype=self._buffer.dtype)

    def get_length(self) -> int:
        with self._lock:
            return len(self._buffer)


class AudioCapture:
    def __init__(
        self,
        sample_rate: int,
        channels: int = 1,
        dtype: str = "float32",
        device_index: int = 0,
    ):
        self._sample_rate = sample_rate
        self._channels = channels
        self._dtype = dtype
        self._device_index = device_index
        self._buffer = AudioBuffer(sample_rate, dtype)
        self._stream: Optional[sd.InputStream] = None
        self._running = False

    @staticmethod
    def get_input_devices() -> List[Dict[str, any]]:
        devices = sd.query_devices()
        input_devices = []
        for idx, device in enumerate(devices):
            if device["max_input_channels"] > 0:
                input_devices.append(
                    {
                        "index": idx,
                        "name": device["name"],
                        "channels": device["max_input_channels"],
                    }
                )
        return input_devices

    @staticmethod
    def get_default_input_device_id() -> any:
        return sd.default.device[0]

    def start(self) -> None:
        if self._running:
            return

        self._running = True

        def callback(indata, frames, time_info, status):
            if self._running:
                self._buffer.append(indata)

        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype=self._dtype,
            callback=callback,
            device=self._device_index,
        )
        self._stream.start()

    def stop(self) -> None:
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def get_chunk(self, chunk_size: int, overlap_size: int) -> Optional[np.ndarray]:
        return self._buffer.extract_chunk(chunk_size, overlap_size)

    def has_data(self, minimum_samples: int) -> bool:
        return self._buffer.get_length() >= minimum_samples

    def release(self) -> None:
        self.stop()
        self._buffer.clear()
