import logging
from typing import Optional, List, Dict, Any

import numpy as np
import scipy.signal
import sounddevice as sd
import threading

logger = logging.getLogger("app.audio.capture")


class AudioBuffer:
    def __init__(self, sample_rate: int, dtype: str = "float32", seconds: float = 10.0):
        """
        Fixed-size ring buffer.

        Args:
            sample_rate: samples per second (used to size buffer).
            dtype: numpy dtype string.
            seconds: capacity in seconds.
        """
        self._sample_rate = sample_rate
        self._dtype = dtype
        self._size = int(sample_rate * seconds)
        if self._size <= 0:
            raise ValueError("Buffer size must be > 0")
        self._buf = np.zeros(self._size, dtype=dtype)
        self._write = 0  # next write position
        self._available = 0  # number of valid samples currently in buffer
        self._lock = threading.Lock()

    def append(self, data: np.ndarray) -> None:
        """
        Append samples into ring buffer. Overwrites oldest samples if overflow.
        This method does only O(1) work aside from copying the incoming block.
        """
        if data.ndim == 2 and data.shape[1] == 1:
            data = data[:, 0]
        data = np.asarray(data, dtype=self._dtype)
        n = data.shape[0]
        if n == 0:
            return

        with self._lock:
            # If incoming larger than buffer, keep only the last _size samples
            if n >= self._size:
                # keep the tail
                tail = data[-self._size :]
                self._buf[:] = tail
                self._write = 0
                self._available = self._size
                return

            end = self._write + n
            if end <= self._size:
                self._buf[self._write : end] = data
            else:
                split = self._size - self._write
                self._buf[self._write :] = data[:split]
                self._buf[: n - split] = data[split:]
            self._write = (self._write + n) % self._size
            # update available, cap at buffer size
            self._available = min(self._size, self._available + n)

            # If we've filled past capacity (shouldn't happen due to min), ensure consistent state
            if self._available == self._size:
                # no extra action necessary; oldest overwritten implicitly

                # note: oldest index can be derived when reading:
                # oldest = (self._write - self._available) % self._size
                pass

    def extract_chunk(self, chunk_size: int, overlap_size: int) -> Optional[np.ndarray]:
        """
        Extract a contiguous chunk of chunk_size samples. Keep overlap_size samples
        from the end of the chunk in the buffer (for next extraction continuity).

        Returns None if not enough samples available.
        """
        if overlap_size < 0 or overlap_size > chunk_size:
            raise ValueError("overlap_size must be between 0 and chunk_size")

        with self._lock:
            if self._available < chunk_size:
                return None

            # oldest sample index
            oldest = (self._write - self._available) % self._size
            # end index (exclusive) for chunk
            chunk_end = (oldest + chunk_size) % self._size

            if oldest < chunk_end:
                chunk = self._buf[oldest:chunk_end].copy()
            else:
                # wrapped case
                chunk = np.concatenate(
                    (self._buf[oldest:], self._buf[:chunk_end])
                ).copy()

            # update available to preserve overlap_size at end of the chunk
            # new_available = (available - chunk_size + overlap_size)
            self._available = self._available - chunk_size + overlap_size
            if self._available < 0:
                self._available = 0

            # move the logical "oldest" forward to account for the chunk we consumed minus overlap
            # easiest representation: write pointer remains same; available updated,
            # so oldest = (self._write - self._available) % self._size for future reads
            return chunk

    def clear(self) -> None:
        with self._lock:
            self._buf[:] = 0
            self._write = 0
            self._available = 0

    def get_buffer_size(self) -> int:
        """Return number of available samples currently stored."""
        with self._lock:
            return int(self._available)

    def capacity(self) -> int:
        """Return total capacity in samples."""
        return self._size


class AudioCapture:
    def __init__(
        self,
        sample_rate: int,
        channels: int = 1,
        dtype: str = "float32",
        device_index: int = 0,
        buffer_seconds: float = 10.0,
    ):
        self._sample_rate = sample_rate
        self._channels = channels
        self._dtype = dtype
        self._device_index = device_index
        self._buffer = AudioBuffer(sample_rate, dtype, seconds=buffer_seconds)
        self._stream: Optional[sd.InputStream] = None
        self._running = False
        self._lock = threading.Lock()  # protects start/stop stream operations

    @staticmethod
    def get_input_devices() -> List[Dict[str, Any]]:
        devices = sd.query_devices()
        input_devices = []
        for idx, device in enumerate(devices):
            if device.get("max_input_channels", 0) > 0:
                input_devices.append(
                    {
                        "index": idx,
                        "name": device.get("name"),
                        "channels": device.get("max_input_channels"),
                    }
                )
        return input_devices

    @staticmethod
    def get_default_input_device_id() -> Any:
        # sounddevice default.device returns (input, output) tuple
        try:
            return sd.default.device[0]
        except Exception:
            return None

    def start(self) -> None:
        with self._lock:
            if self._running:
                logger.warning("Audio capture already running")
                return

            logger.info(
                f"Starting audio capture on device {self._device_index} at {self._sample_rate}Hz"
            )
            self._running = True

            def callback(indata, frames, time_info, status):
                if status:
                    logger.warning(f"Audio callback status: {status}")
                # quick path: append and return; no heavy ops, no allocations besides copying data into buffer
                if self._running:
                    # Note: sounddevice gives float32 by default when dtype='float32'
                    self._buffer.append(indata)

            self._stream = sd.InputStream(
                samplerate=self._sample_rate,
                channels=self._channels,
                dtype=self._dtype,
                callback=callback,
                device=self._device_index,
            )
            self._stream.start()
            logger.info("Audio capture started successfully")

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            logger.info("Stopping audio capture")
            self._running = False
            if self._stream:
                try:
                    self._stream.stop()
                except Exception as e:
                    logger.exception(f"Error stopping stream: {e}")
                try:
                    self._stream.close()
                except Exception as e:
                    logger.exception(f"Error closing stream: {e}")
                self._stream = None
            logger.info("Audio capture stopped")

    def get_chunk(self, chunk_size: int, overlap_size: int) -> Optional[np.ndarray]:
        return self._buffer.extract_chunk(chunk_size, overlap_size)

    def has_data(self, minimum_samples: int) -> bool:
        return self._buffer.get_buffer_size() >= minimum_samples

    def get_buffer_size(self) -> int:
        return self._buffer.get_buffer_size()

    def release(self) -> None:
        logger.info("Releasing audio capture resources")
        self.stop()
        self._buffer.clear()
        logger.info("Audio capture resources released")

    def resample_chunk_to_16k(self, chunk: np.ndarray, sample_rate: int) -> np.ndarray:
        target_rate = 16000
        if sample_rate == target_rate:
            return np.asarray(chunk, dtype=np.float32)
        num_samples = int(len(chunk) * target_rate / sample_rate)
        if num_samples <= 0:
            return np.zeros((0,), dtype=np.float32)
        resampled = scipy.signal.resample(chunk, num_samples)
        return np.asarray(resampled, dtype=np.float32)
