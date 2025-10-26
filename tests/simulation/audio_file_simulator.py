import wave
import numpy as np
import threading
import logging
from typing import Optional
from pathlib import Path

import scipy

from app.audio.audio_capture import AudioBuffer


logger = logging.getLogger("tests.simulation.audio_file_simulator")


class AudioFileSimulator:
    def __init__(self, audio_file_path: str, sample_rate: int, realtime_speed: float = 1.0):
        self._audio_file_path = Path(audio_file_path)
        self._sample_rate = sample_rate
        self._realtime_speed = realtime_speed
        
        self._audio_data: Optional[np.ndarray] = None
        self._current_position = 0
        self._lock = threading.Lock()
        self._running = False
        self._feed_thread: Optional[threading.Thread] = None
        
        self._buffer = AudioBuffer(sample_rate)
        
        self._load_audio_file()
    
    def _load_audio_file(self) -> None:
        if not self._audio_file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {self._audio_file_path}")
        
        logger.info(f"Loading audio file: {self._audio_file_path}")
        
        with wave.open(str(self._audio_file_path), 'rb') as wf:
            file_sample_rate = wf.getframerate()
            n_channels = wf.getnchannels()
            n_frames = wf.getnframes()
            sample_width = wf.getsampwidth()
            
            audio_bytes = wf.readframes(n_frames)
            
            if sample_width == 2:
                audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif sample_width == 4:
                audio_data = np.frombuffer(audio_bytes, dtype=np.int32)
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")
            
            if n_channels == 2:
                audio_data = audio_data.reshape(-1, 2)
                audio_data = audio_data.mean(axis=1)
            
            if file_sample_rate != self._sample_rate:
                from scipy import signal
                num_samples = int(len(audio_data) * self._sample_rate / file_sample_rate)
                audio_data = signal.resample(audio_data, num_samples)
                audio_data = audio_data.astype(np.float32)
            
            self._audio_data = audio_data
            logger.info(f"Loaded {len(self._audio_data)} samples at {self._sample_rate}Hz")
    
    def start(self) -> None:
        if self._running:
            logger.warning("Simulator already running")
            return
                
        logger.info("Starting audio file simulator")
        self._running = True
        self._buffer.append(self._audio_data)
    
    def stop(self) -> None:
        logger.info("Stopping audio file simulator")
    
    def get_chunk(self, chunk_size: int, overlap_size: int) -> Optional[np.ndarray]:
        return self._buffer.extract_chunk(chunk_size, overlap_size)

    def has_data(self, minimum_samples: int) -> bool:
        with self._lock:
            return self._buffer.get_buffer_size() >= minimum_samples
    
    def get_buffer_size(self) -> int:
        return self._buffer.get_buffer_size()
    
    def is_finished(self) -> bool:
        with self._lock:
            return not self._running
        
    def resample_chunk_to_16k(self, chunk, sample_rate):
        target_rate = 16000
        if sample_rate == target_rate:
            return chunk
        num_samples = int(len(chunk) * target_rate / sample_rate)
        resampled_chunk = scipy.signal.resample(chunk, num_samples)
        return np.asarray(resampled_chunk, dtype=np.float32)