import pytest
import numpy as np
from app.audio.audio_capture import AudioBuffer, AudioCapture


class TestAudioBuffer:
    def test_buffer_initialization(self):
        buffer = AudioBuffer(sample_rate=16000)
        assert buffer.get_buffer_size() == 0
    
    def test_append_data(self):
        buffer = AudioBuffer(sample_rate=16000)
        data = np.zeros((100, 1), dtype="float32")
        buffer.append(data)
        assert buffer.get_buffer_size() == 100
    
    def test_extract_chunk_success(self):
        buffer = AudioBuffer(sample_rate=16000)
        data = np.ones((1000, 1), dtype="float32")
        buffer.append(data)
        
        chunk = buffer.extract_chunk(chunk_size=500, overlap_size=100)
        assert chunk is not None
        assert len(chunk) == 500
        assert buffer.get_buffer_size() == 600
    
    def test_extract_chunk_insufficient_data(self):
        buffer = AudioBuffer(sample_rate=16000)
        data = np.ones((100, 1), dtype="float32")
        buffer.append(data)
        
        chunk = buffer.extract_chunk(chunk_size=500, overlap_size=100)
        assert chunk is None
        assert buffer.get_buffer_size() == 100
    
    def test_clear_buffer(self):
        buffer = AudioBuffer(sample_rate=16000)
        data = np.ones((100, 1), dtype="float32")
        buffer.append(data)
        
        buffer.clear()
        assert buffer.get_buffer_size() == 0


class TestAudioCapture:
    def test_get_input_devices(self):
        devices = AudioCapture.get_input_devices()
        assert isinstance(devices, list)
        for device in devices:
            assert 'index' in device
            assert 'name' in device
            assert 'channels' in device
            assert device['channels'] > 0
    
    def test_audio_capture_initialization(self):
        capture = AudioCapture(sample_rate=16000)
        assert capture._sample_rate == 16000
        assert capture._channels == 1
        assert capture._dtype == "float32"
        assert capture._device_index is 0
        assert not capture._running
    
    def test_audio_capture_with_device_index(self):
        capture = AudioCapture(sample_rate=16000, device_index=0)
        assert capture._device_index == 0
    
    def test_has_data(self):
        capture = AudioCapture(sample_rate=16000)
        assert not capture.has_data(minimum_samples=100)

