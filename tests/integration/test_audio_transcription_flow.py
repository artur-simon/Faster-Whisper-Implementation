import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch
from app.models import TranscriptionConfig, Word
from app.transcription.transcription_controller import TranscriptionController
from app.audio.audio_capture import AudioBuffer


class TestAudioTranscriptionFlow:
    @pytest.fixture
    def temp_output_file(self):
        fd, path = tempfile.mkstemp(suffix=".txt")
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.remove(path)
    
    @pytest.fixture
    def config(self):
        return TranscriptionConfig(
            model_size="tiny",
            device="cpu",
            compute_type="int8",
            language="en",
            sample_rate=16000,
            chunk_duration=5.0,
            overlap_duration=1.0,
            no_speech_threshold=0.6,
            should_paste_content=False
        )
        
    @patch('app.transcription.transcription_engine.WhisperModel')
    def test_audio_file_transcription_flow(self, mock_whisper, temp_output_file, config):
        mock_model_instance = Mock()
        
        mock_word1 = Mock()
        mock_word1.word = " Hello"
        mock_word1.start = 0.0
        mock_word1.end = 0.5
        mock_word1.probability = 0.95
        
        mock_word2 = Mock()
        mock_word2.word = " world"
        mock_word2.start = 0.5
        mock_word2.end = 1.0
        mock_word2.probability = 0.93
        
        mock_segment = Mock()
        mock_segment.words = [mock_word1, mock_word2]
        mock_segment.text = " Hello world"
        mock_segment.no_speech_prob = 0.05
        
        mock_model_instance.transcribe.return_value = ([mock_segment], None)
        mock_whisper.return_value = mock_model_instance
        
        controller = TranscriptionController(config)
        
        temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_audio.close()
        
        try:
            controller.transcribe_audio_file(temp_audio.name, temp_output_file)
            
            assert os.path.exists(temp_output_file)
            with open(temp_output_file, "r", encoding="utf-8") as f:
                content = f.read()
                assert "Hello world" in content
        finally:
            if os.path.exists(temp_audio.name):
                os.remove(temp_audio.name)
    
    def test_audio_buffer_chunking_flow(self):
        buffer = AudioBuffer(sample_rate=16000)
        
        chunk_size = 16000 * 5
        overlap_size = 16000 * 1
        
        for i in range(10):
            data = np.random.randn(16000, 1).astype("float32")
            buffer.append(data)
        
        chunks_extracted = []
        while buffer.get_buffer_size() >= chunk_size:
            chunk = buffer.extract_chunk(chunk_size, overlap_size)
            if chunk is not None:
                chunks_extracted.append(chunk)
                assert len(chunk) == chunk_size
        
        assert len(chunks_extracted) > 0
    
    @patch('app.transcription.transcription_engine.WhisperModel')
    def test_controller_config_update_flow(self, mock_whisper, config):
        controller = TranscriptionController(config)
        
        assert controller._config.language == "en"
        assert controller._config.vad_filter is True
        
        controller.update_input_config(language="pt", vad_filter=False)
        
        assert controller._config.language == "pt"
        assert controller._config.vad_filter is False
    
    @patch('app.transcription.transcription_engine.WhisperModel')
    def test_controller_shutdown_cleanup(self, mock_whisper, config):
        controller = TranscriptionController(config)
        
        controller.shutdown()
        
        assert controller._orchestrator is None
        assert controller._engine is None


class TestRealTimeTranscriptionSimulation:
    @pytest.fixture
    def temp_output_file(self):
        fd, path = tempfile.mkstemp(suffix=".txt")
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.remove(path)
    
    @patch('app.transcription.transcription_engine.WhisperModel')
    @patch('app.audio.audio_capture.sd.InputStream')
    def test_simulated_real_time_transcription(self, mock_stream, mock_whisper, temp_output_file):
        from app.transcription.orchestrator import TranscriptionOrchestrator
        
        config = TranscriptionConfig(
            model_size="tiny",
            device="cpu",
            compute_type="int8",
            language="en",
            sample_rate=16000,
            chunk_duration=2.0,
            overlap_duration=0.5,
            no_speech_threshold=0.6,
            should_paste_content=False
        )
        
        mock_model_instance = Mock()
        mock_whisper.return_value = mock_model_instance
        
        from app.audio.audio_data_provider import AudioDataProvider

        orchestrator = TranscriptionOrchestrator(
            config, temp_output_file, AudioDataProvider(sample_rate=config.sample_rate)
        )
        
        for i in range(3):
            mock_word = Mock()
            mock_word.text = f" word{i}"
            mock_word.start = float(i)
            mock_word.end = float(i + 0.5)
            mock_word.probability = 0.9
            
            mock_segment = Mock()
            mock_segment.words = [mock_word]
            mock_segment.text = f" word{i}"
            mock_segment.no_speech_prob = 0.1
            
            mock_model_instance.transcribe.return_value = ([mock_segment], None)
            
            audio_chunk = np.random.randn(16000 * 2).astype("float32")
            orchestrator._handle_transcription_result([mock_segment])
    

class TestEdgeCases:
    @pytest.fixture
    def temp_output_file(self):
        fd, path = tempfile.mkstemp(suffix=".txt")
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.remove(path)
    
    @pytest.fixture
    def config(self):
        return TranscriptionConfig(
            model_size="tiny",
            device="cpu",
            compute_type="int8",
            language="en",
            sample_rate=16000,
            chunk_duration=5.0,
            overlap_duration=1.0,
            no_speech_threshold=0.6,
            should_paste_content=False
        )
        
    @patch('app.transcription.transcription_engine.WhisperModel')
    def test_empty_audio_transcription(self, mock_whisper, temp_output_file, config):
        mock_model_instance = Mock()
        mock_model_instance.transcribe.return_value = ([], None)
        mock_whisper.return_value = mock_model_instance
        
        controller = TranscriptionController(config)
        
        temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_audio.close()
        
        try:
            controller.transcribe_audio_file(temp_audio.name, temp_output_file)
            
            assert os.path.exists(temp_output_file)
        finally:
            if os.path.exists(temp_audio.name):
                os.remove(temp_audio.name)
    
    @patch('app.transcription.transcription_engine.WhisperModel')
    def test_high_no_speech_probability(self, mock_whisper, temp_output_file, config):
        mock_model_instance = Mock()
        
        mock_segment = Mock()
        mock_segment.words = []
        mock_segment.text = ""
        mock_segment.no_speech_prob = 0.95
        
        mock_model_instance.transcribe.return_value = ([mock_segment], None)
        mock_whisper.return_value = mock_model_instance
        
        controller = TranscriptionController(config)
        
        temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_audio.close()
        
        try:
            controller.transcribe_audio_file(temp_audio.name, temp_output_file)
            
            with open(temp_output_file, "r", encoding="utf-8") as f:
                content = f.read()
                assert content == ""
        finally:
            if os.path.exists(temp_audio.name):
                os.remove(temp_audio.name)
    
    def test_buffer_overflow_handling(self):
        buffer = AudioBuffer(sample_rate=16000)
        
        large_data = np.random.randn(16000 * 100, 1).astype("float32")
        buffer.append(large_data)
        
        assert buffer.get_buffer_size() == buffer._size
        
        chunk = buffer.extract_chunk(chunk_size=16000 * 10, overlap_size=16000 * 2)
        assert chunk is not None
        assert len(chunk) == 16000 * 10


