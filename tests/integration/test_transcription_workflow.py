import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch
from app.models import TranscriptionConfig, Word
from app.transcription.orchestrator import TranscriptionOrchestrator
from app.transcription.transcription_engine import TranscriptionEngine
from app.audio.audio_capture import AudioCapture


class TestTranscriptionWorkflow:
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
    
    @pytest.fixture
    def temp_output_file(self):
        fd, path = tempfile.mkstemp(suffix=".txt")
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.remove(path)
    
    @patch('app.transcription.transcription_engine.WhisperModel')
    def test_orchestrator_initialization(self, mock_whisper, config, temp_output_file):
        orchestrator = TranscriptionOrchestrator(config, temp_output_file)
        
        assert orchestrator._config == config
        assert orchestrator._output_path == temp_output_file
        assert not orchestrator._running
    
    @patch('app.transcription.transcription_engine.WhisperModel')
    def test_orchestrator_state_initialization(self, mock_whisper, config, temp_output_file):
        orchestrator = TranscriptionOrchestrator(config, temp_output_file)
        
        assert orchestrator._state.timestamp_offset == 0.0
        assert orchestrator._state.previous_overlap_words == []
        assert orchestrator._state.should_break_line is False
    
    @patch('app.transcription.transcription_engine.WhisperModel')
    @patch('app.audio.audio_capture.sd.InputStream')
    def test_orchestrator_start_stop(self, mock_stream, mock_whisper, config, temp_output_file):
        orchestrator = TranscriptionOrchestrator(config, temp_output_file)
        
        orchestrator.start()
        assert orchestrator._running
        assert orchestrator._thread is not None
        
        orchestrator.stop()
        assert not orchestrator._running
    
    @patch('app.transcription.transcription_engine.WhisperModel')
    def test_extract_words_from_segments(self, mock_whisper, config, temp_output_file):
        orchestrator = TranscriptionOrchestrator(config, temp_output_file)
        
        mock_segment1 = Mock()
        mock_segment1.no_speech_prob = 0.1
        mock_segment1.words = [
            Word(text=" hello", start=0.0, end=1.0, probability=0.9)
        ]
        
        mock_segment2 = Mock()
        mock_segment2.no_speech_prob = 0.8
        mock_segment2.words = [
            Word(text=" silence", start=1.0, end=2.0, probability=0.3)
        ]
        
        words = orchestrator._extract_words_from_segments([mock_segment1, mock_segment2])
        
        assert len(words) == 1
        assert words[0].text == " hello"
    
    @patch('app.transcription.transcription_engine.WhisperModel')
    def test_orchestrator_handles_no_speech(self, mock_whisper, config, temp_output_file):
        orchestrator = TranscriptionOrchestrator(config, temp_output_file)
        
        mock_segment = Mock()
        mock_segment.no_speech_prob = 0.9
        mock_segment.words = []
        
        orchestrator._handle_transcription_result([mock_segment])
        
        assert orchestrator._state.timestamp_offset == 0.0
    
    @patch('app.transcription.transcription_engine.WhisperModel')
    @patch('app.audio.audio_capture.sd.InputStream')
    def test_orchestrator_release(self, mock_stream, mock_whisper, config, temp_output_file):
        orchestrator = TranscriptionOrchestrator(config, temp_output_file)
        
        orchestrator.start()
        orchestrator.release()
        
        assert not orchestrator._running
        assert orchestrator._engine is None


class TestAudioCaptureIntegration:
    def test_audio_buffer_workflow(self):
        from app.audio.audio_capture import AudioBuffer
        
        buffer = AudioBuffer(sample_rate=16000)
        
        for i in range(10):
            data = np.random.randn(1000, 1).astype("float32")
            buffer.append(data)
        
        assert buffer.get_buffer_size() == 10000
        
        chunks = []
        while buffer.get_buffer_size() >= 5000:
            chunk = buffer.extract_chunk(chunk_size=5000, overlap_size=1000)
            if chunk is not None:
                chunks.append(chunk)
        
        assert len(chunks) == 2
        assert len(chunks[0]) == 5000
    
    @patch('app.audio.audio_capture.sd.InputStream')
    def test_audio_capture_lifecycle(self, mock_stream):
        capture = AudioCapture(sample_rate=16000, device_index=0)
        
        assert not capture._running
        
        capture.start()
        assert capture._running
        
        capture.stop()
        assert not capture._running
        
        capture.release()


class TestEndToEndTranscription:
    @pytest.fixture
    def temp_output_file(self):
        fd, path = tempfile.mkstemp(suffix=".txt")
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.remove(path)
    
    @patch('app.transcription.transcription_engine.WhisperModel')
    def test_transcription_with_context_words(self, mock_whisper, temp_output_file):
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
        mock_word = Mock()
        mock_word.word = " test"
        mock_word.start = 0.0
        mock_word.end = 1.0
        mock_word.probability = 0.95
        
        mock_segment = Mock()
        mock_segment.words = [mock_word]
        mock_segment.text = " test"
        mock_segment.no_speech_prob = 0.1
        
        mock_model_instance.transcribe.return_value = ([mock_segment], None)
        mock_whisper.return_value = mock_model_instance
        
        engine = TranscriptionEngine(config)
        
        context = [Word(text=" previous", start=0.0, end=1.0, probability=0.9)]
        audio_data = np.random.randn(16000).astype("float32")
        
        segments = list(engine.transcribe_audio(audio_data, context_words=context))
        
        assert len(segments) > 0
        call_kwargs = mock_model_instance.transcribe.call_args[1]
        assert 'initial_prompt' in call_kwargs
    
    @patch('app.transcription.transcription_engine.WhisperModel')
    def test_orchestrator_writes_to_file(self, mock_whisper, temp_output_file):
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
        mock_word = Mock()
        mock_word.text = " test"
        mock_word.start = 0.0
        mock_word.end = 0.5
        mock_word.probability = 0.95
        
        mock_segment = Mock()
        mock_segment.words = [mock_word]
        mock_segment.text = " test"
        mock_segment.no_speech_prob = 0.1
        
        mock_model_instance.transcribe.return_value = ([mock_segment], None)
        mock_whisper.return_value = mock_model_instance
        
        orchestrator = TranscriptionOrchestrator(config, temp_output_file)
        
        audio_chunk = np.random.randn(16000 * 2).astype("float32")
        orchestrator._handle_transcription_result([mock_segment])
        
        assert os.path.exists(temp_output_file)


