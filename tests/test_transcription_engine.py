import pytest
from unittest.mock import Mock, patch
from app.transcription.transcription_engine import TranscriptionEngine
from app.models import TranscriptionConfig, Word, TranscriptionSegment


class TestTranscriptionEngine:
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
        )
    
    @pytest.fixture
    def mock_whisper_model(self):
        with patch("app.transcription.transcription_engine.WhisperModel") as mock:
            yield mock
    
    def test_engine_initialization(self, config, mock_whisper_model):
        engine = TranscriptionEngine(config)
        
        mock_whisper_model.assert_called_once_with(
            model_size_or_path="tiny",
            device="cpu",
            compute_type="int8"
        )
    
    def test_transcribe_file_no_context(self, config, mock_whisper_model):
        mock_word = Mock()
        mock_word.word = " test"
        mock_word.start = 0.0
        mock_word.end = 1.0
        mock_word.probability = 0.95
        
        mock_segment = Mock()
        mock_segment.words = [mock_word]
        mock_segment.no_speech_prob = 0.1
        
        mock_model_instance = Mock()
        mock_model_instance.transcribe.return_value = ([mock_segment], None)
        mock_whisper_model.return_value = mock_model_instance
        
        engine = TranscriptionEngine(config)
        segments = list(engine.transcribe_audio("test.wav"))
        
        assert len(segments) == 1
        assert len(segments[0].words) == 1
        assert segments[0].words[0].text == " test"
        assert segments[0].words[0].probability == 0.95
        assert segments[0].no_speech_probability == 0.1
    
    def test_transcribe_file_with_context(self, config, mock_whisper_model):
        mock_model_instance = Mock()
        mock_model_instance.transcribe.return_value = ([], None)
        mock_whisper_model.return_value = mock_model_instance
        
        engine = TranscriptionEngine(config)
        context_words = [
            Word(text=" hello", start=0.0, end=1.0, probability=0.9)
        ]
        
        list(engine.transcribe_audio("test.wav", context_words=context_words))
        
        call_kwargs = mock_model_instance.transcribe.call_args[1]
        assert 'initial_prompt' in call_kwargs
        assert call_kwargs['initial_prompt'] == "hello"
        assert call_kwargs['condition_on_previous_text'] is True
    
    def test_release(self, config, mock_whisper_model):
        mock_model_instance = Mock()
        mock_whisper_model.return_value = mock_model_instance
        
        engine = TranscriptionEngine(config)
        engine.release()
        
        assert engine._model is None

