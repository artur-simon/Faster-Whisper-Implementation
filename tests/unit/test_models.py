import pytest
from dataclasses import FrozenInstanceError
from app.models import Word, TranscriptionSegment, TranscriptionConfig


class TestWord:
    def test_word_creation(self):
        word = Word(text=" hello", start=0.0, end=1.0, probability=0.9)
        
        assert word.text == " hello"
        assert word.start == 0.0
        assert word.end == 1.0
        assert word.probability == 0.9
    
    def test_word_is_dataclass(self):
        word = Word(text=" hello", start=0.0, end=1.0, probability=0.9)
        assert hasattr(word, '__dataclass_fields__')
    
    def test_word_equality(self):
        word1 = Word(text=" hello", start=0.0, end=1.0, probability=0.9)
        word2 = Word(text=" hello", start=0.0, end=1.0, probability=0.9)
        
        assert word1 == word2
    
    def test_word_inequality(self):
        word1 = Word(text=" hello", start=0.0, end=1.0, probability=0.9)
        word2 = Word(text=" world", start=0.0, end=1.0, probability=0.9)
        
        assert word1 != word2


class TestTranscriptionSegment:
    def test_segment_creation_with_words(self):
        words = [
            Word(text=" hello", start=0.0, end=1.0, probability=0.9),
            Word(text=" world", start=1.0, end=2.0, probability=0.85)
        ]
        segment = TranscriptionSegment(
            text=" hello world",
            words=words,
            no_speech_prob=0.1
        )
        
        assert segment.text == " hello world"
        assert len(segment.words) == 2
        assert segment.no_speech_prob == 0.1
    
    def test_segment_creation_without_words(self):
        segment = TranscriptionSegment(
            text=" hello world",
            words=[],
            no_speech_prob=0.1
        )
        
        assert segment.text == " hello world"
        assert len(segment.words) == 0
    
    def test_segment_equality(self):
        words = [Word(text=" hello", start=0.0, end=1.0, probability=0.9)]
        segment1 = TranscriptionSegment(text=" hello", words=words, no_speech_prob=0.1)
        segment2 = TranscriptionSegment(text=" hello", words=words, no_speech_prob=0.1)
        
        assert segment1 == segment2


class TestTranscriptionConfig:
    def test_config_creation_with_custom_values(self):
        config = TranscriptionConfig(
            model_size="tiny",
            device="cpu",
            compute_type="int8",
            language="en",
            sample_rate=16000,
            chunk_duration=10.0,
            overlap_duration=2.0,
            no_speech_threshold=0.5,
            vad_filter=True,
            mic_id=5,
            should_paste_content=True
        )
        
        assert config.model_size == "tiny"
        assert config.device == "cpu"
        assert config.compute_type == "int8"
        assert config.language == "en"
        assert config.sample_rate == 16000
        assert config.chunk_duration == 10.0
        assert config.overlap_duration == 2.0
        assert config.no_speech_threshold == 0.5
        assert config.vad_filter is True
        assert config.mic_id == 5
        assert config.should_paste_content is True
    
    def test_config_is_dataclass(self):
        config = TranscriptionConfig(
            model_size="tiny",
            device="cpu",
            compute_type="int8",
            language="en",
            sample_rate=16000,
            chunk_duration=10.0,
            overlap_duration=2.0,
            no_speech_threshold=0.5,
            vad_filter=True,
            mic_id=5,
            should_paste_content=True
        )
        assert hasattr(config, '__dataclass_fields__')
    
    def test_config_mutable(self):
        config = TranscriptionConfig(
            model_size="tiny",
            device="cpu",
            compute_type="int8",
            language="en",
            sample_rate=16000,
            chunk_duration=10.0,
            overlap_duration=2.0,
            no_speech_threshold=0.5,
            vad_filter=True,
            mic_id=5,
            should_paste_content=True
        )
        config.model_size = "base"
        
        assert config.model_size == "base"


