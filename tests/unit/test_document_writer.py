import pytest
import tempfile
import os
from unittest.mock import Mock, patch
from app.models import TranscriptionConfig, Word
from app.utils.document_writer import TranscriptionWriter


class TestTranscriptionWriter:
    @pytest.fixture
    def temp_file(self):
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
    
    def test_writer_initialization(self, temp_file, config):
        writer = TranscriptionWriter(temp_file, config)
        assert writer._output_path == temp_file
        assert writer._encoding == "utf-8"
    
    def test_write_words_empty_list(self, temp_file, config):
        writer = TranscriptionWriter(temp_file, config)
        writer.write_words([])
        
        with open(temp_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        assert content == ""
    
    def test_write_words_single_word(self, temp_file, config):
        writer = TranscriptionWriter(temp_file, config)
        words = [Word(text=" hello", start=0.0, end=1.0, probability=0.9)]
        writer.write_words(words)
        
        with open(temp_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        assert "hello" in content
    
    def test_write_words_multiple_words(self, temp_file, config):
        writer = TranscriptionWriter(temp_file, config)
        words = [
            Word(text=" hello", start=0.0, end=1.0, probability=0.9),
            Word(text=" world", start=1.0, end=2.0, probability=0.85)
        ]
        writer.write_words(words)
        
        with open(temp_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        assert "hello world" in content
    
    def test_write_string_to_file(self, temp_file, config):
        writer = TranscriptionWriter(temp_file, config)
        writer.write_string_to_file("test content")
        
        with open(temp_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        assert content == "test content"
    
    def test_write_string_to_file_append(self, temp_file, config):
        writer = TranscriptionWriter(temp_file, config)
        writer.write_string_to_file("first ")
        writer.write_string_to_file("second")
        
        with open(temp_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        assert content == "first second"
    
    @patch('app.utils.document_writer.paste_content')
    def test_write_string_with_paste_enabled(self, mock_paste, temp_file):
        config = TranscriptionConfig(
            model_size="tiny",
            device="cpu",
            compute_type="int8",
            language="en",
            sample_rate=16000,
            chunk_duration=5.0,
            overlap_duration=1.0,
            no_speech_threshold=0.6,
            should_paste_content=True
        )
        
        writer = TranscriptionWriter(temp_file, config)
        writer.write_string("test")
        
        mock_paste.assert_called_once_with("test")
    
    @patch('app.utils.document_writer.paste_content')
    def test_write_string_with_paste_disabled(self, mock_paste, temp_file, config):
        writer = TranscriptionWriter(temp_file, config)
        writer.write_string("test")
        
        mock_paste.assert_not_called()
    
    def test_format_words(self, temp_file, config):
        writer = TranscriptionWriter(temp_file, config)
        words = [
            Word(text=" hello", start=0.0, end=1.0, probability=0.9),
            Word(text=" world", start=1.0, end=2.0, probability=0.85)
        ]
        
        result = writer._format_words(words)
        assert result == "hello world"
    
    def test_write_words_strips_whitespace(self, temp_file, config):
        writer = TranscriptionWriter(temp_file, config)
        words = [
            Word(text="  hello  ", start=0.0, end=1.0, probability=0.9),
        ]
        writer.write_words(words)
        
        with open(temp_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        assert content == "hello "


