import pytest
import tempfile
import os
import json
from app.utils.config_manager import ConfigManager
from app.models import TranscriptionConfig


class TestConfigManagerIntegration:
    @pytest.fixture
    def temp_config_file(self):
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.remove(path)
    
    def test_config_persistence_workflow(self, temp_config_file):
        manager = ConfigManager(temp_config_file)
        
        initial_config = {
            "model_size": "large-v3",
            "device": "cuda",
            "compute_type": "float16",
            "language": "en",
            "mic_id": 2,
            "vad_filter": True
        }
        
        manager.save_config_to_file(initial_config)
        
        new_manager = ConfigManager(temp_config_file)
        new_manager.load_config_from_file()
        loaded_config = new_manager.get_config_dict()
        
        assert loaded_config["model_size"] == "large-v3"
        assert loaded_config["device"] == "cuda"
        assert loaded_config["mic_id"] == 2
        assert loaded_config["vad_filter"] is True
    
    def test_config_update_workflow(self, temp_config_file):
        manager = ConfigManager(temp_config_file)
        
        initial = {"model_size": "tiny", "device": "cpu"}
        manager.save_config_to_file(initial)
        
        updated = {"model_size": "base", "device": "cuda", "language": "pt"}
        manager.save_config_to_file(updated)
        
        manager.load_config_from_file()
        loaded = manager.get_config_dict()
        
        assert loaded["model_size"] == "base"
        assert loaded["device"] == "cuda"
        assert loaded["language"] == "pt"
    
    def test_config_to_transcription_config(self, temp_config_file):
        manager = ConfigManager(temp_config_file)
        
        ui_config = {
            "model_size": "medium",
            "device": "cuda",
            "compute_type": "int8",
            "language": "en",
            "mic_id": 1,
            "vad_filter": True,
            "sample_rate": 16000,
            "chunk_duration": 5.0,
            "overlap_duration": 1.0,
            "no_speech_threshold": 0.6,
        }
        
        manager.save_config_to_file(ui_config)
        manager.load_config_from_file()
        loaded = manager.get_config_dict()
        
        transcription_config = TranscriptionConfig(
            model_size=loaded["model_size"],
            device=loaded["device"],
            compute_type=loaded["compute_type"],
            language=loaded["language"],
            mic_id=loaded["mic_id"],
            vad_filter=loaded["vad_filter"],
            sample_rate=loaded["sample_rate"],
            chunk_duration=loaded["chunk_duration"],
            overlap_duration=loaded["overlap_duration"],
            no_speech_threshold=loaded["no_speech_threshold"],
        )
        
        assert transcription_config.model_size == "medium"
        assert transcription_config.device == "cuda"
        assert transcription_config.compute_type == "int8"
        assert transcription_config.language == "en"
        assert transcription_config.mic_id == 1
        assert transcription_config.vad_filter is True
    
    def test_config_defaults_fallback(self, temp_config_file):
        manager = ConfigManager(temp_config_file)
        
        partial_config = {"model_size": "tiny"}
        manager.save_config_to_file(partial_config)
        
        manager.load_config_from_file()
        loaded = manager.get_config_dict()
        
        assert loaded["model_size"] == "tiny"
        assert loaded["device"] == ConfigManager.DEFAULT_CONFIG["device"]
        assert loaded["language"] == ConfigManager.DEFAULT_CONFIG["language"]
    
    def test_config_file_permissions(self, temp_config_file):
        manager = ConfigManager(temp_config_file)
        
        test_config = {"model_size": "base"}
        manager.save_config_to_file(test_config)
        
        assert os.path.exists(temp_config_file)
        assert os.access(temp_config_file, os.R_OK)
        assert os.access(temp_config_file, os.W_OK)
    
    def test_multiple_save_load_cycles(self, temp_config_file):
        manager = ConfigManager(temp_config_file)
        
        configs = [
            {"model_size": "tiny", "device": "cpu"},
            {"model_size": "base", "device": "cuda"},
            {"model_size": "medium", "device": "cpu"},
        ]
        
        for config in configs:
            manager.save_config_to_file(config)
            manager.load_config_from_file()
            loaded = manager.get_config_dict()
            assert loaded["model_size"] == config["model_size"]
            assert loaded["device"] == config["device"]

