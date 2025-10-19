import pytest
import os
import tempfile
from app.utils.config_manager import ConfigManager


class TestConfigManager:
    @pytest.fixture
    def temp_config_file(self):
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.remove(path)
    
    def test_load_config_missing_file(self, temp_config_file):
        os.remove(temp_config_file)
        manager = ConfigManager(temp_config_file)
        config = manager.load_config()
        
        assert config == ConfigManager.DEFAULT_CONFIG
    
    def test_save_and_load_config(self, temp_config_file):
        manager = ConfigManager(temp_config_file)
        test_config = {
            "model_size": "large-v3",
            "device": "cpu",
            "compute_type": "float32",
            "language": "en",
            "mic_id": 5,
        }
        
        manager.save_config(test_config)
        loaded_config = manager.load_config()
        
        assert loaded_config == test_config
    
    def test_load_config_with_invalid_json(self, temp_config_file):
        with open(temp_config_file, "w") as f:
            f.write("invalid json {{{")
        
        manager = ConfigManager(temp_config_file)
        config = manager.load_config()
        
        assert config == ConfigManager.DEFAULT_CONFIG
    
    def test_save_config_partial_update(self, temp_config_file):
        manager = ConfigManager(temp_config_file)
        partial_config = {
            "model_size": "tiny",
            "device": "cuda",
        }
        
        manager.save_config(partial_config)
        loaded_config = manager.load_config()
        
        assert loaded_config["model_size"] == "tiny"
        assert loaded_config["device"] == "cuda"
        assert "language" in loaded_config
    
    def test_default_config_values(self):
        assert ConfigManager.DEFAULT_CONFIG["model_size"] == "medium"
        assert ConfigManager.DEFAULT_CONFIG["device"] == "cuda"
        assert ConfigManager.DEFAULT_CONFIG["compute_type"] == "int8"
        assert ConfigManager.DEFAULT_CONFIG["language"] == "pt"
        assert ConfigManager.DEFAULT_CONFIG["mic_id"] is None

