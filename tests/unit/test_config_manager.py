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

    def test_save_and_load_config(self, temp_config_file):
        manager = ConfigManager(temp_config_file)
        test_config = {
            "model_size": "large-v3",
            "device": "cpu",
            "compute_type": "float32",
            "language": "en",
            "mic_id": 5,
            "vad_filter": True,
            "sample_rate": 44100,
            "chunk_duration": 5.0,
            "overlap_duration": 1.0,
            "no_speech_threshold": 0.6,
            "should_paste_content": False,
            "use_previous_context": False
        }

        manager.save_config_to_file(test_config)
        manager.load_config_from_file()

        assert manager.get_config_dict() == test_config

    def test_load_config_missing_file_sould_return_default(self, temp_config_file):
        os.remove(temp_config_file)
        manager = ConfigManager(temp_config_file)
        manager.load_config_from_file()

        assert manager.get_config_dict() == ConfigManager.DEFAULT_CONFIG

    def test_load_config_with_invalid_json_should_return_default(self, temp_config_file):
        with open(temp_config_file, "w") as f:
            f.write("invalid json {{{")

        manager = ConfigManager(temp_config_file)
        manager.load_config_from_file()

        assert manager.get_config_dict() == ConfigManager.DEFAULT_CONFIG

    def test_load_config_provided_path_missing_should_return_default(self, temp_config_file):
        os.remove(temp_config_file)
        manager = ConfigManager("nonexistent.json")
        manager.load_config_from_file(temp_config_file)
        assert manager.get_config_dict() == ConfigManager.DEFAULT_CONFIG
        

    def test_save_config_partial_update(self, temp_config_file):
        manager = ConfigManager(temp_config_file)
        partial_config = {
            "model_size": "tiny",
            "device": "cuda",
        }

        manager.save_config_to_file(partial_config)
        manager.load_config_from_file()
        loaded_config = manager.get_config_dict()

        assert loaded_config["model_size"] == "tiny"
        assert loaded_config["device"] == "cuda"
        assert "language" in loaded_config

    def test_default_config_values(self):
        assert ConfigManager.DEFAULT_CONFIG["model_size"] == "medium"
        assert ConfigManager.DEFAULT_CONFIG["device"] == "cuda"
        assert ConfigManager.DEFAULT_CONFIG["compute_type"] == "int8"
        assert ConfigManager.DEFAULT_CONFIG["language"] == "pt"
        assert ConfigManager.DEFAULT_CONFIG["mic_id"] is None
