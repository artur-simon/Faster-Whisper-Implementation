import json
import os
import logging
from typing import Dict, Any

logger = logging.getLogger("app.utils.config_manager")


class ConfigManager:
    DEFAULT_CONFIG = {
        "model_size": "medium",
        "device": "cuda",
        "compute_type": "int8",
        "language": "pt",
        "mic_id": None,
        "vad_filter": True,
        "sample_rate": 44100,
        "chunk_duration": 5.0,
        "overlap_duration": 1.0,
        "no_speech_threshold": 0.6,
        "should_paste_content": False,
    }

    def __init__(self, config_path: str = "config.json"):
        self._config_path = config_path
        self._config_dict = None

    def load_config_from_file(self) -> Dict[str, Any]:
        if not os.path.exists(self._config_path):
            return self.DEFAULT_CONFIG.copy()

        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            result = self.DEFAULT_CONFIG.copy()
            result.update(config)
            self._config_dict = result
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Error loading config: {e}. Using defaults.")
            self._config_dict = self.DEFAULT_CONFIG.copy()

    def get_config_dict(self):
        return self._config_dict
        
    def save_config_to_file(self, config_dict: Dict[str, Any]) -> None:
        self._config_dict = config_dict
        try:
            with open(self._config_path, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
        except IOError as e:
            raise Exception(f"Error saving config: {e}")
