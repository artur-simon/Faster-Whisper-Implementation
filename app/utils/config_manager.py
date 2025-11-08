import json
import os
import logging
from typing import Dict, Any
from copy import deepcopy

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
        "use_previous_context": True,
        "transcription_algorithm": "simple_overlap_resolve",
        "local_agreement_config": {
            "agreement_count": 2,
            "edit_threshold": 0.2,
            "confidence_threshold": 0.8,
            "min_words": 3,
            "context_size": 100
        },
        "vad_params": {
            "threshold": 0.5,
            "min_speech_duration_ms": 400,
            "min_silence_duration_ms": 400,
        }
    }

    def __init__(self, config_path: str = "config.json"):
        self._config_path = config_path
        self._config_dict = None

    @staticmethod
    def _deep_merge(base: Dict, override: Dict) -> Dict:
        result = deepcopy(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigManager._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def load_config_from_file(self, config_path=None):
        path = config_path or self._config_path
        if not os.path.exists(path):
            self._config_dict = deepcopy(self.DEFAULT_CONFIG)
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                config = json.load(f)
            self._config_dict = self._deep_merge(self.DEFAULT_CONFIG, config)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Error loading config: {e}. Using defaults.")
            self._config_dict = deepcopy(self.DEFAULT_CONFIG)


    def get_config_dict(self):
        return self._config_dict
        
    def save_config_to_file(self, config_dict: Dict[str, Any]) -> None:
        self._config_dict = config_dict
        try:
            with open(self._config_path, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
        except IOError as e:
            raise Exception(f"Error saving config: {e}")
