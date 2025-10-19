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
    }

    def __init__(self, config_path: str = "config.json"):
        self._config_path = config_path

    def load_config(self) -> Dict[str, Any]:
        if not os.path.exists(self._config_path):
            return self.DEFAULT_CONFIG.copy()

        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            result = self.DEFAULT_CONFIG.copy()
            result.update(config)
            return result
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Error loading config: {e}. Using defaults.")
            return self.DEFAULT_CONFIG.copy()

    def save_config(self, config_dict: Dict[str, Any]) -> None:
        try:
            with open(self._config_path, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
        except IOError as e:
            raise Exception(f"Error saving config: {e}")
