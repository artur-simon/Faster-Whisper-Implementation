import json
import os
import logging
from typing import List, Optional

logger = logging.getLogger("app.utils.state_manager")


class AppStateManager:
    DEFAULT_STATE = {
        "current_transcription_file": "transcription.txt",
        "recent_files": [],
        "recent_projects": [],
        "last_save_directory": "",
        "window_geometry": None,
        "last_audio_directory": "",
        "dark_mode": False,
    }

    MAX_RECENT_FILES = 10

    def __init__(self, state_path: str = "app_state.json"):
        self._state_path = state_path
        self._state_dict = None

    def load_state(self) -> None:
        if not os.path.exists(self._state_path):
            logger.info("No state file found, using defaults")
            self._state_dict = self.DEFAULT_STATE.copy()
            return

        try:
            with open(self._state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            self._state_dict = {**self.DEFAULT_STATE, **state}
            logger.debug(f"Loaded application state: {self._state_dict}")
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Error loading state: {e}. Using defaults.")
            self._state_dict = self.DEFAULT_STATE.copy()

    def save_state(self) -> None:
        try:
            with open(self._state_path, "w", encoding="utf-8") as f:
                json.dump(self._state_dict, f, indent=2, ensure_ascii=False)
            logger.debug("Application state saved")
        except IOError as e:
            logger.error(f"Error saving state: {e}")

    def get_current_transcription_file(self) -> str:
        return self._state_dict.get("current_transcription_file", "transcription.txt")

    def set_current_transcription_file(self, file_path: str) -> None:
        self._state_dict["current_transcription_file"] = file_path
        self.add_to_recent_files(file_path)
        self.save_state()

    def get_recent_files(self) -> List[str]:
        recent = self._state_dict.get("recent_files", [])
        return [f for f in recent if os.path.exists(f)]

    def add_to_recent_files(self, file_path: str) -> None:
        recent = self._state_dict.get("recent_files", [])
        
        file_path = os.path.abspath(file_path)
        
        if file_path in recent:
            recent.remove(file_path)
        
        recent.insert(0, file_path)
        
        self._state_dict["recent_files"] = recent[:self.MAX_RECENT_FILES]
        self.save_state()

    def clear_recent_files(self) -> None:
        self._state_dict["recent_files"] = []
        self.save_state()

    def get_recent_projects(self) -> List[str]:
        recent = self._state_dict.get("recent_projects", [])
        return [p for p in recent if os.path.isdir(p)]

    def add_to_recent_projects(self, folder: str) -> None:
        recent = self._state_dict.get("recent_projects", [])
        folder = os.path.abspath(folder)
        if folder in recent:
            recent.remove(folder)
        recent.insert(0, folder)
        self._state_dict["recent_projects"] = recent[:self.MAX_RECENT_FILES]
        self.save_state()

    def clear_recent_projects(self) -> None:
        self._state_dict["recent_projects"] = []
        self.save_state()

    def get_last_save_directory(self) -> str:
        return self._state_dict.get("last_save_directory", "")

    def set_last_save_directory(self, directory: str) -> None:
        self._state_dict["last_save_directory"] = directory
        self.save_state()

    def get_window_geometry(self) -> Optional[str]:
        return self._state_dict.get("window_geometry")

    def set_window_geometry(self, geometry: str) -> None:
        self._state_dict["window_geometry"] = geometry
        self.save_state()

    def get_last_audio_directory(self) -> str:
        return self._state_dict.get("last_audio_directory", "")

    def set_last_audio_directory(self, directory: str) -> None:
        self._state_dict["last_audio_directory"] = directory
        self.save_state()

    def get_dark_mode(self) -> bool:
        return self._state_dict.get("dark_mode", False)

    def set_dark_mode(self, enabled: bool) -> None:
        self._state_dict["dark_mode"] = enabled
        self.save_state()