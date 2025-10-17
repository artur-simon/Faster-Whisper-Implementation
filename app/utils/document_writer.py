from typing import List
from app.models import TranscriptionConfig, Word

from app.utils.os_print import paste_content


class TranscriptionWriter:
    def __init__(
        self,
        output_path: str,
        config: TranscriptionConfig,
        encoding: str = "utf-8",
    ):
        self._output_path = output_path
        self._encoding = encoding
        self._config = config

    def write_words(self, words: List[Word]) -> None:
        if not words:
            return
        text = self._format_words(words)
        self.write_string(text + " ")

    def _format_words(self, words: List[Word]) -> str:
        return "".join([w.text for w in words]).strip()

    def write_string(self, text: str) -> None:
        self.write_string_to_file(text)
        if self._config.should_paste_content:
            paste_content(text)

    def write_string_to_file(self, text: str) -> None:
        with open(self._output_path, "a", encoding=self._encoding) as f:
            f.write(text + "")
