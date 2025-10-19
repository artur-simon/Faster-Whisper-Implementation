import pyautogui
import pyperclip
import logging

logger = logging.getLogger("app.utils.os_print")

@staticmethod
def paste_content(content):
    pyperclip.copy(content)
    logger.debug("Transcription copied to clipboard, pasting")
    pyautogui.hotkey('ctrl', 'v')