import pyautogui
import pyperclip

@staticmethod
def paste_content(content):
    pyperclip.copy(content)
    print("Transcription copied to clipboard.")
    pyautogui.hotkey('ctrl', 'v')