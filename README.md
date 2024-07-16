# Whisper Voice Transcriber

This project provides a simple voice transcription tool that utilizes the `WhisperModel` for speech recognition. The application records audio when the spacebar is held down, saves the audio to a temporary file, transcribes the audio using a Whisper model, and then saves the transcription to a text file. Additionally, the transcribed text is copied to the clipboard and can be automatically pasted into any application where the cursor is active.

## Features

- **Audio Recording**: Start and stop recording audio using the spacebar.
- **Transcription**: Transcribe recorded audio using the `WhisperModel`.
- **File Saving**: Save the transcription to a specified text file.
- **Clipboard Integration**: Copy the transcribed text to the clipboard.
- **Auto-Paste**: Automatically paste the transcribed text where the cursor is active.

## Requirements

- Python 3.x
- sounddevice
- numpy
- pynput
- scipy
- faster_whisper
- pyperclip
- pyautogui

## Installation

Follow the official https://github.com/guillaumekln/faster-whisper for installation


##  Notes
The script uses the pyautogui library to simulate the Ctrl+V hotkey, which pastes the transcribed text. Ensure the cursor is active in the desired text input field when transcription completes.
Adjust the model_size parameter in the WhisperVoice class to use different Whisper model sizes based on your hardware capabilities and accuracy requirements.



# Acknowledgements
The faster_whisper library (https://github.com/guillaumekln/faster-whisper) for the Whisper model implementation.
The developers of sounddevice, numpy, pynput, scipy, pyperclip, and pyautogui for their excellent libraries.