import sounddevice as sd
import numpy as np
from pynput import keyboard
from scipy.io.wavfile import write
import tempfile
import os
import sys
from faster_whisper import WhisperModel
import pyperclip
import pyautogui

class WhisperVoice:
    def __init__(self, model_size="large-v3", sample_rate=44100, assigned_key=keyboard.Key.scroll_lock):
        self.model_size = model_size
        self.sample_rate = sample_rate
        self.model = WhisperModel(model_size, device="cuda", compute_type="float16")
        self.assigned_key = assigned_key
        self.is_recording = False
        # Run on GPU with FP16
        # model = WhisperModel(model_size, device="cuda", compute_type="float16")
        # or run on GPU with INT8
        # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
        # or run on CPU with INT8
        # model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def on_press(self, key):
        if key == self.assigned_key:
            self.is_recording = not self.is_recording
            if self.is_recording:
                print("Recording started.")
            else:
                print("Recording stopped.")
                return False
    
    def record_audio(self):
        recording = np.array([], dtype='float64').reshape(0, 2)
        frames_per_buffer = int(self.sample_rate * 0.1)
        
        with keyboard.Listener(on_press=self.on_press) as listener:
            while True:
                if self.is_recording:
                    chunk = sd.rec(frames_per_buffer, samplerate=self.sample_rate, channels=2, dtype='float64')
                    sd.wait()
                    recording = np.vstack([recording, chunk])
                if not self.is_recording and len(recording) > 0:
                    break
            listener.join()
        return recording

    def save_temp_audio(self, recording):
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        write(temp_file.name, self.sample_rate, recording)
        return temp_file.name
    
    def transcribe_audio(self, file_path, output_file, should_paste_content):
        segments, info = self.model.transcribe(file_path, beam_size=5)
        print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
        full_transcription = ""
        
        with open(output_file, 'a') as f:
            for segment in segments:
                print(segment.text)
                full_transcription += segment.text + " "
                f.write(segment.text + "\n")
        os.remove(file_path)
        
        if should_paste_content:
            pyperclip.copy(full_transcription)
            print("Transcription copied to clipboard.")
            pyautogui.hotkey('ctrl', 'v')

        return full_transcription
    
    def run(self, output_file="transcription.txt", file_path=None):
        if file_path:
            print(f"Transcribing file: {file_path}")
            self.transcribe_audio(file_path, output_file, False)
        else:
            print(f"Press {self.assigned_key} to start/stop recording")
            while True:
                recording = self.record_audio()
                temp_file_path = self.save_temp_audio(recording)
                self.transcribe_audio(temp_file_path, output_file, True)

if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else None
    transcriber = WhisperVoice()
    transcriber.run(file_path=file_path)