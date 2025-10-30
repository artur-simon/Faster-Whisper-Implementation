import uuid
import numpy as np
from scipy.io.wavfile import write
import tempfile
import os
from typing import List


def create_temp_wav_file(audio_data: np.ndarray, sample_rate: int) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_path = tmp.name
    tmp.close()
    write(tmp_path, sample_rate, audio_data)
    return tmp_path

def create_wav_file(audio_data: np.ndarray, sample_rate: int) -> str:
    dir = os.path.dirname(os.path.abspath(__file__))
    wav_path = os.path.join(dir, f"{uuid.uuid4().hex}.wav")

    write(wav_path, sample_rate, audio_data)
    return wav_path

def delete_file(file_path: str) -> None:
    if os.path.exists(file_path):
        os.remove(file_path)


def find_audio_files(directory: str, recursive: bool = True) -> List[str]:
    supported_extensions = {'.mp3', '.wav'}
    audio_files = []
    
    if recursive:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if os.path.splitext(file)[1].lower() in supported_extensions:
                    audio_files.append(os.path.join(root, file))
    else:
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path) and os.path.splitext(file)[1].lower() in supported_extensions:
                audio_files.append(file_path)
    
    return sorted(audio_files)
