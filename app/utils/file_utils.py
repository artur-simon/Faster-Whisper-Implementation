import numpy as np
from scipy.io.wavfile import write
import tempfile
import os


def create_wav_file(audio_data: np.ndarray, sample_rate: int) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_path = tmp.name
    tmp.close()
    write(tmp_path, sample_rate, audio_data)
    return tmp_path


def delete_file(file_path: str) -> None:
    if os.path.exists(file_path):
        os.remove(file_path)
