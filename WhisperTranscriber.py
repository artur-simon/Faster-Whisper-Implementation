from faster_whisper import WhisperModel
from scipy.io.wavfile import write
import sounddevice as sd
import numpy as np
import tempfile
import threading
import time
import os

class LiveWhisperTranscriber:
    def __init__(self, model_size="large-v3", sample_rate=16000, device="cuda", compute_type="float16", language="auto", chunk_duration=5, overlap_duration=1):
        print("[INIT] Start initialization")
        print(f"[INIT] model_size={model_size}, sample_rate={sample_rate}, device={device}, compute_type={compute_type}, chunk_duration={chunk_duration}, overlap_duration={overlap_duration}")
        self.model = WhisperModel(model_size_or_path=model_size, device=device, compute_type=compute_type)
        print("[INIT] Model loaded successfully")
        self.language = language
        self.sample_rate = sample_rate
        self.chunk_samples = int(chunk_duration * sample_rate)
        self.overlap_samples = int(overlap_duration * sample_rate)
        print(f"[INIT] chunk_samples={self.chunk_samples}, overlap_samples={self.overlap_samples}")
        self.buffer = np.zeros((0, 1), dtype=compute_type)
        print(f"[INIT] buffer initialized: shape={self.buffer.shape}, dtype={self.buffer.dtype}")
        self.lock = threading.Lock()
        self.running = False
        self.last_end = 0.0
        print("[INIT] Initialization complete")

    def record_stream(self):
        print("[RECORD] record_stream started")

        def callback(indata, frames, time_info, status):
            if status:
                print(f"[RECORD][CALLBACK] status={status}")
            #print(f"[RECORD][CALLBACK] Received frames={frames}, indata.shape={indata.shape}, time_info={time_info}")
            with self.lock:
                before_shape = self.buffer.shape
                self.buffer = np.concatenate((self.buffer, indata.copy()))
                after_shape = self.buffer.shape
                #print(f"[RECORD][CALLBACK] Buffer updated: before={before_shape}, after={after_shape}")

        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='float32', callback=callback):
            print("[RECORD] InputStream opened")
            try:
                while self.running:
                    #print(f"[RECORD] running={self.running}, buffer_length={len(self.buffer)}")
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("[RECORD] KeyboardInterrupt detected, stopping")
                self.running = False
            print("[RECORD] record_stream stopped")

    def transcribe_loop(self, output_file):
        print("[BUFFER - TRANSCRIBE] transcribe_loop started")
        
        accumulated_text = ""
        try:
            while self.running:
                #print("[BUFFER] Loop iteration start")
                with self.lock:
                    buffer_len = len(self.buffer)
                    #print(f"[BUFFER] buffer_len={buffer_len}, required_chunk={self.chunk_samples}")
                    if buffer_len >= self.chunk_samples:
                        chunk = self.buffer[:self.chunk_samples]
                        self.buffer = self.buffer[self.chunk_samples - self.overlap_samples:]
                        # print(f"[BUFFER] Chunk created: chunk.shape={chunk.shape}, new_buffer_len={len(self.buffer)}")
                    else:
                        chunk = None
                        #print("[BUFFER] Not enough data for chunk")

                if chunk is not None:
                    print("[FILE] Processing chunk")
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                    tmp_path = tmp.name
                    #print(f"[FILE] Temporary file created: {tmp_path}")
                    tmp.close()

                    try:
                        write(tmp_path, self.sample_rate, chunk)
                        #print(f"[FILE] Chunk written to {tmp_path}")
                    except Exception as e:
                        print(f"[FILE][ERROR] Failed to write wav file: {e}")
                        self.running = False
                        break
                    
                    try:
                        segments, info = self.model.transcribe(
                            tmp_path, 
                            vad_filter=True, 
                            language=self.language,
                            word_timestamps=True, 
                            initial_prompt=accumulated_text, 
                            condition_on_previous_text=True)
                        #print(f"[TRANSCRIBE] Transcription complete: info={info}")
                    except Exception as e:
                        print(f"[TRANSCRIBE][ERROR] Transcription failed: {e}")
                        os.remove(tmp_path)
                        continue
                    try:
                        os.remove(tmp_path)
                        #print(f"[FILE] Temporary file removed: {tmp_path}")
                    except Exception as e:
                        print(f"[FILE][ERROR] Failed to remove tmp file: {e}")
                                    
                    full_text = ""
                    #print("[TRANSCRIBE] Iterating segments")
                    total_segments = 0
                    for segment in segments:
                        print(f"[TRANSCRIBE][SEGMENT] {total_segments}: text={segment.text}, start={segment.start}, end={segment.end}, last_end={self.last_end}")
                        for word in segment.words:
                            print(f"[TRANSCRIBE][SEGMENT][WORD] word: {word.word}, start: {word.start}, end: {word.end}, prob: {word.probability}")
                        if segment.start >= self.last_end:
                            print(f"[TRANSCRIBE][SEGMENT] Accepted segment {total_segments}")
                            full_text += segment.text + " "
                            self.last_end = segment.end
                        else:
                            print(f"[TRANSCRIBE][SEGMENT] Rejected segment {total_segments} (overlap)")
                        total_segments += 1
                    self.last_end = 0.0

                    full_text_stripped = full_text.strip()
                    print(f"[TRANSCRIBE] full_text_stripped='{full_text_stripped}'")
                    if full_text_stripped:
                        print("[TRANSCRIBE] Writing transcription to file")
                        accumulated_text = full_text_stripped
                        try:
                            with open(output_file, 'a', encoding="utf-8") as f:
                                f.write(full_text_stripped + "\n")
                            print(f"[TRANSCRIBE] Successfully wrote to {output_file}")
                        except Exception as e:
                            print(f"[TRANSCRIBE][ERROR] Failed to write output_file: {e}")
                else:
                    #print("[BUFFER] No chunk available, sleeping")
                    time.sleep(0.25)
        except KeyboardInterrupt:
            print("[RECORD] KeyboardInterrupt detected, stopping")
            self.running = False
            
        print("[BUFFER - TRANSCRIBE] transcribe_loop stopped")

    def run(self, output_file="transcription.txt"):
        print(f"[RUN] Starting with output_file={output_file}")
        self.running = True
        t_record = threading.Thread(target=self.record_stream, name="Thread-Record")
        t_transcribe = threading.Thread(target=self.transcribe_loop, args=(output_file,), name="Thread-Transcribe")
        t_record.start()
        print("[RUN] Record thread started")
        t_transcribe.start()
        print("[RUN] Transcribe thread started")
        
    def transcribe_audio(self, file_path, output_file="transcription.txt"):
        segments, info = self.model.transcribe(file_path, beam_size=5, vad_filter=True, language=self.language)
        #print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
        full_transcription = ""
        
        with open(output_file, 'a', encoding="utf-8") as f:
            for segment in segments:
                print(f'{segment.text}')
                full_transcription += segment.text + " "
                f.write(segment.text + "\n")
                f.flush()
                os.fsync(f.fileno())

        return full_transcription
    
    def shutdown(self):
        print("[SHUTDOWN] Requested")
        self.running = False
        time.sleep(0.5)  

        with self.lock:
            self.buffer = np.zeros((0, 1), dtype=self.buffer.dtype)
            print("[SHUTDOWN] Buffer cleared")

        try:
            if hasattr(self, "model") and self.model is not None:
                del self.model
                self.model = None
                print("[SHUTDOWN] Model released")
        except Exception as e:
            print(f"[SHUTDOWN][ERROR] Failed to release model: {e}")

        self.last_end = 0.0
        print("[SHUTDOWN] Complete")