import threading
from typing import List, Optional
from dataclasses import dataclass

from app.models import Word, TranscriptionConfig
from app.audio.audio_capture import AudioCapture
from app.transcription.transcription_engine import TranscriptionEngine
from app.transcription.overlap_resolver import (
    resolve_overlapping_words,
    adjust_word_timestamps,
    get_words_after_time,
    get_words_before_time
)
from app.utils.file_utils import create_wav_file, delete_file
from app.utils.document_writer import TranscriptionWriter


@dataclass
class TranscriptionState:
    chunk_offset: float
    previous_overlap_words: List[Word]
    pending_words: List[Word]
    should_break_line: bool


class TranscriptionOrchestrator:
    def __init__(self, config: TranscriptionConfig, output_path: str):
        self._config = config
        self._output_path = output_path
        
        self._audio_capture = AudioCapture(config.sample_rate)
        self._engine = TranscriptionEngine(config)
        self._writer = TranscriptionWriter(output_path)
        
        self._chunk_samples = int(config.chunk_duration * config.sample_rate)
        self._overlap_samples = int(config.overlap_duration * config.sample_rate)
        
        self._state = TranscriptionState(
            chunk_offset=0.0,
            previous_overlap_words=[],
            pending_words=[],
            should_break_line=False
        )
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
    
    def start(self) -> None:
        if self._running:
            return
        
        self._running = True
        self._audio_capture.start()
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> None:
        self._running = False
        self._audio_capture.stop()
        if self._state.pending_words:
            self._writer.write_words(self._state.pending_words)
        if self._thread:
            self._thread.join(timeout=2.0)
    
    def _process_loop(self) -> None:
        while self._running:
            if self._audio_capture.has_data(self._chunk_samples):
                self._process_chunk()
    
    def _process_chunk(self) -> None:
        chunk = self._audio_capture.get_chunk(self._chunk_samples, self._overlap_samples)
        if chunk is None:
            return
        
        audio_path = create_wav_file(chunk, self._config.sample_rate)
        
        try:
            segments = self._engine.transcribe_file(audio_path, self._state.pending_words)
            self._handle_transcription_result(segments)
        finally:
            delete_file(audio_path)
    
    def _handle_transcription_result(self, segments) -> None:
        current_words = self._extract_words_from_segments(segments)
        has_speech = len(current_words) > 0
        
        if current_words:
            print("[words]:")
            for word in current_words:
                print(f"{word.text} - prob:{word.probability:2f}")
                
            current_words = adjust_word_timestamps(current_words, self._state.chunk_offset)
        
        resolved_words = resolve_overlapping_words(
            self._state.previous_overlap_words,
            current_words
        )
        
        if(resolved_words):
            print("[current_words]:", " ".join(word.text for word in current_words))
            print("[previous]", " ".join(word.text for word in self._state.previous_overlap_words))
            print("[resolved_words]:", " ".join(word.text for word in resolved_words))
                
        next_offset = self._config.chunk_duration + self._state.chunk_offset - self._config.overlap_duration
        
        if not has_speech:
            self._writer.write_words(resolved_words)
            
            if (self._state.should_break_line):
                self._writer.write_string_to_file("\n")
                self._state.should_break_line = False
            
            self._state.pending_words = []
            self._state.previous_overlap_words = []
            self._state.chunk_offset = 0.0
        else:
            words_to_write = get_words_before_time(current_words, next_offset)
            self._state.previous_overlap_words = get_words_after_time(current_words, next_offset)
            self._state.pending_words = self._state.previous_overlap_words
            
            if words_to_write:
                self._writer.write_words(words_to_write)
                self._state.should_break_line = True
            
            self._state.chunk_offset = next_offset
    
    def _extract_words_from_segments(self, segments) -> List[Word]:
        words = []
        for segment in segments:
            if segment.no_speech_probability < self._config.no_speech_threshold:
                words.extend(segment.words)
        return words
        
    def release(self) -> None:
        self.stop()
        self._audio_capture.clear_buffer()
        self._engine.release()

