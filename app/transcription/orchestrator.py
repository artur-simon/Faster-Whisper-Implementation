import threading
import logging
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
from app.utils.document_writer import TranscriptionWriter

logger = logging.getLogger("app.transcription.orchestrator")


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
        
        self._audio_capture = AudioCapture(config.sample_rate, device_index=config.mic_id)
        self._engine = TranscriptionEngine(config)
        self._writer = TranscriptionWriter(output_path, config)
        
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
            logger.warning("Orchestrator already running, ignoring start request")
            return
        
        logger.info("Starting transcription orchestrator")
        self._running = True
        self._audio_capture.start()
        self._thread = threading.Thread(target=self._process_loop, name="Orchestrator - process_loop")
        self._thread.start()
        logger.info("Transcription orchestrator started successfully")
    
    
    def stop(self) -> None:
        logger.info("Stopping transcription orchestrator")
        self._running = False
        self._audio_capture.stop()
        if self._state.pending_words:
            logger.debug(f"Writing {len(self._state.pending_words)} pending words before stopping")
            self._writer.write_words(self._state.pending_words)
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("Transcription orchestrator stopped")
    
    
    def _process_loop(self) -> None:
        while self._running:
            if self._audio_capture.has_data(self._chunk_samples):
                self._process_chunk()
    
    
    def _process_chunk(self) -> None:
        chunk = self._audio_capture.get_chunk(self._chunk_samples, self._overlap_samples)
        if chunk is None:
            logger.debug("No chunk available for processing")
            return
        
        logger.debug(f"Processing audio chunk: {len(chunk)} samples")
        np_audio = self._audio_capture.resample_chunk_to_16k(chunk, self._config.sample_rate)
        
        try:
            segments = self._engine.transcribe_audio(np_audio, self._state.pending_words)
            self._handle_transcription_result(segments)
        except Exception as e:
            logger.error(f"Error processing chunk: {e}", exc_info=True)
            
            
            
    def _handle_transcription_result(self, segments) -> None:
        current_words = self._extract_words_from_segments(segments)
        has_speech = len(current_words) > 0
        
        if current_words:
            current_words = adjust_word_timestamps(current_words, self._state.chunk_offset)
        
        resolved_words = resolve_overlapping_words(
            self._state.previous_overlap_words,
            current_words
        )
        
        if(resolved_words):
            logger.debug(f"[current_words]: {' '.join(word.text for word in current_words)}")
            logger.debug(" ".join(f"{w.text}(prob:{w.probability:.2f})" for w in current_words))
            logger.debug(f"[previous]: {' '.join(word.text for word in self._state.previous_overlap_words)}")
            logger.debug(f"[resolved_words]: {' '.join(word.text for word in resolved_words)}")
            
        next_offset = self._config.chunk_duration + self._state.chunk_offset - self._config.overlap_duration
        
        if has_speech:
            words_to_write = get_words_before_time(current_words, next_offset)
            self._state.previous_overlap_words = get_words_after_time(current_words, next_offset)
            self._state.pending_words = self._state.previous_overlap_words
            
            if words_to_write:
                logger.debug(f"Writing {len(words_to_write)} words to file")
                self._writer.write_words(words_to_write)
                self._state.should_break_line = True
            
            next_offset = self._config.chunk_duration + self._state.chunk_offset - self._config.overlap_duration
            self._state.chunk_offset = next_offset
        else:
            if resolved_words:
                logger.debug(f"No speech detected, writing {len(resolved_words)} resolved words")
                self._writer.write_words(resolved_words)
            
            if (self._state.should_break_line):
                logger.debug("Adding line break after speech segment")
                self._writer.write_string("\n")
                self._state.should_break_line = False
            
            self._state.pending_words = []
            self._state.previous_overlap_words = []
            self._state.chunk_offset = 0.0
    
    
    def _extract_words_from_segments(self, segments) -> List[Word]:
        words = []
        for segment in segments:
            if segment.no_speech_prob < self._config.no_speech_threshold:
                words.extend(segment.words)
        return words
    
    
    def release(self) -> None:
        logger.info("Releasing orchestrator resources")
        self.stop()
        self._audio_capture.release()
        
        self._engine.release()
        self._engine = None
        logger.info("Orchestrator resources released")