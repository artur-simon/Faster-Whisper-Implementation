import logging
import time
from typing import List, Optional, Callable
from dataclasses import dataclass, field

from app.models import Word, TranscriptionConfig
from app.transcription.transcription_engine import TranscriptionEngine
from app.transcription.overlap_resolver import (
    resolve_overlapping_words,
    adjust_word_timestamps,
    get_words_after_time,
    get_words_before_time
)
from app.utils.file_utils import create_wav_file
from tests.simulation.audio_file_simulator import AudioFileSimulator

logger = logging.getLogger("tests.simulation.test_harness")


@dataclass
class TranscriptionEvent:
    timestamp: float
    event_type: str
    words: List[Word] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class TranscriptionTestResult:
    all_words: List[Word]
    events: List[TranscriptionEvent]
    final_text: str
    total_chunks: int
    total_duration: float
    
    def get_text_by_timerange(self, start: float, end: float) -> str:
        words_in_range = [w for w in self.all_words if start <= w.start < end]
        return " ".join(w.text for w in words_in_range)
    
    def find_overlap_issues(self) -> List[dict]:
        issues = []
        for i in range(len(self.all_words) - 1):
            w1, w2 = self.all_words[i], self.all_words[i + 1]
            if w1.end > w2.start:
                issues.append({
                    "type": "overlap",
                    "word1": w1.text,
                    "word2": w2.text,
                    "time1": (w1.start, w1.end),
                    "time2": (w2.start, w2.end),
                    "overlap_duration": w1.end - w2.start
                })
        return issues
    
    def find_timing_gaps(self, threshold: float = 1.0) -> List[dict]:
        gaps = []
        for i in range(len(self.all_words) - 1):
            w1, w2 = self.all_words[i], self.all_words[i + 1]
            gap = w2.start - w1.end
            if gap > threshold:
                gaps.append({
                    "type": "gap",
                    "after_word": w1.text,
                    "before_word": w2.text,
                    "gap_duration": gap,
                    "position": (w1.end, w2.start)
                })
        return gaps


class TranscriptionTestHarness:
    def __init__(self, config: TranscriptionConfig):
        self._config = config
        self._chunk_samples = int(config.chunk_duration * config.sample_rate)
        self._overlap_samples = int(config.overlap_duration * config.sample_rate)
        
        self._engine: Optional[TranscriptionEngine] = None
        self._simulator: Optional[AudioFileSimulator] = None
        
        self._all_words: List[Word] = []
        self._events: List[TranscriptionEvent] = []
        
        self.timestamp_offset = 0.0
        self._previous_overlap_words: List[Word] = []
        
        self._running = False
        
        self.total_chunks = 0
        
        self._event_callback: Optional[Callable[[TranscriptionEvent], None]] = None
    
    def set_event_callback(self, callback: Callable[[TranscriptionEvent], None]) -> None:
        self._event_callback = callback
    
    def run_test(
        self, 
        audio_file: str, 
        realtime_speed: float = 10.0,
        max_chunks: Optional[int] = None
    ) -> TranscriptionTestResult:
        
        logger.info(f"Starting transcription test with audio file: {audio_file}")
        logger.info(f"Config: chunk={self._config.chunk_duration}s, overlap={self._config.overlap_duration}s")
        
        start_time = time.time()
        
        self._initialize_test(audio_file, realtime_speed)
        
        try:
            self._run_processing_loop(max_chunks)
        finally:
            self._cleanup_test()
        
        end_time = time.time()
        test_duration = end_time - start_time
        
        final_text = " ".join(w.text for w in self._all_words)
        
        result = TranscriptionTestResult(
            all_words=self._all_words.copy(),
            events=self._events.copy(),
            final_text=final_text,
            total_chunks=self.total_chunks,
            total_duration=test_duration
        )
        
        logger.info(f"Test completed: {len(self._all_words)} words, {test_duration:.2f}s")
        
        return result
    
    def _initialize_test(self, audio_file: str, realtime_speed: float) -> None:
        self._all_words = []
        self._events = []
        self.timestamp_offset = 0.0
        self._previous_overlap_words = []
        
        self._engine = TranscriptionEngine(self._config)
        self._simulator = AudioFileSimulator(audio_file, self._config.sample_rate, realtime_speed)
        self._simulator.start()
        self._running = True
    
    def _run_processing_loop(self, max_chunks: Optional[int]) -> None:
        while self._running:
            
            if self._simulator.has_data(self._chunk_samples):
                self._process_chunk()
            else:
                logger.info(f"Processing final partial chunk")
                self._process_partial_chunk()
                logger.info("Audio simulation finished")
                break
        
        if self._previous_overlap_words:
            logger.info(f"Writing {len(self._previous_overlap_words)} pending words")
            self._all_words.extend(self._previous_overlap_words)
    
    def _process_chunk(self) -> None:
        chunk = self._simulator.get_chunk(self._chunk_samples, self._overlap_samples)
        if chunk is None:
            return
        
        logger.debug(f"Processing chunk: {len(chunk)} samples")
        self.total_chunks += 1
        self._emit_event(TranscriptionEvent(
            timestamp=time.time(),
            event_type="chunk_start",
            metadata={"samples": len(chunk), "chunk_offset": self.timestamp_offset}
        ))
        
        try:
            np_audio = self._simulator.resample_chunk_to_16k(chunk, self._config.sample_rate)
            segments = self._engine.transcribe_audio(np_audio)
            self._handle_transcription_result(segments)
        except Exception as e:
            logger.error(f"Error processing chunk: {e}", exc_info=True)
            self._emit_event(TranscriptionEvent(
                timestamp=time.time(),
                event_type="error",
                metadata={"error": str(e)}
            ))
    
    def _process_partial_chunk(self) -> None:
        remaining_size = self._simulator.get_buffer_size()
        chunk = self._simulator.get_chunk(remaining_size, 0)
        if chunk is None:
            return
        
        logger.debug(f"Processing partial chunk: {len(chunk)} samples")
        self.total_chunks += 1
        self._emit_event(TranscriptionEvent(
            timestamp=time.time(),
            event_type="chunk_start",
            metadata={"samples": len(chunk), "chunk_offset": self.timestamp_offset, "partial": True}
        ))
        
        try:
            np_audio = self._simulator.resample_chunk_to_16k(chunk, self._config.sample_rate)
            segments = self._engine.transcribe_audio(np_audio)
            self._handle_transcription_result(segments)
        except Exception as e:
            logger.error(f"Error processing partial chunk: {e}", exc_info=True)
            self._emit_event(TranscriptionEvent(
                timestamp=time.time(),
                event_type="error",
                metadata={"error": str(e)}
            ))
    
    def _handle_transcription_result(self, segments) -> None:
        current_words = []
        for segment in segments:
            if segment.no_speech_prob < self._config.no_speech_threshold:
                current_words.extend(segment.words)
        
        has_speech = len(current_words) > 0
        
        if current_words:
            current_words = adjust_word_timestamps(current_words, self.timestamp_offset)
        
        self._emit_event(TranscriptionEvent(
            timestamp=time.time(),
            event_type="raw_transcription",
            words=current_words.copy(),
            metadata={
                "has_speech": has_speech,
                "previous_overlap_count": len(self._previous_overlap_words)
            }
        ))
        
        resolved_words = resolve_overlapping_words(
            self._previous_overlap_words,
            current_words
        )
        
        if resolved_words:
            self._emit_event(TranscriptionEvent(
                timestamp=time.time(),
                event_type="resolved_words",
                words=resolved_words.copy(),
                metadata={
                    "previous_text": " ".join(w.text for w in self._previous_overlap_words),
                    "current_text": " ".join(w.text for w in current_words),
                    "resolved_text": " ".join(w.text for w in resolved_words)
                }
            ))
        
        next_offset = self.timestamp_offset + (self._config.chunk_duration - self._config.overlap_duration)
        
        if has_speech:
            words_to_write = get_words_before_time(resolved_words, next_offset)
            self._previous_overlap_words = get_words_after_time(current_words, next_offset)
            
            if words_to_write:
                self._all_words.extend(words_to_write)
                self._emit_event(TranscriptionEvent(
                    timestamp=time.time(),
                    event_type="words_written",
                    words=words_to_write.copy()
                ))
            
            self.timestamp_offset = next_offset
        else:
            if resolved_words:
                self._all_words.extend(resolved_words)
                self._emit_event(TranscriptionEvent(
                    timestamp=time.time(),
                    event_type="words_written",
                    words=resolved_words.copy()
                ))
            
            self._previous_overlap_words = []
            self.timestamp_offset = 0.0
    
    def _emit_event(self, event: TranscriptionEvent) -> None:
        self._events.append(event)
        if self._event_callback:
            self._event_callback(event)
    
    def _cleanup_test(self) -> None:
        self._running = False
        if self._simulator:
            self._simulator.stop()
        if self._engine:
            self._engine.release()

