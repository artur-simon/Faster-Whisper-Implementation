import threading
import logging
from typing import List, Optional
from dataclasses import dataclass

from app.models import Word, TranscriptionConfig
from app.audio.audio_data_provider import AudioDataProvider
from app.transcription.transcription_engine import TranscriptionEngine
from app.transcription.overlap_resolver import (
    resolve_overlapping_words,
    adjust_word_timestamps,
    get_words_after_time,
    get_words_before_time
)
from app.transcription.local_agreement import LocalAgreementTracker
from app.utils.document_writer import TranscriptionWriter

logger = logging.getLogger("app.transcription.orchestrator")


@dataclass
class TranscriptionState:
    timestamp_offset: float
    previous_overlap_words: List[Word]
    accumulated_text: str
    should_break_line: bool
    agreement_tracker: Optional['LocalAgreementTracker'] = None


class TranscriptionOrchestrator:
    def __init__(self, config: TranscriptionConfig, output_path: str, audio_data_provider: AudioDataProvider):
        self._config = config
        self._output_path = output_path
        self._audio_data_provider = audio_data_provider

        self._engine = TranscriptionEngine(config)
        self._writer = TranscriptionWriter(output_path, config)
        
        self._chunk_samples = int(config.chunk_duration * config.sample_rate)
        self._overlap_samples = int(config.overlap_duration * config.sample_rate)
        
        self._state = TranscriptionState(
            timestamp_offset=0.0,
            previous_overlap_words=[],
            accumulated_text="",
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
        self._thread = threading.Thread(target=self._process_loop, name="Orchestrator - process_loop")
        self._thread.start()
        logger.info("Transcription orchestrator started successfully")
    
    
    def stop(self) -> None:
        logger.info("Stopping transcription orchestrator")
        self._running = False
        if self._state.previous_overlap_words:
            logger.debug(f"Writing {len(self._state.previous_overlap_words)} pending words before stopping")
            self._writer.write_words(self._state.previous_overlap_words)
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("Transcription orchestrator stopped")
    
    
    def _process_loop(self) -> None:
        while self._running:
            if self._audio_data_provider.has_data(self._chunk_samples):
                self._process_chunk()
    
    
    def _process_chunk(self) -> None:
        chunk = self._audio_data_provider.get_transcription_chunk(self._chunk_samples, self._overlap_samples)
        if chunk is None:
            logger.debug("No chunk available for processing")
            return

        logger.debug(f"Processing audio chunk: {len(chunk)} samples")

        try:
            np_audio = self._audio_data_provider.resample_chunk_to_16k(chunk, self._config.sample_rate)
            
            context_prompt = None
            if self._config.use_previous_context:
                context_prompt = (
                    self._state.agreement_tracker.get_context_buffer() 
                    if self._config.transcription_algorithm == 'local_agreement' and self._state.agreement_tracker 
                    else self._state.accumulated_text[-150:] #150 characters max.
                )
            
            segments = self._engine.transcribe_audio(
                audio_source=np_audio, 
                context_prompt=context_prompt,
            )
            self._handle_transcription_result(segments)
        except Exception as e:
            logger.error(f"Error processing chunk: {e}", exc_info=True)
            
            
            
    def _handle_transcription_result(self, segments) -> None:
        current_words = self._extract_words_from_segments(segments)
        has_speech = len(current_words) > 0
        
        if current_words:
            current_words = adjust_word_timestamps(current_words, self._state.timestamp_offset)
        
        if self._config.transcription_algorithm == 'local_agreement':
            self._handle_local_agreement(current_words, has_speech)
        else:
            self._handle_simple_overlap_resolve(current_words, has_speech)
    
    def _handle_simple_overlap_resolve(self, current_words: List[Word], has_speech: bool) -> None:
        resolved_words = resolve_overlapping_words(
            self._state.previous_overlap_words, current_words)
        
        if(resolved_words):
            logger.debug(f"[current_words]: {' '.join(word.text for word in current_words)}")
            logger.debug(" ".join(f"{w.text}(prob:{w.probability:.2f})" for w in current_words))
            logger.debug(f"[previous]: {' '.join(word.text for word in self._state.previous_overlap_words)}")
            logger.debug(f"[resolved_words]: {' '.join(word.text for word in resolved_words)}")
            
        next_offset = self._state.timestamp_offset + (self._config.chunk_duration - self._config.overlap_duration)
        
        if has_speech:
            words_to_write = get_words_before_time(resolved_words, next_offset)
            self._state.previous_overlap_words = get_words_after_time(current_words, next_offset)
            
            if words_to_write:
                logger.debug(f"Writing {len(words_to_write)} words to file")
                self._writer.write_words(words_to_write)
                self._state.accumulated_text += "".join([w.text for w in words_to_write])
                self._state.should_break_line = True
            
            self._state.timestamp_offset = next_offset
        else:
            if resolved_words:
                logger.debug(f"No speech detected, writing {len(resolved_words)} resolved words")
                self._writer.write_words(resolved_words)
                
            if (self._state.should_break_line):
                logger.debug("Adding line break after speech segment")
                self._writer.write_string_to_file("\n")
                self._state.should_break_line = False
            
            self._state.accumulated_text = ""
            self._state.previous_overlap_words = []
            self._state.timestamp_offset = 0.0
    
    def _handle_local_agreement(self, current_words: List[Word], has_speech: bool) -> None:
        if self._state.agreement_tracker is None:
            logger.info("Initializing LocalAgreementTracker")
            self._state.agreement_tracker = LocalAgreementTracker(self._config.local_agreement_config)
        
        if has_speech:
            locked_words, remaining_words, disagreement = self._state.agreement_tracker.process_hypothesis(current_words)
            
            if locked_words:
                logger.debug(f"Emitting {len(locked_words)} locked words")
                self._writer.write_words(locked_words)
                self._state.should_break_line = True
            
            self._state.previous_overlap_words = remaining_words
            
            if self._config.use_previous_context:
                self._state.accumulated_text = self._state.agreement_tracker.get_context_buffer()
            
            if disagreement:
                logger.debug("Disagreement detected, waiting for next chunk")
            
            next_offset = self._state.timestamp_offset + (self._config.chunk_duration - self._config.overlap_duration)
            self._state.timestamp_offset = next_offset
        else:
            if self._state.previous_overlap_words:
                logger.debug(f"No speech detected, writing {len(self._state.previous_overlap_words)} remaining words")
                self._writer.write_words(self._state.previous_overlap_words)
            
            if self._state.should_break_line:
                logger.debug("Adding line break after speech segment")
                self._writer.write_string_and_paste("\n")
                self._state.should_break_line = False
            
            self._state.accumulated_text = ""
            self._state.previous_overlap_words = []
            self._state.timestamp_offset = 0.0
            if self._state.agreement_tracker:
                self._state.agreement_tracker.reset()
    
    
    def _extract_words_from_segments(self, segments) -> List[Word]:
        words = []
        for segment in segments:
            if segment.no_speech_prob < self._config.no_speech_threshold:
                words.extend(segment.words)
        return words
    
    
    def release(self) -> None:
        logger.info("Releasing orchestrator resources")
        self.stop()

        self._engine.release()
        self._engine = None
        logger.info("Orchestrator resources released")