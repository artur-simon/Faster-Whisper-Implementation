import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple
from collections import deque
from app.models import Word, LocalAgreementConfig

logger = logging.getLogger("app.transcription.local_agreement")


@dataclass
class LockedSegment:
    words: List[Word]
    start_idx: int
    end_idx: int


@dataclass
class AgreementState:
    previous_hypothesis: List[Word]
    locked_segments: List[LockedSegment]
    context_buffer: deque
    agreement_counter: int
    last_locked_end_time: float


def compute_word_edit_distance(words1: List[Word], words2: List[Word]) -> float:
    if not words1 and not words2:
        return 0.0
    if not words1 or not words2:
        return 1.0
    
    text1 = [w.text.lower().strip() for w in words1]
    text2 = [w.text.lower().strip() for w in words2]
    
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    edit_distance = dp[m][n]
    max_len = max(m, n)
    normalized_distance = edit_distance / max_len if max_len > 0 else 0.0
    
    return normalized_distance


def compute_confidence_similarity(words1: List[Word], words2: List[Word]) -> float:
    if not words1 and not words2:
        return 1.0
    if not words1 or not words2:
        return 0.0
    
    text1 = [w.text.lower().strip() for w in words1]
    text2 = [w.text.lower().strip() for w in words2]
    
    matching_words = 0
    confidence_sum = 0.0
    
    min_len = min(len(text1), len(text2))
    for i in range(min_len):
        if text1[i] == text2[i]:
            matching_words += 1
            confidence_sum += (words1[i].probability + words2[i].probability) / 2
    
    if matching_words == 0:
        return 0.0
    
    avg_confidence = confidence_sum / matching_words
    match_ratio = matching_words / max(len(text1), len(text2))
    
    return avg_confidence * match_ratio


def compare_hypotheses(
    prev_words: List[Word], 
    curr_words: List[Word], 
    config: LocalAgreementConfig
) -> Tuple[float, int, int]:
    if not prev_words or not curr_words:
        return 0.0, 0, 0
    
    edit_distance = compute_word_edit_distance(prev_words, curr_words)
    confidence_similarity = compute_confidence_similarity(prev_words, curr_words)
    
    edit_similarity = 1.0 - edit_distance
    combined_similarity = (edit_similarity + confidence_similarity) / 2
    
    overlap_start = 0
    overlap_end = min(len(prev_words), len(curr_words))
    
    return combined_similarity, overlap_start, overlap_end


def find_agreement_span(
    prev_words: List[Word], 
    curr_words: List[Word], 
    config: LocalAgreementConfig
) -> Optional[Tuple[int, int]]:
    if not prev_words or not curr_words:
        return None
    
    if len(prev_words) < config.min_words or len(curr_words) < config.min_words:
        return None
    
    similarity, start_idx, end_idx = compare_hypotheses(prev_words, curr_words, config)
    
    edit_distance = compute_word_edit_distance(prev_words, curr_words)
    confidence_similarity = compute_confidence_similarity(prev_words, curr_words)
    
    meets_edit_threshold = edit_distance <= config.edit_threshold
    meets_confidence_threshold = confidence_similarity >= config.confidence_threshold
    
    if meets_edit_threshold and meets_confidence_threshold:
        agreement_length = min(len(prev_words), len(curr_words))
        if agreement_length >= config.min_words:
            return (start_idx, agreement_length)
    
    return None


def lock_segment(
    words: List[Word], 
    start_idx: int, 
    end_idx: int, 
    config: LocalAgreementConfig
) -> LockedSegment:
    return LockedSegment(
        words=words[start_idx:end_idx],
        start_idx=start_idx,
        end_idx=end_idx
    )


def get_words_beyond_lock(words: List[Word], last_locked_end_idx: int) -> List[Word]:
    return words[last_locked_end_idx:]


def rollback_to_last_lock(all_words: List[Word], locked_segments: List[LockedSegment]) -> List[Word]:
    if not locked_segments:
        return all_words
    
    last_segment = locked_segments[-1]
    last_locked_end_time = last_segment.words[-1].end if last_segment.words else 0.0
    
    return [w for w in all_words if w.start >= last_locked_end_time]


class LocalAgreementTracker:
    def __init__(self, config: LocalAgreementConfig):
        self._config = config
        self._state = AgreementState(
            previous_hypothesis=[],
            locked_segments=[],
            context_buffer=deque(maxlen=config.context_size),
            agreement_counter=0,
            last_locked_end_time=0.0
        )
        logger.info(f"LocalAgreementTracker initialized with config: {config}")
    
    def process_hypothesis(self, current_words: List[Word]) -> Tuple[List[Word], List[Word], bool]:
        logger.debug(f"Processing hypothesis with {len(current_words)} words")
        
        if not current_words:
            return [], [], False
        
        if not self._state.previous_hypothesis:
            self._state.previous_hypothesis = current_words
            logger.debug("First hypothesis, storing for next comparison")
            return [], current_words, False
        
        agreement_span = find_agreement_span(
            self._state.previous_hypothesis, 
            current_words, 
            self._config
        )
        
        disagreement_flag = False
        locked_words_to_emit = []
        
        if agreement_span:
            start_idx, length = agreement_span
            self._state.agreement_counter += 1
            logger.debug(f"Agreement found: span=({start_idx}, {length}), counter={self._state.agreement_counter}")
            
            if self._state.agreement_counter >= self._config.agreement_count:
                logger.info(f"Locking segment: {length} words after {self._state.agreement_counter} agreements")
                
                locked_segment = lock_segment(current_words, start_idx, length, self._config)
                self._state.locked_segments.append(locked_segment)
                
                locked_words_to_emit = locked_segment.words
                
                for word in locked_segment.words:
                    self._state.context_buffer.append(word.text)
                
                if locked_segment.words:
                    self._state.last_locked_end_time = locked_segment.words[-1].end
                
                remaining_words = get_words_beyond_lock(current_words, length)
                self._state.previous_hypothesis = remaining_words
                self._state.agreement_counter = 0
                
                return locked_words_to_emit, remaining_words, False
            else:
                self._state.previous_hypothesis = current_words
                return [], current_words, False
        else:
            logger.debug("Disagreement detected, marking for potential rollback")
            disagreement_flag = True
            self._state.agreement_counter = 0
            
            self._state.previous_hypothesis = current_words
            
            return [], current_words, disagreement_flag
    
    def get_context_buffer(self) -> str:
        return " ".join(list(self._state.context_buffer))
    
    def reset(self) -> None:
        logger.info("Resetting LocalAgreementTracker state")
        self._state = AgreementState(
            previous_hypothesis=[],
            locked_segments=[],
            context_buffer=deque(maxlen=self._config.context_size),
            agreement_counter=0,
            last_locked_end_time=0.0
        )

