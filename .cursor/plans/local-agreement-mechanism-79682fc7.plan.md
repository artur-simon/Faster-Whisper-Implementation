<!-- 79682fc7-359b-48b6-8a34-7da11525e586 0d9d8020-0e70-43ec-b9ba-6cfb314a921f -->
# Local Agreement Mechanism for Transcription Stabilization

## Overview

Implement a local agreement mechanism that stabilizes real-time transcription by comparing consecutive ASR hypotheses, locking agreed-upon segments, and maintaining rolling context. The existing simple overlap resolver remains available via configuration selection.

## Implementation Plan

### 1. Configuration Model Updates (`app/models.py`)

- Create `LocalAgreementConfig` dataclass containing:
- `agreement_count: int` (default: `2`) - consecutive agreements needed
- `edit_threshold: float` (default: `0.2`) - normalized edit distance threshold
- `confidence_threshold: float` (default: `0.8`) - confidence score threshold
- `min_words: int` (default: `3`) - minimum words to lock
- `context_size: int` (default: `100`) - rolling buffer size in words
- Add `transcription_algorithm: str` field to `TranscriptionConfig` (default: `'simple_overlap_resolve'`)
- Add `local_agreement_config: LocalAgreementConfig` field to `TranscriptionConfig` (default: `LocalAgreementConfig()`)

### 2. Config Manager Updates (`app/utils/config_manager.py`)

- Add `transcription_algorithm: 'simple_overlap_resolve'` to `DEFAULT_CONFIG`
- Add `local_agreement_config` nested dict with all local agreement defaults:
- `agreement_count: 2`
- `edit_threshold: 0.2`
- `confidence_threshold: 0.8`
- `min_words: 3`
- `context_size: 100`
- Update config loading/saving to handle nested structure (recursive dict handling)

### 3. UI Configuration Window (`app/ui/config_window.py`)

- Add new "Transcription" tab to the notebook (in addition to existing "Model" and "Audio" tabs)
- In the Transcription tab, add:
- Combobox for `transcription_algorithm` selection:
- Options: `['simple_overlap_resolve', 'local_agreement']`
- Default: `'simple_overlap_resolve'`
- When `local_agreement` is selected, show related parameters (nested under `local_agreement_config`):
- Agreement count (integer entry: `local_agreement_config.agreement_count`)
- Edit distance threshold (float entry: `local_agreement_config.edit_threshold`)
- Confidence threshold (float entry: `local_agreement_config.confidence_threshold`)
- Minimum words (integer entry: `local_agreement_config.min_words`)
- Context buffer size (integer entry: `local_agreement_config.context_size`)
- Update `_get_config_dict()` to properly serialize nested `local_agreement_config` structure
- Bind algorithm selection change to show/hide local agreement parameters dynamically

### 4. Local Agreement Module (`app/transcription/local_agreement.py`)

Create new module with:

#### 4.1 Data Structures

- `LockedSegment` dataclass: stores locked word span with start/end indices
- `AgreementState` dataclass: tracks current hypothesis history, locked segments, and rolling context

#### 4.2 Core Functions

- `compute_word_edit_distance(words1: List[Word], words2: List[Word]) -> float`: Normalized Levenshtein distance on word sequences
- `compute_confidence_similarity(words1: List[Word], words2: List[Word]) -> float`: Average confidence-weighted similarity
- `compare_hypotheses(prev_words: List[Word], curr_words: List[Word], config) -> tuple`: Returns (similarity_score, overlap_span_start, overlap_span_end)
- `find_agreement_span(prev_words: List[Word], curr_words: List[Word], config) -> Optional[tuple]`: Finds overlapping span that meets agreement criteria
- `lock_segment(words: List[Word], start_idx: int, end_idx: int, config) -> LockedSegment`: Creates locked segment
- `get_words_beyond_lock(words: List[Word], last_locked_end: int) -> List[Word]`: Returns words after last locked boundary
- `rollback_to_last_lock(all_words: List[Word], locked_segments: List[LockedSegment]) -> List[Word]`: Returns words starting from last locked boundary

#### 4.3 Main Class: `LocalAgreementTracker`

- `__init__(config: LocalAgreementConfig)`: Initialize with `LocalAgreementConfig` instance
- `process_hypothesis(current_words: List[Word]) -> tuple`: Main processing function
- Compare with previous hypothesis
- Check for agreement convergence (using `config.edit_threshold`, `config.confidence_threshold`, `config.agreement_count`, `config.min_words`)
- Update locked segments
- Return (locked_words_to_emit, remaining_words_for_next_chunk, disagreement_flag)
- `get_context_buffer() -> str`: Returns rolling context string (last `config.context_size` words from locked segments)
- `reset()`: Reset state (for new transcription session)

### 5. Orchestrator Updates (`app/transcription/orchestrator.py`)

- Modify `_handle_transcription_result()` to check `config.transcription_algorithm`
- If `'simple_overlap_resolve'`: Use existing logic (current implementation)
- If `'local_agreement'`: 
- Initialize `LocalAgreementTracker` if not exists (pass `config.local_agreement_config`)
- Call `process_hypothesis()` with current words
- Handle locked words emission (immediate write)
- Update `previous_overlap_words` with remaining words
- Update `accumulated_text` from context buffer
- Handle disagreement: mark but wait for next chunk

### 6. Integration Points

- `TranscriptionState` in orchestrator: Add optional `agreement_tracker` field
- Context prompt: When using local agreement, use `get_context_buffer()` instead of `accumulated_text`
- Word writing: Locked segments written immediately via `writer.write_words()`
- Disagreement handling: Store disagreement flag, wait for next chunk before rollback decision

## Key Design Decisions

- Both algorithms coexist; selection via `transcription_algorithm` config field
- Locked segments maintained in separate data structure
- Rolling context buffer limits future context while preserving agreements
- Immediate emission of locked text for streaming output
- Disagreement marked but deferred to next chunk for decision
- All local agreement configs nested in `LocalAgreementConfig` object for better organization
- No `enabled` flag in `LocalAgreementConfig` - algorithm selection done via `transcription_algorithm`

## Testing Considerations

- Unit tests for similarity computation functions
- Unit tests for agreement detection logic
- Integration tests for orchestrator with both algorithms
- Verify existing overlap resolver still works unchanged

### To-dos

- [ ] Add new config fields to TranscriptionConfig dataclass in app/models.py
- [ ] Add default values for new config fields in ConfigManager.DEFAULT_CONFIG
- [ ] Create app/transcription/local_agreement.py with similarity computation, agreement tracking, and locked segment management
- [ ] Modify TranscriptionOrchestrator to support algorithm selection and integrate LocalAgreementTracker
- [ ] Add transcription algorithm selection and local agreement parameters to config window UI
- [ ] Verify existing simple_overlap_resolve logic remains unchanged and functional