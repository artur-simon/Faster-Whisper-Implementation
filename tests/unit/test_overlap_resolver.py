import pytest
from app.models import Word
from app.transcription.overlap_resolver import (
    resolve_overlapping_words,
    find_best_match,
    calculate_overlap_ratio,
    select_better_word,
    is_in_overlap_region,
    adjust_word_timestamps,
    get_words_after_time,
    get_words_before_time
)


class TestOverlapResolver:
    @pytest.fixture
    def sample_words(self):
        return [
            Word(text=" hello", start=0.0, end=1.0, probability=0.9),
            Word(text=" world", start=1.0, end=2.0, probability=0.85),
            Word(text=" test", start=2.0, end=3.0, probability=0.95)
        ]
    
    def test_resolve_overlapping_words_no_previous(self, sample_words):
        result = resolve_overlapping_words([], sample_words)
        assert result == sample_words
    
    def test_resolve_overlapping_words_with_overlap(self):
        previous_words = [
            Word(text=" hello", start=0.0, end=1.0, probability=0.8),
        ]
        current_words = [
            Word(text=" hello", start=0.4, end=1.4, probability=0.9),
            Word(text=" world", start=1.5, end=2.5, probability=0.85),
        ]
        
        result = resolve_overlapping_words(previous_words, current_words)
        assert len(result) == 2
        assert result[0].probability == 0.9
    
    def test_find_best_match_no_overlap(self):
        target = Word(text=" hello", start=1.0, end=2.0, probability=0.9)
        candidates = [
            Word(text=" world", start=0.0, end=1.0, probability=0.85),
            Word(text=" world", start=2.0, end=3.0, probability=0.85),
        ]
        
        result = find_best_match(target, candidates, set(), threshold=0.5)
        assert result is None
    
    def test_find_best_match_with_overlap(self):
        target = Word(text=" hello", start=0.0, end=1.0, probability=0.9)
        candidates = [
            Word(text=" hello", start=0.4, end=1.4, probability=0.95),
            Word(text=" world", start=5.0, end=6.0, probability=0.85),
        ]
        
        result = find_best_match(target, candidates, set(), threshold=0.5)
        assert result == 0
    
    def test_calculate_overlap_ratio_no_overlap(self):
        word1 = Word(text=" hello", start=0.0, end=1.0, probability=0.9)
        word2 = Word(text=" world", start=2.0, end=3.0, probability=0.85)
        
        ratio = calculate_overlap_ratio(word1, word2)
        assert ratio == 0.0
    
    def test_calculate_overlap_ratio_full_overlap(self):
        word1 = Word(text=" hello", start=0.0, end=1.0, probability=0.9)
        word2 = Word(text=" hello", start=0.0, end=1.0, probability=0.95)
        
        ratio = calculate_overlap_ratio(word1, word2)
        assert ratio == 1.0
    
    def test_calculate_overlap_ratio_partial_overlap(self):
        word1 = Word(text=" hello", start=0.0, end=1.0, probability=0.9)
        word2 = Word(text=" hello", start=0.5, end=1.5, probability=0.95)
        
        ratio = calculate_overlap_ratio(word1, word2)
        assert 0.0 < ratio < 1.0
    
    def test_select_better_word_higher_probability(self):
        word1 = Word(text=" hello", start=0.0, end=1.0, probability=0.9)
        word2 = Word(text=" hello", start=0.0, end=1.0, probability=0.95)
        
        result = select_better_word(word1, word2)
        assert result == word2
    
    def test_select_better_word_lower_probability(self):
        word1 = Word(text=" hello", start=0.0, end=1.0, probability=0.95)
        word2 = Word(text=" hello", start=0.0, end=1.0, probability=0.85)
        
        result = select_better_word(word1, word2)
        assert result == word1
    
    def test_is_in_overlap_region_true(self):
        word = Word(text=" test", start=0.5, end=1.5, probability=0.9)
        overlap_words = [
            Word(text=" hello", start=0.0, end=1.0, probability=0.9),
        ]
        
        result = is_in_overlap_region(word, overlap_words)
        assert result is True
    
    def test_is_in_overlap_region_false(self):
        word = Word(text=" test", start=2.0, end=3.0, probability=0.9)
        overlap_words = [
            Word(text=" hello", start=0.0, end=1.0, probability=0.9),
        ]
        
        result = is_in_overlap_region(word, overlap_words)
        assert result is False
    
    def test_adjust_word_timestamps(self, sample_words):
        offset = 5.0
        result = adjust_word_timestamps(sample_words, offset)
        
        assert len(result) == len(sample_words)
        assert result[0].start == 5.0
        assert result[0].end == 6.0
        assert result[1].start == 6.0
        assert result[1].end == 7.0
    
    def test_get_words_after_time(self, sample_words):
        result = get_words_after_time(sample_words, 1.5)
        
        assert len(result) == 2
        assert result[0].text == " world"
        assert result[1].text == " test"
    
    def test_get_words_before_time(self, sample_words):
        result = get_words_before_time(sample_words, 1.5)
        
        assert len(result) == 1
        assert result[0].text == " hello"
    

