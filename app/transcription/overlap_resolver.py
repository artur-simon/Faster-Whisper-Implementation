from typing import List, Set, Optional
from app.models import Word


def resolve_overlapping_words(
    previous_words: List[Word],
    current_words: List[Word],
    overlap_threshold: float = 0.5,
) -> List[Word]:
    if not previous_words:
        return current_words

    used_current: Set[int] = set()
    resolved: List[Word] = []

    for _, prev_word in enumerate(previous_words):
        best_match_idx = find_best_match(
            prev_word, current_words, used_current, overlap_threshold
        )

        if best_match_idx is not None:
            selected_word = select_better_word(prev_word, current_words[best_match_idx])
            resolved.append(selected_word)
            used_current.add(best_match_idx)
        else:
            resolved.append(prev_word)

    for curr_idx, current_word in enumerate(current_words):
        if curr_idx not in used_current and not is_in_overlap_region(
            current_word, previous_words
        ):
            resolved.append(current_word)

    resolved.sort(key=lambda w: w.start)
    return resolved


# Iterate through a list of candidates, searching for the one with the greatest overlap with target_word, greater than threshold.
# Ignores previously used candidates (used_indices).
# Returns the index of the best candidate, or None.
def find_best_match(
    target_word: Word, candidates: List[Word], used_indices: Set[int], threshold: float
) -> Optional[int]:

    best_match_idx = None
    best_overlap = 0

    for idx, candidate in enumerate(candidates):
        if idx in used_indices:
            continue

        overlap_ratio = calculate_overlap_ratio(target_word, candidate)

        if overlap_ratio > threshold and overlap_ratio > best_overlap:
            best_overlap = overlap_ratio
            best_match_idx = idx

    return best_match_idx


# The fraction of temporal overlap between two intervals normalized by the average duration of both.
def calculate_overlap_ratio(word1: Word, word2: Word) -> float:
    overlap = max(0, min(word1.end, word2.end) - max(word1.start, word2.start))
    avg_duration = ((word1.end - word1.start) + (word2.end - word2.start)) / 2
    return overlap / avg_duration if avg_duration > 0 else 0


def select_better_word(word1: Word, word2: Word) -> Word:
    return word1 if word1.probability > word2.probability else word2


# TODO não preciso verificar isso, só pegar o offset da sequencia,
# se a palavra tiver start menor que tamanho do offset, é provavel estar de overlap
# mas ta, depois disso ainda tenho que voltar pra ver se realmente teve overlap com palavras passadas,
# a nova palavra pode estar na região de overlap mas não ser.
def is_in_overlap_region(word: Word, overlap_words: List[Word]) -> bool:
    return any(word.start < overlap_word.end for overlap_word in overlap_words)


def adjust_word_timestamps(words: List[Word], offset: float) -> List[Word]:
    return [
        Word(
            text=w.text,
            start=w.start + offset,
            end=w.end + offset,
            probability=w.probability,
        )
        for w in words
    ]


def get_words_after_time(words: List[Word], start_time: float) -> List[Word]:
    return [w for w in words if w.start >= start_time]


def get_words_before_time(words: List[Word], end_time: float) -> List[Word]:
    return [w for w in words if w.start < end_time]
