import json
from typing import List, Dict, Any
from dataclasses import asdict
from pathlib import Path

from tests.simulation.transcription_test_harness import TranscriptionTestResult, TranscriptionEvent


class TranscriptionResultAnalyzer:
    
    @staticmethod
    def generate_full_report(result: TranscriptionTestResult) -> Dict[str, Any]:
        return {
            "summary": TranscriptionResultAnalyzer.get_summary(result),
            "quality_metrics": TranscriptionResultAnalyzer.get_quality_metrics(result),
            "timing_analysis": TranscriptionResultAnalyzer.get_timing_analysis(result),
            "word_details": TranscriptionResultAnalyzer.get_word_details(result),
            "event_analysis": TranscriptionResultAnalyzer.get_event_analysis(result)
        }
    
    @staticmethod
    def get_summary(result: TranscriptionTestResult) -> Dict[str, Any]:
        return {
            "total_words": len(result.all_words),
            "total_chunks": result.total_chunks,
            "total_duration": result.total_duration,
            "final_text": result.final_text,
            "text_length": len(result.final_text),
            "avg_words_per_chunk": len(result.all_words) / max(result.total_chunks, 1)
        }
    
    @staticmethod
    def get_quality_metrics(result: TranscriptionTestResult) -> Dict[str, Any]:
        overlap_issues = result.find_overlap_issues()
        gaps = result.find_timing_gaps(threshold=1.0)
        
        if result.all_words:
            avg_confidence = sum(w.probability for w in result.all_words) / len(result.all_words)
            min_confidence = min(w.probability for w in result.all_words)
            max_confidence = max(w.probability for w in result.all_words)
            low_confidence_words = [w for w in result.all_words if w.probability < 0.5]
        else:
            avg_confidence = 0
            min_confidence = 0
            max_confidence = 0
            low_confidence_words = []
        
        return {
            "overlap_count": len(overlap_issues),
            "gap_count": len(gaps),
            "avg_confidence": avg_confidence,
            "min_confidence": min_confidence,
            "max_confidence": max_confidence,
            "low_confidence_word_count": len(low_confidence_words),
            "low_confidence_words": [
                {
                    "text": w.text,
                    "confidence": w.probability,
                    "time": (w.start, w.end)
                }
                for w in low_confidence_words[:20]
            ],
            "overlap_details": overlap_issues[:10],
            "gap_details": gaps[:10]
        }
    
    @staticmethod
    def get_timing_analysis(result: TranscriptionTestResult) -> Dict[str, Any]:
        if not result.all_words:
            return {
                "total_audio_duration": 0,
                "first_word_time": 0,
                "last_word_time": 0,
                "avg_word_duration": 0,
                "timeline": []
            }
        
        first_word_time = result.all_words[0].start
        last_word_time = result.all_words[-1].end
        total_audio_duration = last_word_time - first_word_time
        
        word_durations = [w.end - w.start for w in result.all_words]
        avg_word_duration = sum(word_durations) / len(word_durations)
        
        timeline_entries = []
        for i, word in enumerate(result.all_words[:100]):
            timeline_entries.append({
                "index": i,
                "word": word.text,
                "start": word.start,
                "end": word.end,
                "duration": word.end - word.start,
                "confidence": word.probability
            })
        
        return {
            "total_audio_duration": total_audio_duration,
            "first_word_time": first_word_time,
            "last_word_time": last_word_time,
            "avg_word_duration": avg_word_duration,
            "timeline": timeline_entries
        }
    
    @staticmethod
    def get_word_details(result: TranscriptionTestResult) -> Dict[str, Any]:
        if not result.all_words:
            return {
                "word_count": 0,
                "unique_words": 0,
                "word_frequency": {},
                "longest_words": [],
                "shortest_words": []
            }
        
        word_texts = [w.text.strip() for w in result.all_words]
        word_frequency = {}
        for text in word_texts:
            word_frequency[text] = word_frequency.get(text, 0) + 1
        
        sorted_by_duration = sorted(result.all_words, key=lambda w: w.end - w.start, reverse=True)
        
        return {
            "word_count": len(result.all_words),
            "unique_words": len(set(word_texts)),
            "word_frequency": dict(sorted(word_frequency.items(), key=lambda x: x[1], reverse=True)[:20]),
            "longest_words": [
                {
                    "text": w.text,
                    "duration": w.end - w.start,
                    "time": (w.start, w.end)
                }
                for w in sorted_by_duration[:10]
            ],
            "shortest_words": [
                {
                    "text": w.text,
                    "duration": w.end - w.start,
                    "time": (w.start, w.end)
                }
                for w in sorted_by_duration[-10:]
            ]
        }
    
    @staticmethod
    def get_event_analysis(result: TranscriptionTestResult) -> Dict[str, Any]:
        event_types = {}
        for event in result.events:
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
        
        chunk_events = [e for e in result.events if e.event_type == "chunk_start"]
        words_written_events = [e for e in result.events if e.event_type == "words_written"]
        
        words_per_event = []
        for event in words_written_events:
            words_per_event.append(len(event.words))
        
        avg_words_per_write = sum(words_per_event) / len(words_per_event) if words_per_event else 0
        
        return {
            "total_events": len(result.events),
            "event_type_counts": event_types,
            "chunk_count": len(chunk_events),
            "write_count": len(words_written_events),
            "avg_words_per_write": avg_words_per_write,
            "words_per_write_distribution": {
                "min": min(words_per_event) if words_per_event else 0,
                "max": max(words_per_event) if words_per_event else 0,
                "avg": avg_words_per_write
            }
        }
    
    @staticmethod
    def save_report_to_file(result: TranscriptionTestResult, output_path: str) -> None:
        report = TranscriptionResultAnalyzer.generate_full_report(result)
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def print_summary(result: TranscriptionTestResult) -> None:
        report = TranscriptionResultAnalyzer.generate_full_report(result)
        
        print("\n" + "="*80)
        print("TRANSCRIPTION TEST REPORT")
        print("="*80)
        
        print("\n[SUMMARY]")
        for key, value in report["summary"].items():
            if key != "final_text":
                print(f"  {key}: {value}")
        
        print("\n[QUALITY METRICS]")
        qm = report["quality_metrics"]
        print(f"  Overlap issues: {qm['overlap_count']}")
        print(f"  Timing gaps: {qm['gap_count']}")
        print(f"  Average confidence: {qm['avg_confidence']:.3f}")
        print(f"  Low confidence words: {qm['low_confidence_word_count']}")
        
        if qm['overlap_count'] > 0:
            print("\n  First overlap issues:")
            for issue in qm['overlap_details'][:3]:
                print(f"    '{issue['word1']}' overlaps '{issue['word2']}' by {issue['overlap_duration']:.3f}s")
        
        if qm['gap_count'] > 0:
            print("\n  First timing gaps:")
            for gap in qm['gap_details'][:3]:
                print(f"    {gap['gap_duration']:.2f}s gap after '{gap['after_word']}'")
        
        print("\n[TIMING]")
        ta = report["timing_analysis"]
        print(f"  Total audio duration: {ta['total_audio_duration']:.2f}s")
        print(f"  Average word duration: {ta['avg_word_duration']:.3f}s")
        print(f"  First word at: {ta['first_word_time']:.2f}s")
        print(f"  Last word at: {ta['last_word_time']:.2f}s")
        
        print("\n[TRANSCRIPTION]")
        print(f"  {report['summary']['final_text'][:200]}...")
        
        print("\n" + "="*80 + "\n")


class TranscriptionComparator:
    
    @staticmethod
    def compare_results(result1: TranscriptionTestResult, result2: TranscriptionTestResult) -> Dict[str, Any]:
        return {
            "word_count_diff": len(result2.all_words) - len(result1.all_words),
            "chunk_count_diff": result2.total_chunks - result1.total_chunks,
            "text_similarity": TranscriptionComparator._calculate_text_similarity(
                result1.final_text, 
                result2.final_text
            ),
            "overlap_diff": len(result2.find_overlap_issues()) - len(result1.find_overlap_issues()),
            "result1_summary": TranscriptionResultAnalyzer.get_summary(result1),
            "result2_summary": TranscriptionResultAnalyzer.get_summary(result2)
        }
    
    @staticmethod
    def _calculate_text_similarity(text1: str, text2: str) -> float:
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)

