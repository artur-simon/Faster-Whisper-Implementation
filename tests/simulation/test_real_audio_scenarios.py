import pytest
import logging
from pathlib import Path

from app.models import TranscriptionConfig
from tests.simulation.transcription_test_harness import TranscriptionTestHarness

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def base_config():
    return TranscriptionConfig(
        model_size="turbo",
        device="cuda",
        compute_type="float32",
        sample_rate=16000,
        chunk_duration=5.0,
        overlap_duration=1.0,
        vad_filter=True,
        no_speech_threshold=0.6,
        language="auto",
        mic_id=0
    )


@pytest.fixture
def sample_audio_file():
    fixtures_dir = Path(__file__).parent.parent / "fixtures"
    audio_file = fixtures_dir / "sample.wav"
    if not audio_file.exists():
        pytest.skip(f"Audio file not found: {audio_file}")
    return str(audio_file)


def test_basic_transcription(base_config, sample_audio_file):
    harness = TranscriptionTestHarness(base_config)
    
    result = harness.run_test(sample_audio_file, realtime_speed=50.0)
    
    logger.info(f"\n{'='*60}")
    logger.info("TRANSCRIPTION RESULT:")
    logger.info(f"{'='*60}")
    logger.info(f"Total words: {len(result.all_words)}")
    logger.info(f"Total chunks processed: {result.total_chunks}")
    logger.info(f"Test duration: {result.total_duration:.2f}s")
    logger.info(f"\nFinal text:\n{result.final_text}")
    logger.info(f"{'='*60}\n")
    
    assert len(result.all_words) > 0, "No words transcribed"
    assert result.total_chunks > 0, "No chunks processed"


def test_overlap_resolution(base_config, sample_audio_file):
    harness = TranscriptionTestHarness(base_config)
    
    result = harness.run_test(sample_audio_file, realtime_speed=50.0)
    
    overlap_issues = result.find_overlap_issues()
    
    logger.info(f"\n{'='*60}")
    logger.info("OVERLAP ANALYSIS:")
    logger.info(f"{'='*60}")
    logger.info(f"Total words: {len(result.all_words)}")
    logger.info(f"Overlap issues found: {len(overlap_issues)}")
    
    if overlap_issues:
        logger.warning("Overlap issues detected:")
        for issue in overlap_issues[:10]:
            logger.warning(
                f"  '{issue['word1']}' ({issue['time1'][0]:.2f}-{issue['time1'][1]:.2f}) "
                f"overlaps '{issue['word2']}' ({issue['time2'][0]:.2f}-{issue['time2'][1]:.2f}) "
                f"by {issue['overlap_duration']:.3f}s"
            )
    else:
        logger.info("No overlap issues detected")
    
    logger.info(f"{'='*60}\n")
    
    assert len(overlap_issues) == 0, f"Found {len(overlap_issues)} overlap issues"


def test_timing_consistency(base_config, sample_audio_file):
    harness = TranscriptionTestHarness(base_config)
    
    result = harness.run_test(sample_audio_file, realtime_speed=50.0)
    
    gaps = result.find_timing_gaps(threshold=2.0)
    
    logger.info(f"\n{'='*60}")
    logger.info("TIMING ANALYSIS:")
    logger.info(f"{'='*60}")
    logger.info(f"Total words: {len(result.all_words)}")
    logger.info(f"Large gaps (>2s) found: {len(gaps)}")
    
    if gaps:
        logger.info("Large gaps detected:")
        for gap in gaps[:10]:
            logger.info(
                f"  {gap['gap_duration']:.2f}s gap after '{gap['after_word']}' "
                f"before '{gap['before_word']}' at {gap['position'][0]:.2f}s"
            )
    
    logger.info(f"{'='*60}\n")
    
    for i, word in enumerate(result.all_words):
        assert word.start <= word.end, f"Word {i} has invalid timing: {word.text} ({word.start} > {word.end})"
    
    if len(result.all_words) > 1:
        assert result.all_words[0].start <= result.all_words[-1].end, "Words are not in chronological order"


def test_event_tracking(base_config, sample_audio_file):
    harness = TranscriptionTestHarness(base_config)
    
    event_log = []
    
    def event_callback(event):
        event_log.append(event)
    
    harness.set_event_callback(event_callback)
    result = harness.run_test(sample_audio_file, realtime_speed=50.0)
    
    logger.info(f"\n{'='*60}")
    logger.info("EVENT TRACKING:")
    logger.info(f"{'='*60}")
    logger.info(f"Total events: {len(event_log)}")
    
    event_types = {}
    for event in event_log:
        event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
    
    logger.info("Event breakdown:")
    for event_type, count in event_types.items():
        logger.info(f"  {event_type}: {count}")
    
    logger.info(f"{'='*60}\n")
    
    assert len(event_log) > 0, "No events captured"
    assert "chunk_start" in event_types, "Missing chunk_start events"
    assert "words_written" in event_types, "Missing words_written events"


def test_different_chunk_sizes(base_config, sample_audio_file):
    configs_to_test = [
        {"chunk_duration": 3.0, "overlap_duration": 0.5},
        {"chunk_duration": 5.0, "overlap_duration": 1.0},
        {"chunk_duration": 10.0, "overlap_duration": 2.0},
    ]
    
    logger.info(f"\n{'='*60}")
    logger.info("TESTING DIFFERENT CHUNK SIZES:")
    logger.info(f"{'='*60}")
    
    results = []
    
    for config_override in configs_to_test:
        test_config = TranscriptionConfig(
            model_size=base_config.model_size,
            device=base_config.device,
            compute_type=base_config.compute_type,
            sample_rate=base_config.sample_rate,
            chunk_duration=config_override["chunk_duration"],
            overlap_duration=config_override["overlap_duration"],
            vad_filter=base_config.vad_filter,
            no_speech_threshold=base_config.no_speech_threshold,
            language=base_config.language,
            mic_id=base_config.mic_id
        )
        
        harness = TranscriptionTestHarness(test_config)
        result = harness.run_test(sample_audio_file, realtime_speed=50.0)
        
        overlap_issues = result.find_overlap_issues()
        
        logger.info(f"\nChunk: {config_override['chunk_duration']}s, Overlap: {config_override['overlap_duration']}s")
        logger.info(f"  Words: {len(result.all_words)}")
        logger.info(f"  Chunks: {result.total_chunks}")
        logger.info(f"  Overlaps: {len(overlap_issues)}")
        logger.info(f"  Text: {result.final_text[:100]}...")
        
        results.append({
            "config": config_override,
            "result": result,
            "overlap_count": len(overlap_issues)
        })
    
    logger.info(f"{'='*60}\n")
    
    for r in results:
        assert r["overlap_count"] == 0, \
            f"Config {r['config']} produced {r['overlap_count']} overlaps"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

