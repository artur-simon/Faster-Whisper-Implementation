import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.models import TranscriptionConfig
from tests.simulation.transcription_test_harness import TranscriptionTestHarness
from tests.simulation.result_analyzer import TranscriptionResultAnalyzer

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def run_manual_test(audio_file: str, output_report: str = None):
    
    config = TranscriptionConfig(
        model_size="turbo",
        device="cuda",
        compute_type="float32",
        sample_rate=16000,
        chunk_duration=3.0,
        overlap_duration=2.0,
        vad_filter=True,
        no_speech_threshold=0.6,
        language="pt",
        mic_id=0
    )
    
    logger.info(f"Testing with audio file: {audio_file}")
    logger.info(f"Config: chunk={config.chunk_duration}s, overlap={config.overlap_duration}s")
    
    harness = TranscriptionTestHarness(config)
    
    def log_event(event):
        if event.event_type == "chunk_start":
            logger.info(f"Chunk started")
        elif event.event_type == "words_written":
            text = " ".join(w.text for w in event.words)
            logger.info(f"Chunk - Written: {text}")
    
    harness.set_event_callback(log_event)
    
    result = harness.run_test(audio_file, realtime_speed=1.0)
    
    TranscriptionResultAnalyzer.print_summary(result)
    
    overlap_issues = result.find_overlap_issues()
    if overlap_issues:
        logger.warning(f"\nFound {len(overlap_issues)} overlap issues:")
        for issue in overlap_issues[:5]:
            logger.warning(
                f"  '{issue['word1']}' ({issue['time1'][0]:.2f}-{issue['time1'][1]:.2f}) "
                f"overlaps '{issue['word2']}' ({issue['time2'][0]:.2f}-{issue['time2'][1]:.2f})"
            )
    
    gaps = result.find_timing_gaps(threshold=2.0)
    if gaps:
        logger.info(f"\nFound {len(gaps)} large timing gaps (>2s):")
        for gap in gaps[:5]:
            logger.info(
                f"  {gap['gap_duration']:.2f}s gap after '{gap['after_word']}' "
                f"at {gap['position'][0]:.2f}s"
            )
    
    if output_report:
        TranscriptionResultAnalyzer.save_report_to_file(result, output_report)
        logger.info(f"\nFull report saved to: {output_report}")
    
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_manual_test.py <audio_file> [output_report.json]")
        print("\nExample:")
        print("  python run_manual_test.py tests/fixtures/sample.wav report.json")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    output_report = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(audio_file).exists():
        print(f"Error: Audio file not found: {audio_file}")
        sys.exit(1)
    
    run_manual_test(audio_file, output_report)

