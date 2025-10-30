import os
import logging
from dataclasses import dataclass
from typing import List
from app.models import TranscriptionConfig
from app.transcription.transcription_engine import TranscriptionEngine
from app.utils.document_writer import TranscriptionWriter
from app.utils.file_utils import find_audio_files

logger = logging.getLogger("app.transcription.batch_processor")


@dataclass
class BatchResult:
    total_files: int
    successful: int
    failed: int
    errors: List[str]


class BatchProcessor:
    def __init__(self, config: TranscriptionConfig):
        self._config = config
        self._engine = None

    def process_folder(self, folder_path: str, progress_callback=None) -> BatchResult:
        logger.info(f"Starting batch processing for folder: {folder_path}")
        audio_files = find_audio_files(folder_path)
        
        if not audio_files:
            logger.warning(f"No audio files found in folder: {folder_path}")
            return BatchResult(total_files=0, successful=0, failed=0, errors=[])
        
        logger.info(f"Found {len(audio_files)} audio files to process")
        
        if self._engine is None:
            self._engine = TranscriptionEngine(self._config)
        
        successful = 0
        failed = 0
        errors = []
        
        for idx, audio_path in enumerate(audio_files, 1):
            try:
                if progress_callback:
                    progress_callback(idx, len(audio_files), os.path.basename(audio_path))
                
                output_path = self._get_output_path(audio_path)
                self._transcribe_file(audio_path, output_path)
                successful += 1
                logger.info(f"Successfully transcribed: {audio_path} -> {output_path}")
            except Exception as e:
                failed += 1
                error_msg = f"{os.path.basename(audio_path)}: {str(e)}"
                errors.append(error_msg)
                logger.error(f"Failed to transcribe {audio_path}: {e}", exc_info=True)
        
        result = BatchResult(
            total_files=len(audio_files),
            successful=successful,
            failed=failed,
            errors=errors
        )
        
        logger.info(f"Batch processing completed: {successful} successful, {failed} failed")
        return result

    def _get_output_path(self, audio_path: str) -> str:
        base_name = os.path.splitext(audio_path)[0]
        return f"{base_name}.txt"

    def _transcribe_file(self, audio_path: str, output_path: str) -> None:
        if os.path.exists(output_path):
            os.remove(output_path)
        
        writer = TranscriptionWriter(output_path, self._config)
        
        segments = self._engine.transcribe_audio(audio_path)
        segment_count = 0
        for segment in segments:
            if segment.no_speech_prob < self._config.no_speech_threshold:
                writer.write_string_to_file(segment.text)
                writer.write_string_to_file("\n")
                segment_count += 1
        
        logger.debug(f"Transcribed {segment_count} segments from {audio_path}")

    def release(self) -> None:
        logger.info("Releasing batch processor resources")
        if self._engine:
            self._engine.release()
            self._engine = None
        logger.info("Batch processor resources released")

