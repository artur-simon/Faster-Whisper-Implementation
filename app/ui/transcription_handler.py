import os
import threading
import logging
from tkinter import filedialog, messagebox
import tkinter as tk
from app.models import TranscriptionConfig
from app.transcription.transcription_controller import TranscriptionController
from app.transcription.batch_processor import BatchProcessor
from app.utils.file_utils import find_audio_files

logger = logging.getLogger("app.ui.transcription_handler")


class TranscriptionHandler:
    def __init__(self, config_manager, state_manager, status_callback, tray_icon_callback, button_update_callback):
        self.config_manager = config_manager
        self.state_manager = state_manager
        self.status_callback = status_callback
        self.tray_icon_callback = tray_icon_callback
        self.button_update_callback = button_update_callback
        
        self.transcriber = None
        self.is_running = False

    def toggle_model(self, root):
        if self.transcriber is None:
            try:
                config_dict = self.config_manager.get_config_dict()
                config = TranscriptionConfig.from_dict(config_dict)
                
                logger.info(f"Activating model: {config.model_size} on {config.device}")
                self.transcriber = TranscriptionController(config)
                logger.info("Model activated successfully")
                
                self.button_update_callback(model_button_text="Release Model", start_button_state=tk.NORMAL)
                is_model_activated = True
            except Exception as e:
                logger.error(f"Failed to initialize model: {e}", exc_info=True)
                messagebox.showerror("Error", "Failed to initialize model, refer to logs for more details.")
                is_model_activated = False
        else:
            logger.info("Releasing model")
            self.transcriber.shutdown()
            self.transcriber = None
            
            logger.info("Model released")
            self.button_update_callback(model_button_text="Activate Model", start_button_state=tk.DISABLED)
            is_model_activated = False
        
        status_text = "Ready" if is_model_activated else "Off"
        self.status_callback(status_text)
        self.tray_icon_callback("READY" if is_model_activated else "OFF")
        
        return is_model_activated

    def toggle_recording(self, root):
        if not self.is_running:
            logger.info("Starting recording")
            self.is_running = True
            transcription_file = self.state_manager.get_current_transcription_file()
            self.transcriber.run(output_file=transcription_file)
            
            self.status_callback("Recording")
            self.button_update_callback(start_button_text="Stop Recording")
            self.tray_icon_callback("RECORDING")
        else:
            if self.transcriber:
                self.is_running = False
                self.transcriber.stop()
                logger.info("Stopping recording")
            
            self.status_callback("Ready")
            self.button_update_callback(start_button_text="Start Recording")
            self.tray_icon_callback("READY")

    def select_audio_file(self, root):
        filetypes = (("MP3 files", "*.mp3"), ("WAV files", "*.wav"), ("All files", "*.*"))
        filepath = filedialog.askopenfilename(title="Select an audio file", filetypes=filetypes)
        if filepath and self.transcriber:
            logger.info(f"Selected audio file for transcription: {filepath}")
            self.status_callback("Transcribing file")
            
            def transcribe_file():
                try:
                    transcription_file = self.state_manager.get_current_transcription_file()
                    self.transcriber.transcribe_audio_file(filepath, transcription_file)
                    logger.info(f"File transcription completed: {filepath}")
                    root.after(0, lambda: self.status_callback("Transcription completed"))
                except Exception as e:
                    logger.error(f"File transcription failed: {e}", exc_info=True)
                    root.after(0, lambda: messagebox.showerror("Error", "File transcription failed, refer to logs for more details."))
                    root.after(0, lambda: self.status_callback("Error"))
            
            threading.Thread(target=transcribe_file, name="TranscriptionHandler - transcribe_audio_file").start()

    def batch_process_folder(self, root):
        if not self.transcriber:
            messagebox.showwarning("Model Not Activated", "Please activate the model first.")
            return
        
        folder_path = filedialog.askdirectory(title="Select folder with audio files")
        if not folder_path:
            return
        
        audio_files = find_audio_files(folder_path)
        
        if not audio_files:
            messagebox.showinfo("No Audio Files", f"No supported audio files (.mp3, .wav) found in:\n{folder_path}")
            return
        
        confirm_msg = f"Found {len(audio_files)} audio file(s) in:\n{folder_path}\n\nProceed with batch transcription?"
        if not messagebox.askyesno("Confirm Batch Processing", confirm_msg):
            return
        
        config_dict = self.config_manager.get_config_dict()
        config = TranscriptionConfig.from_dict(config_dict)
        
        def process_batch():
            processor = BatchProcessor(config)
            try:
                def progress_callback(current, total, filename):
                    status_text = f"Batch: {current}/{total} - {filename}"
                    def update_status(text=status_text):
                        self.status_callback(text)
                    root.after(0, update_status)
                
                result = processor.process_folder(folder_path, progress_callback)
                processor.release()
                
                def show_result():
                    if result.failed == 0:
                        messagebox.showinfo(
                            "Batch Processing Complete",
                            f"Successfully transcribed {result.successful} file(s)."
                        )
                        self.status_callback("Batch processing completed")
                    else:
                        error_summary = "\n".join(result.errors[:10])
                        if len(result.errors) > 10:
                            error_summary += f"\n... and {len(result.errors) - 10} more error(s)"
                        
                        messagebox.showwarning(
                            "Batch Processing Complete",
                            f"Completed: {result.successful} successful, {result.failed} failed\n\n"
                            f"Errors:\n{error_summary}"
                        )
                        self.status_callback(f"Batch: {result.successful}/{result.total_files} completed")
                
                root.after(0, show_result)
            except Exception as e:
                logger.error(f"Batch processing failed: {e}", exc_info=True)
                error_msg = str(e)
                def show_error(msg=error_msg):
                    messagebox.showerror(
                        "Error",
                        f"Batch processing failed: {msg}\nRefer to logs for more details."
                    )
                    self.status_callback("Batch processing failed")
                root.after(0, show_error)
        
        self.status_callback("Batch processing: Starting...")
        threading.Thread(target=process_batch, name="TranscriptionHandler - batch_process_folder").start()

    def shutdown(self):
        if self.transcriber:
            self.transcriber.shutdown()
            self.transcriber = None

    def update_config(self, **kwargs):
        if self.transcriber:
            self.transcriber.update_input_config(**kwargs)

