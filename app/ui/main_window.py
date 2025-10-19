from time import sleep
from tkinter import filedialog, messagebox
from PIL import Image, ImageDraw
import tkinter as tk
import pyperclip
import pystray
import threading
import logging

from app.transcription.transcription_controller import TranscriptionController
from app.ui.live_text_view import LiveTextViewer
from app.ui.logging_window import LoggingWindow
from app.audio.audio_capture import AudioCapture
from app.utils.config_manager import ConfigManager
from app.utils.logging_manager import LoggingManager

logger = logging.getLogger("app.ui.main_window")


class MainWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("WispLive")
        self.root.iconbitmap(default=self.get_icon_path())
        self.root.protocol("WM_DELETE_WINDOW", self.exit_app)
        
        logger.info("Initializing WispLive main window")
        self.config_manager = ConfigManager()
        config = self.config_manager.load_config()
        logger.debug(f"Loaded configuration: {config}")
        
        self.configure_menu_bar(root, config)
        
        configs_frame = tk.Frame(root)
        configs_frame.pack(pady=5, padx=5, fill='x')
        self.configure_model_toolbar(configs_frame, config)
        self.configure_input_toolbar(configs_frame, config)
        
        self.configure_status_toolbar(root, config)
        
        self.text_viewer = LiveTextViewer(root, "transcription.txt")
        
        self.transcriber = None
        self.is_running = False
        self.tray_icon = None
        
        self.logging_manager = LoggingManager.get_instance()
        self.logging_window = None
    
    
    def configure_menu_bar(self, root, config):
        
        menubar = tk.Menu(root)
        
        # ===== File Menu =====
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Audio File", command=lambda: filedialog.askopenfilename(), state=tk.DISABLED)
        file_menu.add_command(label="Open Recent", command=lambda: print("Open recent files"), state=tk.DISABLED)
        file_menu.add_command(label="Batch Process Folder", command=lambda: filedialog.askdirectory(), state=tk.DISABLED)
        file_menu.add_command(label="Import Session", command=lambda: filedialog.askopenfilename(), state=tk.DISABLED)
        file_menu.add_separator()
        file_menu.add_command(label="Export as TXT", command=lambda: print("Export TXT"), state=tk.DISABLED)
        file_menu.add_command(label="Export as SRT", command=lambda: print("Export SRT"), state=tk.DISABLED)
        file_menu.add_command(label="Export as VTT", command=lambda: print("Export VTT"), state=tk.DISABLED)
        file_menu.add_command(label="Export as DOCX", command=lambda: print("Export DOCX"), state=tk.DISABLED)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.exit_app)
        menubar.add_cascade(label="File", menu=file_menu)

        # ===== Edit Menu =====
        edit_menu = tk.Menu(menubar, tearoff=0)
        edit_menu.add_command(label="Find / Replace", command=lambda: print("Find Replace"), state=tk.DISABLED)
        edit_menu.add_command(label="Clear Transcript", command=lambda: print("Clear Transcript"), state=tk.DISABLED)
        edit_menu.add_command(label="Undo", command=lambda: print("Undo"), state=tk.DISABLED)
        edit_menu.add_command(label="Redo", command=lambda: print("Redo"), state=tk.DISABLED)
        edit_menu.add_separator()
        edit_menu.add_command(label="Timestamp Alignment", command=lambda: print("Align timestamps"), state=tk.DISABLED)
        edit_menu.add_command(label="Merge Paragraphs", command=lambda: print("Merge paragraphs"), state=tk.DISABLED)
        edit_menu.add_command(label="Split Paragraphs", command=lambda: print("Split paragraphs"), state=tk.DISABLED)
        menubar.add_cascade(label="Edit", menu=edit_menu)

        # ===== View Menu =====
        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_command(label="Hide to System tray", command=self.hide_window)
        view_menu.add_command(label="Toggle Waveform", command=lambda: print("Waveform toggle"), state=tk.DISABLED)
        view_menu.add_command(label="Show/Hide Timestamps", command=lambda: print("Timestamps toggle"), state=tk.DISABLED)
        view_menu.add_command(label="Word Confidence Heatmap", command=lambda: print("Confidence heatmap"), state=tk.DISABLED)
        view_menu.add_command(label="Real-Time Highlighting", command=lambda: print("Highlighting"), state=tk.DISABLED)
        view_menu.add_command(label="Toggle Dark Mode", state=tk.DISABLED)
        menubar.add_cascade(label="View", menu=view_menu)

        # ===== Tools Menu =====
        tools_menu = tk.Menu(menubar, tearoff=0)
        self.vad_filter_var = tk.BooleanVar(value=config.get('vad_filter', False))
        tools_menu.add_checkbutton(label="Use VAD filter", 
                                   command=lambda: self.on_config_value_change(vad_filter=self.vad_filter_var.get()), 
                                   variable=self.vad_filter_var)
        tools_menu.add_command(label="Language Auto-Detect", command=lambda: print("Auto-detect language"), state=tk.DISABLED)
        tools_menu.add_command(label="Speaker Diarization", command=lambda: print("Speaker diarization"), state=tk.DISABLED)
        tools_menu.add_command(label="Toggle Punctuation Restoration", command=lambda: print("Punctuation restoration"), state=tk.DISABLED)
        tools_menu.add_command(label="Batch Accuracy Stats", command=lambda: print("Batch stats"), state=tk.DISABLED)
        tools_menu.add_command(label="Custom Vocabulary", command=lambda: print("Custom vocab"), state=tk.DISABLED)
        menubar.add_cascade(label="Tools", menu=tools_menu)

        # ===== Settings Menu =====
        settings_menu = tk.Menu(menubar, tearoff=0)
        settings_menu.add_command(label="Model Parameters", command=lambda: print("Model params"), state=tk.DISABLED)
        settings_menu.add_command(label="Save Preset", command=self.save_config)
        settings_menu.add_command(label="Load Preset", command=lambda: print("Load preset"), state=tk.DISABLED)
        settings_menu.add_command(label="Audio Input Routing", command=lambda: print("Audio routing"), state=tk.DISABLED)
        settings_menu.add_command(label="Hotkeys", command=lambda: print("Hotkeys"), state=tk.DISABLED)
        
        logging_menu = tk.Menu(settings_menu, tearoff=0)
        logging_menu.add_command(label="Show Logging Console", command=self.show_logging_window)
        logging_menu.add_separator()
        logging_menu.add_command(label="Set Level: DEBUG", command=lambda: self.set_log_level("DEBUG"))
        logging_menu.add_command(label="Set Level: INFO", command=lambda: self.set_log_level("INFO"))
        logging_menu.add_command(label="Set Level: WARNING", command=lambda: self.set_log_level("WARNING"))
        logging_menu.add_command(label="Set Level: ERROR", command=lambda: self.set_log_level("ERROR"))
        settings_menu.add_cascade(label="Logging Verbosity", menu=logging_menu)
        
        menubar.add_cascade(label="Settings", menu=settings_menu)

        # ===== Help Menu =====
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=lambda: messagebox.showinfo("About", "WispLive Voice Transcriber"))
        help_menu.add_command(label="Model Info", command=lambda: print("Model info"), state=tk.DISABLED)
        help_menu.add_command(label="Benchmark Test", command=lambda: print("Benchmark"), state=tk.DISABLED)
        help_menu.add_command(label="Open Logs Folder", command=lambda: print("Logs folder"), state=tk.DISABLED)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        root.config(menu=menubar)
        
    
    def configure_model_toolbar(self, root, config):
        model_frame = tk.Frame(root)
        model_frame.pack(side="left", anchor="w", fill='x')
        
        self.toggle_model_button = tk.Button(model_frame, text="Activate Model", command=self.toggle_model)
        self.toggle_model_button.grid(row=0, column=0)
        
        self.model_size = tk.StringVar(value=config.get("model_size", "medium"))
        tk.OptionMenu(model_frame, self.model_size, "tiny", "base", "small", "medium", "large-v3", "turbo").grid(row=0, column=1)
        
        self.device_var = tk.StringVar(value=config.get("device", "cuda"))
        tk.OptionMenu(model_frame, self.device_var, "cpu", "cuda").grid(row=0, column=2)

        self.compute_type_var = tk.StringVar(value=config.get("compute_type", "int8"))
        tk.OptionMenu(model_frame, self.compute_type_var, "float32", "float16", "int8_float16", "int8").grid(row=0, column=3)

        self._model_frame = model_frame
    
    
    def configure_input_toolbar(self, root, config):
        config_frame = tk.Frame(root)
        config_frame.pack(side="right", anchor='e')
        
        self.language_var = tk.StringVar(value=config.get("language", "en"))
        self.language_var.trace_add("write", lambda *args:self.on_config_value_change(language=self.language_var.get()))
        tk.OptionMenu(config_frame, self.language_var, "pt", "en").grid(row=0, column=3)
        
        self.input_devices = AudioCapture.get_input_devices()
        self.device_index_map = {f"{d['index']}: {d['name']}": d['index'] for d in self.input_devices}
        
        mic_names = list(self.device_index_map.keys())
        selected_mic = "No devices"
        
        if mic_names:
            default_id = AudioCapture.get_default_input_device_id()
            selected_mic = next(
                (f"{d['index']}: {d['name']}" for d in self.input_devices if d["index"] == default_id),
                None
            )
                
        saved_mic_idx = config.get("mic_id")
        selected_mic = next(
            (name for name, idx in self.device_index_map.items() if idx == saved_mic_idx),
            selected_mic
        )
        
        self.microphone_var = tk.StringVar(value=selected_mic)
        self.microphone_var.trace_add(
            "write",
            lambda *args: self.on_config_value_change(
                mic_id=self.device_index_map.get(self.microphone_var.get())
            )
        )
        
        mic_menu = (
            tk.OptionMenu(config_frame, self.microphone_var, *mic_names) 
            if mic_names 
            else tk.OptionMenu(config_frame, self.microphone_var, "No devices")
        )
        
        mic_menu.grid(row=0, column=4)
    
    
    def configure_status_toolbar(self, root, config):
        button_frame = tk.Frame(root)
        button_frame.pack(pady=5, padx=5, fill='x')
        
        record_frame = tk.Frame(button_frame)
        record_frame.pack(side="left", anchor="w")
        self.start_button = tk.Button(record_frame, text="Start Recording", command=self.toggle_recording, state=tk.DISABLED)
        self.start_button.grid(row=0, column=0)
        
        self.copy_button = tk.Button(record_frame, text="Copy Text", command=self.copy_text)
        self.copy_button.grid(row=0, column=2)

        self.select_file_button = tk.Button(record_frame, text="Select audio file", command=self.select_audio_file)
        self.select_file_button.grid(row=0, column=3)
        
        self.should_paste_content_var = tk.BooleanVar(value=config.get("should_paste_content", "False"))
        self.should_paste_content_checkbutton = tk.Checkbutton(
            record_frame, 
            text='Paste transcription', 
            variable=self.should_paste_content_var,
            onvalue=1,
            offvalue=0,
            command=lambda:self.on_config_value_change(should_paste_content=self.should_paste_content_var.get())
        )
        self.should_paste_content_checkbutton.grid(row=0, column=4)
        
        status_frame = tk.Frame(button_frame)
        status_frame.pack(side="right", anchor="e")
        self.status_label = tk.Label(status_frame, text="Status: Stopped")
        self.status_label.grid(row=0, column=0)
                
    
    def on_config_value_change(self, **kwargs):
        if(self.transcriber):
            self.transcriber.update_input_config(**kwargs)
        
        
    def get_icon_path(self):
        import sys, os
        base_path = (
            sys._MEIPASS if getattr(sys, 'frozen', False) 
            else os.path.dirname(os.path.abspath(sys.argv[0]))
        )
        return os.path.join(base_path, "wisp.ico")
    
        
    def toggle_model(self):
        if self.transcriber is None:            
            try:
                logger.info(f"Activating model: {self.model_size.get()} on {self.device_var.get()}")
                self.transcriber = TranscriptionController(
                    device = self.device_var.get(),
                    compute_type = self.compute_type_var.get(),
                    model_size = self.model_size.get(),
                    language = self.language_var.get(),
                    mic_id = self.device_index_map.get(self.microphone_var.get()),
                    should_paste_content = self.should_paste_content_var.get()
                )
                
                logger.info("Model activated successfully")
                self.toggle_model_button.config(text="Release Model")
                self.start_button.config(state=tk.NORMAL)
                self.select_file_button.config(state=tk.NORMAL)
                for child in self._model_frame.winfo_children():
                    if isinstance(child, tk.OptionMenu):
                        child.config(state=tk.DISABLED)
            except Exception as e:
                logger.error(f"Failed to initialize model: {e}", exc_info=True)
                messagebox.showerror("Error", f"Failed to initialize model, refer to logs for more details.")
        else:
            logger.info("Releasing model")
            self.transcriber.shutdown()
            self.transcriber = None
            
            logger.info("Model released")
            self.toggle_model_button.config(text="Activate Model")
            self.start_button.config(state=tk.DISABLED)
            self.select_file_button.config(state=tk.DISABLED)
            for child in self._model_frame.winfo_children():
                if isinstance(child, tk.OptionMenu):
                    child.config(state=tk.NORMAL)


    def toggle_recording(self):
        if not self.is_running:
            logger.info("Starting recording")
            self.is_running = True
            self.transcriber.run(output_file="transcription.txt")
            self.status_label.config(text="Status: Recording...")
            self.start_button.config(text="Stop Recording")
        elif self.transcriber:
            logger.info("Stopping recording")
            self.is_running = False
            self.transcriber.stop()
            self.status_label.config(text="Status: Stopped")
            self.start_button.config(text="Start Recording")

        
    def select_audio_file(self):
        filetypes = (("MP3 files","*.mp3"), ("WAV files", "*.wav"), ("All files", "*.*"))
        filepath = filedialog.askopenfilename(title="Selecione um arquivo de áudio", filetypes=filetypes)
        if filepath and self.transcriber:
            logger.info(f"Selected audio file for transcription: {filepath}")
            self.status_label.config(text="Status: Transcribing file...")
            def transcribe_file():
                try:
                    self.transcriber.transcribe_audio_file(filepath, "transcription.txt")
                    logger.info(f"File transcription completed: {filepath}")
                    self.root.after(0, lambda: self.status_label.config(text="Status: Transcrição concluída"))
                except Exception as e:
                    logger.error(f"File transcription failed: {e}", exc_info=True)
                    messagebox.showerror("Error", f"File transcription failed, refer to logs for more details.")
                    self.root.after(0, lambda: self.status_label.config(text=f"Error"))
            threading.Thread(target=transcribe_file, name="Main - transcribe_audio_file").start()


    def save_config(self):
        config = {
            "model_size": self.model_size.get(),
            "device": self.device_var.get(),
            "compute_type": self.compute_type_var.get(),
            "language": self.language_var.get(),
            "mic_id": self.device_index_map.get(self.microphone_var.get()),
            "should_paste_content": self.should_paste_content_var.get(),
            "vad_filter": self.vad_filter_var.get(),
        }
        try:
            logger.info("Saving configuration")
            self.config_manager.save_config(config)
            logger.info("Configuration saved successfully")
            messagebox.showinfo("Sucess", "Configuration saved!")
        except Exception as e:
            messagebox.showerror("Error", f"Error saving config")
            logger.error(f"Error saving config: {e}", exc_info=True)


    def copy_text(self):
        content = self.text_viewer.text.get(1.0, tk.END)
        pyperclip.copy(content.strip())
        self.status_label.config(text="Copied!")


    def hide_window(self):
        self.root.withdraw()
        self.setup_tray_icon()


    def setup_tray_icon(self):
        image = Image.new('RGB', (64, 64), color=(0, 0, 255))
        draw = ImageDraw.Draw(image)
        draw.ellipse((16, 16, 48, 48), fill=(255, 255, 255))

        self.tray_icon = pystray.Icon("Whisper Voice", image, "Whisper Voice", menu=pystray.Menu(
            pystray.MenuItem("Restaurar", self.show_window),
            pystray.MenuItem("Sair", self.exit_app)
        ))

        threading.Thread(target=self.tray_icon.run, name="Main - Tray Icon" , daemon=True).start()


    def show_window(self, icon=None, item=None):
        self.root.deiconify()
        if self.tray_icon:
            self.tray_icon.stop()
    
    
    def show_logging_window(self):
        if self.logging_window is None:
            self.logging_window = LoggingWindow(self.root, self.logging_manager.log_queue)
        self.logging_window.show()
    
    
    def set_log_level(self, level_name: str):
        import logging
        level = getattr(logging, level_name)
        self.logging_manager.set_log_level(level)
    
    
    def exit_app(self):
        logger.info("Shutting down application")
        if self.tray_icon:
            self.tray_icon.stop()
        
        if self.logging_window:
            self.logging_window.destroy()
            
        if self.transcriber:
            self.transcriber.shutdown()
            self.transcriber = None
        logger.info("Application shutdown complete")
        self.root.quit()

