from tkinter import filedialog, messagebox
from PIL import Image, ImageDraw
import tkinter as tk
import pyperclip
import pystray
import threading
import logging

from app.models import TranscriptionConfig
from app.transcription.transcription_controller import TranscriptionController
from app.ui.live_text_view import LiveTextViewer
from app.ui.logging_window import LoggingWindow
from app.ui.config_window import ConfigWindow
from app.utils.config_manager import ConfigManager
from app.utils.logging_manager import LoggingManager

logger = logging.getLogger("app.ui.main_window")


class MainWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("WispLive!")
        self.root.iconbitmap(default=self.get_icon_path())
        self.root.protocol("WM_DELETE_WINDOW", self.exit_app)
        
        logger.info("Initializing WispLive main window")
        
        self.config_manager = ConfigManager()
        self.config_manager.load_config_from_file()
        config = self.config_manager.get_config_dict()
        logger.debug(f"Loaded configuration: {config}")
        self.configure_menu_bar(root, config)
        self.configure_toolbar(root, config)
        self.text_viewer = LiveTextViewer(root, "transcription.txt")
        self.configure_status_bar(root, config)
        
        self.setup_tray_icon()
        
        self.is_running = False
        self.transcriber = None
        self.logging_manager = LoggingManager.get_instance()
        self.logging_window = None
        self.config_window = None
        
    
    def configure_menu_bar(self, root, config):
        
        menubar = tk.Menu(root)
        
        # ===== File Menu =====
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Transcribe Audio File", command=self.select_audio_file, state=tk.DISABLED)
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
        self.file_menu = file_menu

        # ===== Edit Menu =====
        edit_menu = tk.Menu(menubar, tearoff=0)
        edit_menu.add_command(label="Copy Text", command=self.copy_text)
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
        view_menu.add_command(label="Show Logging Console", command=self.show_logging_window)
        view_menu.add_command(label="Hide to System tray", command=self.hide_window)
        view_menu.add_command(label="Toggle Dark Mode", state=tk.DISABLED)
        view_menu.add_command(label="Toggle Waveform", command=lambda: print("Waveform toggle"), state=tk.DISABLED)
        view_menu.add_command(label="Show/Hide Timestamps", command=lambda: print("Timestamps toggle"), state=tk.DISABLED)
        view_menu.add_command(label="Word Confidence Heatmap", command=lambda: print("Confidence heatmap"), state=tk.DISABLED)
        view_menu.add_command(label="Real-Time Highlighting", command=lambda: print("Highlighting"), state=tk.DISABLED)
        menubar.add_cascade(label="View", menu=view_menu)

        # ===== Tools Menu =====
        tools_menu = tk.Menu(menubar, tearoff=0)
        tools_menu.add_command(label="Language Auto-Detect", command=lambda: print("Auto-detect language"), state=tk.DISABLED)
        tools_menu.add_command(label="Speaker Diarization", command=lambda: print("Speaker diarization"), state=tk.DISABLED)
        tools_menu.add_command(label="Toggle Punctuation Restoration", command=lambda: print("Punctuation restoration"), state=tk.DISABLED)
        tools_menu.add_command(label="Batch Accuracy Stats", command=lambda: print("Batch stats"), state=tk.DISABLED)
        tools_menu.add_command(label="Custom Vocabulary", command=lambda: print("Custom vocab"), state=tk.DISABLED)
        menubar.add_cascade(label="Tools", menu=tools_menu)

        # ===== Settings Menu =====
        settings_menu = tk.Menu(menubar, tearoff=0)
        settings_menu.add_command(label="Save Preset", command=self.save_config)
        settings_menu.add_command(label="Load Preset", command=self.load_config_file)
        settings_menu.add_command(label="Hotkeys", command=lambda: print("Hotkeys"), state=tk.DISABLED)
        menubar.add_cascade(label="Settings", menu=settings_menu)

        # ===== Help Menu =====
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About",command=lambda: messagebox.showinfo("About", (
            "WispLive Voice Transcriber\n"
            "Version 0.2.0\n"
            "Developed by Artur Simon\n"
            "For support or updates, visit: https://artursimon.dev/wisplive"
            )))
        help_menu.add_command(label="Model Info", command=lambda: print("Model info"), state=tk.DISABLED)
        help_menu.add_command(label="Benchmark Test", command=lambda: print("Benchmark"), state=tk.DISABLED)
        help_menu.add_command(label="Open Logs Folder", command=lambda: print("Logs folder"), state=tk.DISABLED)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        root.config(menu=menubar)
        
    
    def configure_toolbar(self, root, config):
        
        toolbar_frame = tk.Frame(root)
        toolbar_frame.pack(pady=5, padx=5, fill='x')
        
        model_frame = tk.Frame(toolbar_frame)
        model_frame.pack(side="left", anchor="w", fill='x')
        self._model_frame = model_frame
        
        column = 0
        
        self.toggle_model_button = tk.Button(model_frame, text="Activate Model", command=self.toggle_model)
        self.toggle_model_button.grid(row=0, column=column, padx=(0, 5))
        column += 1
        
        self.start_button = tk.Button(model_frame, text="Start Recording", command=self.toggle_recording, state=tk.DISABLED)
        self.start_button.grid(row=0, column=column, padx=(0, 5))
        column += 1
        
        self.configuration_button = tk.Button(model_frame, text="Configuration", command=self.open_config_window)
        self.configuration_button.grid(row=0, column=column, padx=(0, 5))
        column += 1
        
        self.should_paste_content_var = tk.BooleanVar(value=config.get("should_paste_content", "False"))
        self.should_paste_content_checkbutton = tk.Checkbutton(
            model_frame, 
            text='Auto paste', 
            variable=self.should_paste_content_var,
            onvalue=1,
            offvalue=0,
            command=lambda:self.update_transcriber_configs(should_paste_content=self.should_paste_content_var.get())
        )
        self.should_paste_content_checkbutton.grid(row=0, column=column)
        column += 1
    
    
    def configure_status_bar(self, root, config):
        statusbar = tk.Frame(root, bd=1, relief=tk.SUNKEN, padx=0, pady=0)
        statusbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_label = tk.Label(statusbar, text="Off", anchor=tk.W)
        self.status_label.pack(side=tk.LEFT)
        sep1 = tk.Label(statusbar, text=" | ", anchor=tk.W)
        sep1.pack(side=tk.LEFT)
        
        config_string = f'Mic input: {config['mic_id']} | Language: {config['language']}'
        self.config_label = tk.Label(statusbar, text=config_string)
        self.config_label.pack(side=tk.LEFT)
        
        version = tk.Label(statusbar, text="v0.2.0", anchor=tk.E)
        version.pack(side=tk.RIGHT)
        sep2 = tk.Label(statusbar, text=" | ")
        sep2.pack(side=tk.RIGHT)


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
                config_dict = self.config_manager.get_config_dict()
                config = TranscriptionConfig(**config_dict)
                
                logger.info(f"Activating model: {config.model_size} on {config.device}")
                self.transcriber = TranscriptionController(config)
                logger.info("Model activated successfully")
                
                self.toggle_model_button.config(text="Release Model")
                self.start_button.config(state=tk.NORMAL)
                self.file_menu.entryconfig("Transcribe Audio File", state=tk.NORMAL)
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
            self.file_menu.entryconfig("Transcribe Audio File", state=tk.DISABLED)
        
        is_model_activated = self.transcriber is not None
        self.status_label.config(text= "Ready" if is_model_activated else "Off")
        self.tray_icon.icon = self.get_tray_icon("READY" if is_model_activated else "OFF")
        if self.config_window is not None: self.config_window.toggle_widgets(is_model_activated)


    def toggle_recording(self):
        if not self.is_running:
            logger.info("Starting recording")
            self.is_running = True
            self.transcriber.run(output_file="transcription.txt")
            
            self.status_label.config(text="Recording")
            self.start_button.config(text="Stop Recording")
            self.tray_icon.icon = self.get_tray_icon("RECORDING")
        else:
            if self.transcriber:
                self.is_running = False
                self.transcriber.stop()
                logger.info("Stopping recording")
                
            self.status_label.config(text="Ready")
            self.start_button.config(text="Start Recording")
            self.tray_icon.icon = self.get_tray_icon("READY")

        
    def select_audio_file(self):
        filetypes = (("MP3 files","*.mp3"), ("WAV files", "*.wav"), ("All files", "*.*"))
        filepath = filedialog.askopenfilename(title="Select an audio file", filetypes=filetypes)
        if filepath and self.transcriber:
            logger.info(f"Selected audio file for transcription: {filepath}")
            self.status_label.config(text="Transcribing file")
            def transcribe_file():
                try:
                    self.transcriber.transcribe_audio_file(filepath, "transcription.txt")
                    logger.info(f"File transcription completed: {filepath}")
                    self.root.after(0, lambda: self.status_label.config(text="Transcription completed"))
                except Exception as e:
                    logger.error(f"File transcription failed: {e}", exc_info=True)
                    messagebox.showerror("Error", f"File transcription failed, refer to logs for more details.")
                    self.root.after(0, lambda: self.status_label.config(text=f"Error"))
            threading.Thread(target=transcribe_file, name="Main - transcribe_audio_file").start()


    def save_config(self):
        current_config = self.config_manager.get_config_dict()
        config = {
            **current_config,
            "should_paste_content": self.should_paste_content_var.get(),
        }
        try:
            logger.info("Saving configuration")
            self.config_manager.save_config_to_file(config)
            logger.info("Configuration saved successfully")
            self.status_label.config(text="Config saved")
        except Exception as e:
            self.status_label.config(text="Error saving config")
            logger.error(f"Error saving config: {e}", exc_info=True)


    def load_config_file(self):
        filetypes = [("Json files","*.json")]
        filepath = filedialog.askopenfilename(title="Select a config file", filetypes=filetypes)
        if filepath:
            self.config_manager.load_config_from_file(filepath)
            self.on_config_change(self.config_manager.get_config_dict())
            self.status_label.config(text="Config loaded")


    def copy_text(self):
        content = self.text_viewer.text.get(1.0, tk.END)
        pyperclip.copy(content.strip())
        self.status_label.config(text="Copied")


    def hide_window(self):
        self.root.withdraw()


    def setup_tray_icon(self):
        image = self.get_tray_icon("OFF")
        
        self.tray_icon = pystray.Icon("WispLive", image, "WispLive", menu=pystray.Menu(
            pystray.MenuItem("Restaurar", self.show_window),
            pystray.MenuItem("Sair", self.exit_app)
        ))

        threading.Thread(target=self.tray_icon.run, name="Main - Tray Icon" , daemon=True).start()
        
        
    def get_tray_icon(self, state):
        image = Image.new('RGBA', (64, 64), color=(0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        
        if state == "READY":
            draw.ellipse((16, 16, 48, 48), fill=(0, 255, 0, 255))
        elif state == "RECORDING":
            draw.ellipse((16, 16, 48, 48), fill=(255, 0, 0, 255))
        else:
            draw.ellipse((16, 16, 48, 48), fill=(0, 0, 255, 255))
            
        return image


    def show_window(self, icon=None, item=None):
        self.root.deiconify()
    
    
    def show_logging_window(self):
        if self.logging_window is None:
            self.logging_window = LoggingWindow(self.root, self.logging_manager.log_queue)
        self.logging_window.show()
    
    
    def open_config_window(self):
        config = self.config_manager.get_config_dict()
        is_running = self.transcriber is not None
        self.config_window = ConfigWindow(self, self.config_manager, config, is_running)
    
    
    def on_config_change(self, config):
        self.update_transcriber_configs(**config)
        
        config_string = f'Mic input: {config['mic_id']} | Language: {config['language']}'
        self.config_label.config(text= config_string)
                
    
    def update_transcriber_configs(self, **kwargs):
        if(self.transcriber):
            self.transcriber.update_input_config(**kwargs)
        
    
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

