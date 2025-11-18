import os
from tkinter import filedialog
import tkinter as tk
from tkinter import ttk
import pyperclip
import logging

from app.ui.audio_visualizer import AudioVisualizer
from app.ui.live_text_view import LiveTextViewer
from app.ui.logging_window import LoggingWindow
from app.ui.config_window import ConfigWindow
from app.ui.menu_bar import MenuBar
from app.ui.transcription_handler import TranscriptionHandler
from app.ui.file_handler import FileHandler
from app.ui.tray_icon_manager import TrayIconManager
from app.ui.theme_manager import ThemeManager
from app.utils.app_state_manager import AppStateManager
from app.utils.config_manager import ConfigManager
from app.utils.logging_manager import LoggingManager

logger = logging.getLogger("app.ui.main_window")


class MainWindow:
    def __init__(self, root):
        logger.info("Initializing WispLive main window")
        self.root = root
        self.root.title("WispLive!")
        self.root.iconbitmap(default=self.get_icon_path())
        self.root.protocol("WM_DELETE_WINDOW", self.exit_app)
        
        self.config_manager = ConfigManager()
        self.config_manager.load_config_from_file()
        config = self.config_manager.get_config_dict()
        logger.debug(f"Loaded configuration: {config}")
        
        self.state_manager = AppStateManager()
        self.state_manager.load_state()
        
        self.theme_manager = ThemeManager(self.state_manager.get_dark_mode())
        self.theme_manager.register_theme_change_callback(self._on_theme_changed)
        
        saved_geometry = self.state_manager.get_window_geometry()
        if saved_geometry:
            self.root.geometry(saved_geometry)
        
        self.configure_status_bar(root, config)
        self.configure_toolbar(root, config)
        
        current_file = self.state_manager.get_current_transcription_file()
        self.text_viewer = LiveTextViewer(root, current_file)
        
        self.logging_manager = LoggingManager.get_instance()
        self.logging_window = None
        self.config_window = None
        
        callbacks = {
            "on_new_file": self.new_transcription_file,
            "on_open_file": self.open_transcription_file,
            "on_transcribe_file": self.select_audio_file,
            "on_batch_process": self.batch_process_folder,
            "on_exit": self.exit_app,
            "on_copy_text": self.copy_text,
            "on_show_logging": self.show_logging_window,
            "on_hide_window": self.hide_window,
            "on_save_config": self.save_config,
            "on_load_config": self.load_config_file,
            "on_toggle_dark_mode": self.toggle_dark_mode,
            "on_audio_waveform_analysis": self.show_audio_waveform_analysis,
        }
        
        self.menu_bar = MenuBar(root, callbacks)
        
        self.transcription_handler = TranscriptionHandler(
            self.config_manager,
            self.state_manager,
            self._update_status,
            self._update_tray_icon,
            self._update_buttons
        )
        
        self.file_handler = FileHandler(
            self.state_manager,
            self.text_viewer,
            self.menu_bar.recent_menu
        )
        
        self.tray_icon_manager = TrayIconManager(
            self.show_window,
            self.exit_app
        )
        
        self.file_handler.update_recent_files_menu()
        
        self.theme_manager.apply_theme(self.root)

    def configure_toolbar(self, root, config):
        toolbar_frame = ttk.Frame(root)
        toolbar_frame.pack(pady=5, padx=5, fill='x')
        
        model_frame = ttk.Frame(toolbar_frame)
        model_frame.pack(side="left", anchor="w", fill='x')
        
        column = 0
        
        self.toggle_model_button = ttk.Button(model_frame, text="Activate Model", command=self.toggle_model)
        self.toggle_model_button.grid(row=0, column=column, padx=(0, 5))
        column += 1
        
        self.start_button = ttk.Button(model_frame, text="Start Recording", command=self.toggle_recording, state=tk.DISABLED)
        self.start_button.grid(row=0, column=column, padx=(0, 5))
        column += 1
        
        self.configuration_button = ttk.Button(model_frame, text="Configuration", command=self.open_config_window)
        self.configuration_button.grid(row=0, column=column, padx=(0, 5))
        column += 1
        
        self.should_paste_content_var = tk.BooleanVar(value=config.get("should_paste_content", "False"))
        self.should_paste_content_checkbutton = ttk.Checkbutton(
            model_frame, 
            text='Auto paste', 
            variable=self.should_paste_content_var,
            command=lambda: self.update_should_paste(should_paste_content=self.should_paste_content_var.get())
        )
        self.should_paste_content_checkbutton.grid(row=0, column=column)
        
    def update_should_paste(self, **kwargs):
        config = self.config_manager.get_config_dict()
        config.update(**kwargs)
        self.update_transcriber_configs(**kwargs)
        
    def configure_status_bar(self, root, config):
        statusbar = ttk.Frame(root)
        statusbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_label = ttk.Label(statusbar, text="Off", anchor=tk.W)
        self.status_label.pack(side=tk.LEFT)
        sep1 = ttk.Label(statusbar, text=" | ", anchor=tk.W)
        sep1.pack(side=tk.LEFT)
        
        config_string = f'Mic input: {config['mic_id']} | Language: {config['language']}'
        self.config_label = ttk.Label(statusbar, text=config_string)
        self.config_label.pack(side=tk.LEFT)
        
        version = ttk.Label(statusbar, text="v0.2.0", anchor=tk.E)
        version.pack(side=tk.RIGHT)
        sep2 = ttk.Label(statusbar, text=" | ")
        sep2.pack(side=tk.RIGHT)

    def get_icon_path(self):
        import sys
        base_path = (
            sys._MEIPASS if getattr(sys, 'frozen', False) 
            else os.path.dirname(os.path.abspath(sys.argv[0]))
        )
        return os.path.join(base_path, "wisp.ico")

    def _update_status(self, text):
        self.status_label.config(text=text)

    def _update_tray_icon(self, state):
        self.tray_icon_manager.update_state(state)

    def _update_buttons(self, model_button_text=None, start_button_text=None, start_button_state=None):
        if model_button_text:
            self.toggle_model_button.config(text=model_button_text)
        if start_button_text:
            self.start_button.config(text=start_button_text)
        if start_button_state is not None:
            self.start_button.config(state=start_button_state)

    def toggle_model(self):
        is_model_activated = self.transcription_handler.toggle_model(self.root)
        self.menu_bar.set_model_state(is_model_activated)
        if self.config_window is not None:
            self.config_window.toggle_widgets(is_model_activated)

    def toggle_recording(self):
        self.transcription_handler.toggle_recording(self.root)

    def select_audio_file(self):
        self.transcription_handler.select_audio_file(self.root)

    def batch_process_folder(self):
        self.transcription_handler.batch_process_folder(self.root)

    def new_transcription_file(self):
        self.file_handler.new_transcription_file()

    def open_transcription_file(self, file_path=None):
        self.file_handler.open_transcription_file(file_path)

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
        filetypes = [("Json files", "*.json")]
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

    def show_window(self, icon=None, item=None):
        self.root.deiconify()

    def _on_theme_changed(self, is_dark: bool) -> None:
        self.theme_manager.apply_theme(self.root)
        if self.config_window is not None:
            self.theme_manager.apply_theme(self.config_window.window)
        if self.logging_window is not None and self.logging_window.window is not None:
            self.theme_manager.apply_theme(self.logging_window.window)

    def toggle_dark_mode(self) -> None:
        is_dark = self.theme_manager.toggle()
        self.state_manager.set_dark_mode(is_dark)

    def show_audio_waveform_analysis(self):
        logger.info("Showing audio waveform analysis")
        visualizer = AudioVisualizer(
            sample_rate=self.config_manager.get_config_dict()['sample_rate'],
            chunk_samples=1024,
            update_interval=20,
            show_spectrogram=True,
        )
        visualizer.start()
        visualizer.show()

    def show_logging_window(self):
        if self.logging_window is None:
            self.logging_window = LoggingWindow(self.root, self.logging_manager.log_queue, self.theme_manager)
        self.logging_window.show()

    def open_config_window(self):
        config = self.config_manager.get_config_dict()
        is_running = self.transcription_handler.transcriber is not None
        self.config_window = ConfigWindow(self, self.config_manager, config, is_running, self.theme_manager)

    def on_config_change(self, config):
        self.update_transcriber_configs(**config)
        config_string = f'Mic input: {config['mic_id']} | Language: {config['language']}'
        self.config_label.config(text=config_string)

    def update_transcriber_configs(self, **kwargs):
        self.transcription_handler.update_config(**kwargs)

    def exit_app(self):
        logger.info("Shutting down application")
        if self.tray_icon_manager:
            self.tray_icon_manager.stop()
        
        if self.logging_window:
            self.logging_window.destroy()
        
        self.transcription_handler.shutdown()
        logger.info("Application shutdown complete")
        self.root.quit()
