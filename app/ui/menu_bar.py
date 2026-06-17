import tkinter as tk
from tkinter import messagebox
import logging

logger = logging.getLogger("app.ui.menu_bar")


class MenuBar:
    def __init__(self, root, callbacks):
        self.root = root
        self.callbacks = callbacks
        self.menubar = tk.Menu(root)
        self.file_menu = None
        self.recent_menu = None
        self._create_menus()
        root.config(menu=self.menubar)

    def _create_menus(self):
        file_menu = tk.Menu(self.menubar, tearoff=0)
        file_menu.add_command(label="New Transcription", command=self.callbacks.get("on_new_file"))
        file_menu.add_command(label="Open Transcription", command=self.callbacks.get("on_open_file"))
        self.recent_menu = tk.Menu(file_menu, tearoff=0)
        file_menu.add_cascade(label="Open Recent", menu=self.recent_menu)
        file_menu.add_separator()
        file_menu.add_command(label="Open in Studio", command=self.callbacks.get("on_open_studio"))
        self.recent_projects_menu = tk.Menu(file_menu, tearoff=0)
        file_menu.add_cascade(label="Recent Studio Projects", menu=self.recent_projects_menu)
        file_menu.add_separator()
        file_menu.add_command(label="Transcribe Audio File", command=self.callbacks.get("on_transcribe_file"), state=tk.DISABLED)
        file_menu.add_command(label="Batch Process Folder", command=self.callbacks.get("on_batch_process"), state=tk.DISABLED)
        file_menu.add_separator()
        file_menu.add_command(label="Export as TXT", command=lambda: print("Export TXT"), state=tk.DISABLED)
        file_menu.add_command(label="Export as SRT", command=lambda: print("Export SRT"), state=tk.DISABLED)
        file_menu.add_command(label="Export as VTT", command=lambda: print("Export VTT"), state=tk.DISABLED)
        file_menu.add_command(label="Export as DOCX", command=lambda: print("Export DOCX"), state=tk.DISABLED)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.callbacks.get("on_exit"))
        self.menubar.add_cascade(label="File", menu=file_menu)
        self.file_menu = file_menu

        edit_menu = tk.Menu(self.menubar, tearoff=0)
        edit_menu.add_command(label="Copy Text", command=self.callbacks.get("on_copy_text"))
        edit_menu.add_command(label="Find / Replace", command=lambda: print("Find Replace"), state=tk.DISABLED)
        edit_menu.add_command(label="Clear Transcript", command=lambda: print("Clear Transcript"), state=tk.DISABLED)
        edit_menu.add_command(label="Undo", command=lambda: print("Undo"), state=tk.DISABLED)
        edit_menu.add_command(label="Redo", command=lambda: print("Redo"), state=tk.DISABLED)
        edit_menu.add_separator()
        edit_menu.add_command(label="Timestamp Alignment", command=lambda: print("Align timestamps"), state=tk.DISABLED)
        edit_menu.add_command(label="Merge Paragraphs", command=lambda: print("Merge paragraphs"), state=tk.DISABLED)
        edit_menu.add_command(label="Split Paragraphs", command=lambda: print("Split paragraphs"), state=tk.DISABLED)
        self.menubar.add_cascade(label="Edit", menu=edit_menu)

        view_menu = tk.Menu(self.menubar, tearoff=0)
        view_menu.add_command(label="Show Logging Console", command=self.callbacks.get("on_show_logging"))
        view_menu.add_command(label="Hide to System tray", command=self.callbacks.get("on_hide_window"))
        view_menu.add_command(label="Toggle Dark Mode", command=self.callbacks.get("on_toggle_dark_mode"))
        view_menu.add_command(label="Audio Waveform Analysis", command=self.callbacks.get("on_audio_waveform_analysis"))
        view_menu.add_command(label="Show/Hide Timestamps", command=lambda: print("Timestamps toggle"), state=tk.DISABLED)
        view_menu.add_command(label="Word Confidence Heatmap", command=lambda: print("Confidence heatmap"), state=tk.DISABLED)
        view_menu.add_command(label="Real-Time Highlighting", command=lambda: print("Highlighting"), state=tk.DISABLED)
        self.menubar.add_cascade(label="View", menu=view_menu)

        tools_menu = tk.Menu(self.menubar, tearoff=0)
        tools_menu.add_command(label="Language Auto-Detect", command=lambda: print("Auto-detect language"), state=tk.DISABLED)
        tools_menu.add_command(label="Speaker Diarization", command=lambda: print("Speaker diarization"), state=tk.DISABLED)
        tools_menu.add_command(label="Toggle Punctuation Restoration", command=lambda: print("Punctuation restoration"), state=tk.DISABLED)
        tools_menu.add_command(label="Batch Accuracy Stats", command=lambda: print("Batch stats"), state=tk.DISABLED)
        tools_menu.add_command(label="Custom Vocabulary", command=lambda: print("Custom vocab"), state=tk.DISABLED)
        self.menubar.add_cascade(label="Tools", menu=tools_menu)

        settings_menu = tk.Menu(self.menubar, tearoff=0)
        settings_menu.add_command(label="Save Preset", command=self.callbacks.get("on_save_config"))
        settings_menu.add_command(label="Load Preset", command=self.callbacks.get("on_load_config"))
        settings_menu.add_command(label="Hotkeys", command=lambda: print("Hotkeys"), state=tk.DISABLED)
        self.menubar.add_cascade(label="Settings", menu=settings_menu)

        help_menu = tk.Menu(self.menubar, tearoff=0)
        help_menu.add_command(label="About", command=lambda: messagebox.showinfo("About", (
            "WispLive Voice Transcriber\n"
            "Version 0.2.0\n"
            "Developed by Artur Simon\n"
            "For support or updates, visit: https://artursimon.dev/wisplive"
        )))
        help_menu.add_command(label="Model Info", command=lambda: print("Model info"), state=tk.DISABLED)
        help_menu.add_command(label="Benchmark Test", command=lambda: print("Benchmark"), state=tk.DISABLED)
        help_menu.add_command(label="Open Logs Folder", command=lambda: print("Logs folder"), state=tk.DISABLED)
        self.menubar.add_cascade(label="Help", menu=help_menu)

    def set_model_state(self, is_activated):
        state = tk.NORMAL if is_activated else tk.DISABLED
        self.file_menu.entryconfig("Transcribe Audio File", state=state)
        self.file_menu.entryconfig("Batch Process Folder", state=state)

