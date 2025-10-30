import os
import logging
import tkinter as tk
from tkinter import filedialog

logger = logging.getLogger("app.ui.file_handler")


class FileHandler:
    def __init__(self, state_manager, text_viewer, recent_menu):
        self.state_manager = state_manager
        self.text_viewer = text_viewer
        self.recent_menu = recent_menu

    def new_transcription_file(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialdir=self.state_manager.get_last_save_directory()
        )
        
        if file_path:
            self.state_manager.set_last_save_directory(os.path.dirname(file_path))
            self.state_manager.set_current_transcription_file(file_path)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("")
            
            self.text_viewer.set_file_path(file_path)
            self.update_recent_files_menu()
            logger.info(f"New transcription file created: {file_path}")

    def open_transcription_file(self, file_path=None):
        if not file_path:
            file_path = filedialog.askopenfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                initialdir=self.state_manager.get_last_save_directory()
            )
        
        if file_path and os.path.exists(file_path):
            self.state_manager.set_last_save_directory(os.path.dirname(file_path))
            self.state_manager.set_current_transcription_file(file_path)
            
            self.text_viewer.set_file_path(file_path)
            self.update_recent_files_menu()
            logger.info(f"Opened transcription file: {file_path}")

    def update_recent_files_menu(self):
        self.recent_menu.delete(0, tk.END)
        
        recent_files = self.state_manager.get_recent_files()
        
        if not recent_files:
            self.recent_menu.add_command(label="(No recent files)", state=tk.DISABLED)
        else:
            for file_path in recent_files:
                file_name = os.path.basename(file_path)
                self.recent_menu.add_command(
                    label=file_name,
                    command=lambda f=file_path: self.open_transcription_file(f)
                )
            
            self.recent_menu.add_separator()
            self.recent_menu.add_command(
                label="Clear Recent Files",
                command=self.clear_recent_files
            )

    def clear_recent_files(self):
        self.state_manager.clear_recent_files()
        self.update_recent_files_menu()

