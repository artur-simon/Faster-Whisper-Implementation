"""Opens transcription projects in the standalone PySide6 Studio window.

Kept separate from the Tk transcription flow: this only *launches* the Studio
(its own process / Qt event loop) and tracks recently opened projects. The
heavy import of PySide6 is deferred until a project is actually opened, so it
never slows down Tk startup.
"""
import logging
import os
import tkinter as tk
from tkinter import filedialog, messagebox

from app.projects.project_store import TRANSCRIPT_FILENAME, default_projects_root

logger = logging.getLogger("app.ui.studio_handler")


class StudioHandler:
    def __init__(self, state_manager, recent_projects_menu):
        self.state_manager = state_manager
        self.recent_projects_menu = recent_projects_menu

    def open_project_dialog(self) -> None:
        root_dir = default_projects_root()
        initialdir = root_dir if os.path.isdir(root_dir) else ""
        folder = filedialog.askdirectory(
            title="Open a transcription project folder", initialdir=initialdir
        )
        if folder:
            self.open_project(folder)

    def open_project(self, folder: str) -> None:
        """Validate and launch the Studio for an existing project folder."""
        if not os.path.isfile(os.path.join(folder, TRANSCRIPT_FILENAME)):
            messagebox.showwarning(
                "Not a transcription project",
                f"This folder has no {TRANSCRIPT_FILENAME}:\n{folder}",
            )
            return
        self.register_project(folder)

    def register_project(self, folder: str) -> None:
        """Record the project as recent and open it in the Studio."""
        self.state_manager.add_to_recent_projects(folder)
        self.update_recent_projects_menu()
        self._launch(folder)

    def _launch(self, folder: str) -> None:
        # Deferred import: pulls in PySide6 only when actually opening a window.
        from app.ui.qt.transcription_studio import launch_studio

        logger.info(f"Opening transcription project in Studio: {folder}")
        try:
            launch_studio(folder)
        except Exception as e:
            logger.error(f"Failed to launch Studio: {e}", exc_info=True)
            messagebox.showerror(
                "Error", "Failed to open the Transcription Studio, refer to logs."
            )

    def update_recent_projects_menu(self) -> None:
        self.recent_projects_menu.delete(0, tk.END)
        recent = self.state_manager.get_recent_projects()
        if not recent:
            self.recent_projects_menu.add_command(
                label="(No recent projects)", state=tk.DISABLED
            )
            return
        for folder in recent:
            self.recent_projects_menu.add_command(
                label=os.path.basename(os.path.normpath(folder)),
                command=lambda f=folder: self.open_project(f),
            )
        self.recent_projects_menu.add_separator()
        self.recent_projects_menu.add_command(
            label="Clear Recent Projects", command=self.clear_recent_projects
        )

    def clear_recent_projects(self) -> None:
        self.state_manager.clear_recent_projects()
        self.update_recent_projects_menu()
