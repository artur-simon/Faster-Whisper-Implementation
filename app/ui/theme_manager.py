import tkinter as tk
from tkinter import ttk, scrolledtext
import logging
from typing import Callable

logger = logging.getLogger("app.ui.theme_manager")


class ThemeManager:
    LIGHT_THEME = {
        "bg": "#EEEEEE",
        "fg": "#000000",
        "select_bg": "#316AC5",
        "select_fg": "#E9E9E9",
        "insert_bg": "#000000",
        "hover": "#E0E0E0",
        "disabled_bg": "#F5F5F5",
        "disabled_fg": "#4C4C4C",
        "tab_bg": "#F0F0F0",
    }

    DARK_THEME = {
        "bg": "#1E1E1E",
        "fg": "#FFFFFF",
        "select_bg": "#4E6676",
        "select_fg": "#FFFFFF",
        "insert_bg": "#D4D4D4",
        "hover": "#2D2D2D",
        "disabled_bg": "#2D2D2D",
        "disabled_fg": "#000000",
        "tab_bg": "#2D2D2D",
    }

    def __init__(self, initial_dark_mode: bool = False):
        self._dark_mode = initial_dark_mode
        self._theme_change_callbacks: list[Callable[[bool], None]] = []

    def is_dark_mode(self) -> bool:
        return self._dark_mode

    def toggle(self) -> bool:
        self._dark_mode = not self._dark_mode
        self._notify_callbacks()
        return self._dark_mode

    def set_dark_mode(self, enabled: bool) -> None:
        if self._dark_mode != enabled:
            self._dark_mode = enabled
            self._notify_callbacks()

    def register_theme_change_callback(self, callback: Callable[[bool], None]) -> None:
        self._theme_change_callbacks.append(callback)

    def _notify_callbacks(self) -> None:
        for callback in self._theme_change_callbacks:
            try:
                callback(self._dark_mode)
            except Exception as e:
                logger.error(f"Error in theme change callback: {e}", exc_info=True)

    def get_theme(self) -> dict:
        return self.DARK_THEME if self._dark_mode else self.LIGHT_THEME

    def _configure_ttk_styles(self, theme: dict) -> None:
        style = ttk.Style()
        style.theme_use("default")
        
        if self._dark_mode:
            for widget in ["TFrame", "TLabel", "TButton", "TCheckbutton"]:
                style.configure(f"Dark.{widget}", background=theme["bg"], foreground=theme["fg"])
            
            style.configure("Dark.TButton", disabledbackground=theme["disabled_bg"])
            style.map("Dark.TButton", background=[("active", theme["hover"])])
            
            style.map("Dark.TCheckbutton", background=[("active", theme["hover"])])
            
            style.configure("Dark.TEntry", fieldbackground=theme["bg"], foreground=theme["fg"])
            style.configure("Dark.TNotebook", background=theme["bg"])
            style.configure("Dark.TNotebook.Tab", background=theme["tab_bg"], foreground=theme["fg"])
            style.map("Dark.TNotebook.Tab", background=[("selected", theme["bg"])])
            
            style.configure("Dark.TCombobox", fieldbackground=theme["bg"], foreground=theme["fg"], 
                          background=theme["bg"], arrowcolor=theme["fg"])
            style.map("Dark.TCombobox", fieldbackground=[("readonly", theme["bg"])])
        else:
            style.configure("TNotebook.Tab", background=theme["tab_bg"], foreground=theme["fg"])
            style.map("TNotebook.Tab", background=[("selected", theme["bg"])])

    def _apply_common_colors(self, widget: tk.Widget, theme: dict) -> None:
        try:
            widget.configure(
                bg=theme["bg"],
                fg=theme["fg"],
                selectbackground=theme["select_bg"],
                selectforeground=theme["select_fg"],
                disabledforeground=theme["disabled_fg"]
            )
        except tk.TclError:
            try:
                widget.configure(bg=theme["bg"], fg=theme["fg"])
            except tk.TclError:
                pass

    def _configure_widget(self, widget: tk.Widget, theme: dict) -> None:
        if isinstance(widget, (tk.Tk, tk.Toplevel)):
            widget.configure(bg=theme["bg"])
            self._configure_ttk_styles(theme)
        
        elif isinstance(widget, ttk.Widget):
            widget_class = widget.winfo_class()
            if widget_class in ["TNotebook", "TFrame", "TLabel", "TButton", "TCheckbutton", "TEntry", "TCombobox"]:
                if self._dark_mode:
                    widget.configure(style=f"Dark.{widget_class}")
                else:
                    widget.configure(style=f"{widget_class}")
        
        elif isinstance(widget, (tk.Text, tk.Entry)):
            self._apply_common_colors(widget, theme)
            try:
                widget.configure(insertbackground=theme["insert_bg"])
            except tk.TclError:
                pass
        
        elif isinstance(widget, scrolledtext.ScrolledText):
            self._apply_common_colors(widget.text, theme)
            widget.text.configure(insertbackground=theme["insert_bg"])
        
        elif isinstance(widget, tk.Menu):
            widget.configure(bg=theme["bg"], fg=theme["fg"], selectcolor=theme["select_bg"],
                           activebackground=theme["select_bg"], activeforeground=theme["select_fg"],
                           disabledforeground=theme["disabled_fg"])
        
        else: self._apply_common_colors(widget, theme)

    def apply_theme(self, widget: tk.Widget) -> None:
        theme = self.get_theme()
        self._configure_widget(widget, theme)
        for child in widget.winfo_children():
            try:
                self.apply_theme(child)
            except Exception as e:
                logger.debug(f"Could not apply theme to {type(child).__name__}: {e}")
