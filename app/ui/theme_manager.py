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
        "button_disabled": "#F5F5F5",
        "tab_bg": "#F0F0F0",
        "scrollbar_bg": "#EEEEEE",
        "scrollbar_active": "#316AC5",
    }

    DARK_THEME = {
        "bg": "#1E1E1E",
        "fg": "#D4D4D4",
        "select_bg": "#094771",
        "select_fg": "#FFFFFF",
        "insert_bg": "#D4D4D4",
        "hover": "#2D2D2D",
        "button_disabled": "#2D2D2D",
        "tab_bg": "#2D2D2D",
        "scrollbar_bg": "#3D3D3D",
        "scrollbar_active": "#555555",
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

    def on_theme_change(self, callback: Callable[[bool], None]) -> None:
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
        style.theme_use("clam" if self._dark_mode else "default")

        if self._dark_mode:
            for widget in ["TFrame", "TLabel", "TButton", "TCheckbutton"]:
                style.configure(f"Dark.{widget}", background=theme["bg"], foreground=theme["fg"])
            
            style.configure("Dark.TButton", disabledbackground=theme["button_disabled"])
            style.map("Dark.TButton", background=[("active", theme["hover"])])
            
            style.configure("Dark.TEntry", fieldbackground=theme["bg"], foreground=theme["fg"])
            style.configure("Dark.TNotebook", background=theme["bg"])
            style.configure("Dark.TNotebook.Tab", background=theme["tab_bg"], foreground=theme["fg"])
            style.map("Dark.TNotebook.Tab", background=[("selected", theme["bg"])])
            
            style.configure("Dark.TCombobox", fieldbackground=theme["bg"], foreground=theme["fg"], 
                          background=theme["bg"], arrowcolor=theme["fg"])
            style.map("Dark.TCombobox", fieldbackground=[("readonly", theme["bg"])])
            
            style.configure("Dark.TScrollbar", background=theme["scrollbar_bg"], troughcolor=theme["bg"], 
                          arrowcolor=theme["fg"])
            style.map("Dark.TScrollbar", background=[("active", theme["scrollbar_active"])])
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
            )
        except tk.TclError:
            try:
                widget.configure(bg=theme["bg"], fg=theme["fg"])
            except tk.TclError:
                pass

    def _configure_scrollbar(self, widget: tk.Scrollbar, theme: dict) -> None:
        widget.configure(bg=theme["scrollbar_bg"], troughcolor=theme["bg"], activebackground=theme["scrollbar_active"], 
                        highlightbackground=theme["bg"], borderwidth=0)

    def _style_combobox_popup(self, combobox: ttk.Combobox, theme: dict) -> None:
        root = combobox.winfo_toplevel()
        for widget in root.winfo_children():
            if isinstance(widget, tk.Toplevel) and widget.winfo_viewable():
                self._apply_theme_recursive(widget, theme)

    def _apply_theme_recursive(self, parent: tk.Widget, theme: dict) -> None:
        try:
            parent.configure(bg=theme["bg"])
        except tk.TclError:
            pass
        
        for child in parent.winfo_children():
            self._apply_common_colors(child, theme)
            if isinstance(child, (tk.Frame, tk.Toplevel)):
                self._apply_theme_recursive(child, theme)

    def _configure_widget(self, widget: tk.Widget, theme: dict) -> None:
        if isinstance(widget, (tk.Tk, tk.Toplevel)):
            widget.configure(bg=theme["bg"])
            self._configure_ttk_styles(theme)
        
        elif isinstance(widget, ttk.Widget):
            if self._dark_mode:
                widget_class = widget.winfo_class()
                if widget_class in ["TNotebook", "TFrame", "TLabel", "TButton", "TCheckbutton", "TEntry", "TScrollbar"]:
                    widget.configure(style=f"Dark.{widget_class}")
                elif widget_class == "TCombobox":
                    widget.configure(style="Dark.TCombobox")
                    original = widget.cget("postcommand")
                    widget.configure(postcommand=lambda: (
                        original() if original else None,
                        widget.after(50, lambda: self._style_combobox_popup(widget, theme))
                    ))
        
        elif isinstance(widget, (tk.Text, tk.Entry)):
            self._apply_common_colors(widget, theme)
            try:
                widget.configure(insertbackground=theme["insert_bg"])
            except tk.TclError:
                pass
        
        elif isinstance(widget, tk.Scrollbar):
            self._configure_scrollbar(widget, theme)
        
        elif isinstance(widget, scrolledtext.ScrolledText):
            self._apply_common_colors(widget.text, theme)
            widget.text.configure(insertbackground=theme["insert_bg"])
            for bar in ['vbar', 'hbar']:
                if hasattr(widget, bar):
                    self._configure_scrollbar(getattr(widget, bar), theme)
        
        elif isinstance(widget, tk.Menu):
            widget.configure(bg=theme["bg"], fg=theme["fg"], selectcolor=theme["select_bg"],
                           activebackground=theme["select_bg"], activeforeground=theme["select_fg"])
        
        else:
            self._apply_common_colors(widget, theme)

    def apply_theme(self, widget: tk.Widget) -> None:
        theme = self.get_theme()
        self._configure_widget(widget, theme)
        for child in widget.winfo_children():
            try:
                self.apply_theme(child)
            except Exception as e:
                logger.debug(f"Could not apply theme to {type(child).__name__}: {e}")
