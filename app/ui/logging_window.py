import tkinter as tk
from tkinter import ttk, scrolledtext
import queue
import logging
from typing import Optional


class LoggingWindow:
    def __init__(self, parent, log_queue: queue.Queue):
        self.window: Optional[tk.Toplevel] = None
        self.parent = parent
        self.log_queue = log_queue
        self.is_running = False
        self.auto_scroll = True

        self.level_colors = {
            logging.DEBUG: "#808080",
            logging.INFO: "#000000",
            logging.WARNING: "#FF8C00",
            logging.ERROR: "#FF0000",
            logging.CRITICAL: "#8B0000",
        }

        self.current_filter_level = logging.DEBUG
        self.source_filters: dict[str, tk.BooleanVar] = {}

    def show(self):
        if self.window is not None:
            self.window.deiconify()
            self.window.lift()
            self.window.focus_force()
        else:
            self.window = tk.Toplevel(self.parent)
            self.window.title("Logging Console")
            self.window.geometry("900x600")
            self.window.protocol("WM_DELETE_WINDOW", self.hide)
            self._create_widgets()

        self.is_running = True
        self._poll_logs()

    def _create_widgets(self):
        toolbar = tk.Frame(self.window)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        tk.Label(toolbar, text="Log Level:").pack(side=tk.LEFT, padx=5)

        self.level_var = tk.StringVar(value="DEBUG")
        level_menu = ttk.Combobox(
            toolbar,
            textvariable=self.level_var,
            values=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            state="readonly",
            width=10,
        )
        level_menu.pack(side=tk.LEFT, padx=5)
        level_menu.bind("<<ComboboxSelected>>", self._on_level_change)

        tk.Button(toolbar, text="Clear", command=self._clear_logs).pack(
            side=tk.LEFT, padx=5
        )

        self.auto_scroll_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            toolbar,
            text="Auto-scroll",
            variable=self.auto_scroll_var,
            command=self._toggle_auto_scroll,
        ).pack(side=tk.LEFT, padx=5)

        # container for source checkboxes
        self.filter_frame = tk.Frame(toolbar)
        self.filter_frame.pack(side=tk.RIGHT, padx=5, pady=3)

        text_frame = tk.Frame(self.window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.text_widget = scrolledtext.ScrolledText(
            text_frame, wrap=tk.WORD, font=("Consolas", 9), state=tk.DISABLED
        )
        self.text_widget.pack(fill=tk.BOTH, expand=True)

        for level, color in self.level_colors.items():
            tag_name = logging.getLevelName(level)
            self.text_widget.tag_config(tag_name, foreground=color)

    def _ensure_source_checkbox(self, name: str):
        if name not in self.source_filters:
            var = tk.BooleanVar(value=True)
            self.source_filters[name] = var
            cb = tk.Checkbutton(
                self.filter_frame, text=name, variable=var, onvalue=True, offvalue=False
            )
            cb.pack(side=tk.TOP, anchor=tk.NW, padx=2)

    def _on_level_change(self, event=None):
        level_name = self.level_var.get()
        self.current_filter_level = getattr(logging, level_name)

    def _toggle_auto_scroll(self):
        self.auto_scroll = self.auto_scroll_var.get()

    def _clear_logs(self):
        self.text_widget.config(state=tk.NORMAL)
        self.text_widget.delete(1.0, tk.END)
        self.text_widget.config(state=tk.DISABLED)

    def _poll_logs(self):
        if not self.is_running or self.window is None:
            return

        try:
            while True:
                record = self.log_queue.get_nowait()
                self._ensure_source_checkbox(record.name)
                if record.levelno >= self.current_filter_level:
                    if self.source_filters.get(
                        record.name, tk.BooleanVar(value=True)
                    ).get(): self._display_log(record)
        except queue.Empty:
            pass

        if self.window is not None:
            self.window.after(100, self._poll_logs)

    def _display_log(self, record: logging.LogRecord):
        if self.window is None:
            return

        self.text_widget.config(state=tk.NORMAL)
        formatted_message = self._format_record(record)
        tag_name = logging.getLevelName(record.levelno)
        self.text_widget.insert(tk.END, formatted_message + "\n", tag_name)

        if self.auto_scroll:
            self.text_widget.see(tk.END)

        self.text_widget.config(state=tk.DISABLED)

    def _format_record(self, record: logging.LogRecord) -> str:
        time_str = logging.Formatter("%(asctime)s", datefmt="%H:%M:%S").format(record)
        level_str = f"[{record.levelname:8s}]"
        name_str = f"[{record.name}]"
        return f"{time_str} {level_str} {name_str} {record.getMessage()}"

    def hide(self):
        self.is_running = False
        if self.window is not None:
            self.window.withdraw()

    def destroy(self):
        self.is_running = False
        if self.window is not None:
            self.window.destroy()
            self.window = None
