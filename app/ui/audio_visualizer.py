import threading
import time
from typing import Callable, Optional

import numpy as np
import tkinter as tk
from tkinter import ttk

import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QVBoxLayout, QApplication
import sys

from app.audio.audio_data_provider import AudioDataProvider

class AudioVisualizer:
    """
    Realtime audio visualizer in a new Tk Toplevel window.

    Args:
        parent: tkinter root or parent widget.
        get_audio_chunk: callable -> np.ndarray (1D). Return newest samples (float32).
        sample_rate: sample rate in Hz (used for x axis / spectrogram).
        chunk_samples: expected number of samples per chunk (for sizing buffer).
        update_interval: ms between UI updates.
        show_spectrogram: if True, maintain and draw a scrolling spectrogram.
    """

    def __init__(
        self,
        parent,
        get_audio_chunk: Callable[[], np.ndarray],
        sample_rate: int = 44100,
        chunk_samples: int = 1024,
        update_interval: int = 40,
        show_spectrogram: bool = False,
    ):
        self.parent = parent
        self.get_audio_chunk = get_audio_chunk
        self.sample_rate = sample_rate
        self.chunk_samples = chunk_samples
        self.update_interval = update_interval
        self.show_spectrogram = show_spectrogram
        
        self._audio_data_provider = AudioDataProvider(sample_rate, device_index=2)
        
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # rolling buffer for waveform (seconds)
        self._buffer_seconds = 2.0
        self._buffer_size = int(self.sample_rate * self._buffer_seconds)
        self._buffer = np.zeros(self._buffer_size, dtype=np.float32)
        self._buf_pos = 0  # write index in circular buffer

        # spectrogram state (history of spectra columns)
        if self.show_spectrogram:
            # number of time columns to keep
            self._spec_history = 200
            self._spec_data = np.zeros(
                (self._spec_history, self.chunk_samples // 2 + 1), dtype=np.float32
            )

        self._qt_app = None
        self._init_qt_app()
        self._create_window()
        self._init_plot()

    def _init_qt_app(self):
        if QApplication.instance() is None:
            self._qt_app = QApplication(sys.argv)
        else:
            self._qt_app = QApplication.instance()

    def _create_window(self):
        self.win = tk.Toplevel(self.parent)
        self.win.title("Audio Input Visualizer")
        self.win.protocol("WM_DELETE_WINDOW", self.stop)
        self.win.geometry("800x400")

        self.frame = ttk.Frame(self.win, padding=6)
        self.frame.pack(fill=tk.BOTH, expand=True)

        ctrl = ttk.Frame(self.frame)
        ctrl.pack(side=tk.TOP, fill=tk.X)
        self.start_btn = ttk.Button(ctrl, text="Start", command=self.start)
        self.start_btn.pack(side=tk.LEFT)
        self.stop_btn = ttk.Button(
            ctrl, text="Stop", command=self.stop, state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT)
        ttk.Label(ctrl, text=f"SR: {self.sample_rate} Hz").pack(side=tk.RIGHT)

        self.plot_container = ttk.Frame(self.frame)
        self.plot_container.pack(fill=tk.BOTH, expand=True)

    def _init_plot(self):
        try:
            pg.setConfigOptions(antialias=True, useOpenGL=True)
        except Exception:
            pg.setConfigOptions(antialias=True, useOpenGL=False)
        
        self.qt_widget = QWidget()
        self.qt_widget.show()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.qt_widget.setLayout(layout)
        
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', 'amplitude')
        self.plot_widget.setLabel('bottom', 'seconds')
        self.plot_widget.setXRange(-self._buffer_seconds, 0)
        self.plot_widget.setYRange(-1.0, 1.0)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
        layout.addWidget(self.plot_widget)
        
        if self.show_spectrogram:
            self.spec_widget = pg.PlotWidget()
            self.spec_widget.setLabel('left', 'frequency', units='Hz')
            self.spec_widget.setLabel('bottom', 'seconds')
            self.spec_widget.setXRange(-self._buffer_seconds, 0)
            self.spec_widget.setYRange(0, self.sample_rate / 2)
            self.spec_widget.showGrid(x=True, y=True, alpha=0.3)
            layout.addWidget(self.spec_widget)
            self._spec_img = pg.ImageItem()
            self.spec_widget.addItem(self._spec_img)
            try:
                self._spec_img.setLookupTable(pg.colormap.get('viridis').getLookupTable())
            except Exception:
                pass
        
        t = np.linspace(-self._buffer_seconds, 0, self._buffer_size)
        self._line = self.plot_widget.plot(t, self._buffer, pen='c')
        
        self._embed_qt_widget()

    def _embed_qt_widget(self):        
        def update_size():
            try:
                width = self.plot_container.winfo_width()
                height = self.plot_container.winfo_height()
                if width > 1 and height > 1:
                    self.qt_widget.resize(width, height)
            except Exception:
                pass

        try:
            from PySide6.QtGui import QWindow
            hwnd = int(self.plot_container.winfo_id())
            window = QWindow.fromWinId(hwnd)
            container = QWidget.createWindowContainer(window, self.qt_widget)
            container.setParent(self.qt_widget)
            self.plot_container.update()
            self.win.update()
            update_size()
            self._qt_app.processEvents()
        except Exception:
            pass

    def start(self):
        if self._running:
            return
        self._running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        self._audio_data_provider.start()
        
        # start background poll thread (non-blocking)
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        # start UI updater
        self._thread2 = threading.Thread(target=self._schedule_ui_update, daemon=True)
        self._thread2.start()

    def stop(self):
        if not self._running:
            try:
                self.win.destroy()
            except Exception:
                pass
            return
        self._running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        # thread will exit; keep the window open so user can restart if desired

    def close(self):
        self.stop()
        try:
            self.win.destroy()
        except Exception:
            pass

    def _poll_loop(self):
        """
        Background thread: pull audio chunks quickly and append to rolling buffer.
        Keep this loop tight but cooperative to avoid hogging CPU.
        """
        while self._running:
            try:
                chunk = self._audio_data_provider.get_audio_chunk(441)
                if chunk is None or len(chunk) == 0:
                    time.sleep(0.01)
                    continue
                chunk = np.asarray(chunk, dtype=np.float32)
                self._append_buffer(chunk)
                if self.show_spectrogram:
                    self._update_spectrogram(chunk)
            except Exception:
                # swallow errors to keep visualizer alive
                time.sleep(0.05)

    def _append_buffer(self, chunk: np.ndarray):
        n = len(chunk)
        if n >= self._buffer_size:
            self._buffer[:] = chunk[-self._buffer_size :]
            self._buf_pos = 0
            return
        end = self._buf_pos + n
        if end <= self._buffer_size:
            self._buffer[self._buf_pos : end] = chunk
        else:
            split = self._buffer_size - self._buf_pos
            self._buffer[self._buf_pos :] = chunk[:split]
            self._buffer[: n - split] = chunk[split:]
        self._buf_pos = (self._buf_pos + n) % self._buffer_size

    def _update_spectrogram(self, chunk: np.ndarray):
        # compute magnitude spectrum for the chunk
        # window and rfft
        window = np.hanning(len(chunk))
        spec = np.abs(np.fft.rfft(chunk * window)) / len(chunk)
        # normalize (small epsilon)
        spec = 20 * np.log10(spec + 1e-9)
        # push into spec_data as last column
        self._spec_data = np.roll(self._spec_data, -1, axis=0)
        # if length mismatch between rfft bins and spec_data columns, crop/resize
        bins = spec.shape[0]
        if bins != self._spec_data.shape[1]:
            # resize the columns to match (simple crop or pad)
            new = np.zeros((self._spec_history, bins), dtype=np.float32)
            minbins = min(bins, self._spec_data.shape[1])
            new[:, :minbins] = self._spec_data[:, :minbins]
            self._spec_data = new
        self._spec_data[-1, : spec.shape[0]] = spec

    def _schedule_ui_update(self):
        if not self._running:
            self._draw()
            return
        self._draw()
        if self._qt_app:
            self._qt_app.processEvents()
        try:
            width = self.plot_container.winfo_width()
            height = self.plot_container.winfo_height()
            if width > 1 and height > 1:
                self.qt_widget.resize(width, height)
        except Exception:
            pass
        self.win.after(self.update_interval, self._schedule_ui_update)

    def _draw(self):
        if self._buf_pos == 0:
            data = self._buffer
        else:
            data = np.concatenate(
                (self._buffer[self._buf_pos :], self._buffer[: self._buf_pos])
            )
        t = np.linspace(-self._buffer_seconds, 0, self._buffer_size)
        
        self._line.setData(t, data)
        
        mx = max(1e-3, np.max(np.abs(self._buffer)))
        self.plot_widget.setYRange(-mx * 1.1, mx * 1.1)
        
        if self.show_spectrogram:
            spec_display = np.flipud(self._spec_data.T)
            self._spec_img.setImage(spec_display, autoLevels=False)
            vmin = np.nanpercentile(spec_display, 1)
            vmax = np.nanpercentile(spec_display, 99)
            self._spec_img.setLevels([vmin, vmax])
            
            from PySide6.QtGui import QTransform
            tr = QTransform()
            tr.scale(self._buffer_seconds / self._spec_history, self.sample_rate / 2 / spec_display.shape[0])
            tr.translate(-self._buffer_seconds, 0)
            self._spec_img.setTransform(tr)
