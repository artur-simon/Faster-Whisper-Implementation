import threading
import time
import queue
from typing import Callable, Optional

import numpy as np

import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QHBoxLayout, QLabel, QApplication
from PySide6.QtCore import QTimer

from app.audio.audio_data_provider import AudioDataProvider

class AudioVisualizer(QWidget):
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

    _app = None

    @staticmethod
    def _ensure_app():
        if QApplication.instance() is None:
            AudioVisualizer._app = QApplication([])
        else:
            AudioVisualizer._app = QApplication.instance()
            
    def __init__(
        self,
        sample_rate: int = 44100,
        chunk_samples: int = 1024,
        update_interval: int = 40,
        show_spectrogram: bool = False,
    ):
        self._ensure_app() 
        super().__init__()
        
        self.sample_rate = sample_rate
        self.chunk_samples = chunk_samples
        self.update_interval = update_interval
        self.show_spectrogram = show_spectrogram
        
        self._audio_data_provider = AudioDataProvider(sample_rate, device_index=2)
        
        self._thread: Optional[threading.Thread] = None
        
        self._running = False
        self._init_state()
        self._init_ui()
        self._init_plot()
        self._init_timer()
        
    def _init_state(self):
        self._buffer_seconds = 2.0
        self._buffer_size = int(self.sample_rate * self._buffer_seconds)
        self._buffer = np.zeros(self._buffer_size, dtype=np.float32)
        self._buf_pos = 0
        self._buffer_lock = threading.Lock()
        
        # spectrogram state (history of spectra columns)
        if self.show_spectrogram:
            self._spec_queue = queue.Queue(maxsize=10)
            resolution = self.chunk_samples // 2 + 1
            
            # number of time columns to keep
            self._spec_history = 200
            self._spec_data = np.zeros(
                (self._spec_history, resolution), dtype=np.float32
            )        
            #spectogram pre-calculations
            self._hann = np.hanning(self.chunk_samples).astype(np.float32)
            self._fft_in = np.zeros(self.chunk_samples, dtype=np.float32)
            self._fft_out = np.zeros(resolution, dtype=np.complex64)
            self._mag = np.zeros(resolution, dtype=np.float32)
            self._spec_db = np.zeros(resolution, dtype=np.float32)
            
            freqs_lin = np.linspace(0, self.sample_rate/2, resolution)
            freqs_log = np.geomspace(20.0, self.sample_rate/2, resolution)
            self._interp_index = np.interp(freqs_log, freqs_lin, 
                                           np.arange(freqs_lin.size)
                                           )
            
        self._time_axis = np.linspace(-self._buffer_seconds, 0, self._buffer_size)


            
    def _init_ui(self):
        layout = QVBoxLayout(self)
        ctrl = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        ctrl.addWidget(self.start_btn)
        ctrl.addWidget(self.stop_btn)
        ctrl.addWidget(QLabel(f"SR: {self.sample_rate} Hz"))
        layout.addLayout(ctrl)

        # plot container
        self.plot_container = QVBoxLayout()
        layout.addLayout(self.plot_container)

        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)
        
    def _init_plot(self):
        pg.setConfigOptions(antialias=True, useOpenGL=True)
        
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', 'amplitude')
        self.plot_widget.setLabel('bottom', 'seconds')
        self.plot_widget.setXRange(-self._buffer_seconds, 0)
        self.plot_widget.setYRange(-1.0, 1.0)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
        self.plot_container.addWidget(self.plot_widget)
        
        if self.show_spectrogram:
            self.spec_widget = pg.PlotWidget()
            self.spec_widget.setLabel('left', 'frequency', units='Hz')
            self.spec_widget.setLabel('bottom', 'seconds')
            self.spec_widget.setXRange(-self._buffer_seconds, 0)
            self.spec_widget.setYRange(0, self.sample_rate / 2)
            self.spec_widget.showGrid(x=True, y=True, alpha=0.3)
            self.plot_container.addWidget(self.spec_widget)
            
            self._spec_img = pg.ImageItem(autoDownsample=True)
            self.spec_widget.addItem(self._spec_img)
            self._spec_img.setLookupTable(pg.colormap.get('viridis').getLookupTable())
        
        t = np.linspace(-self._buffer_seconds, 0, self._buffer_size)
        self._line = self.plot_widget.plot(t, self._buffer, pen='c')
    
    def _init_timer(self):
        self.timer = QTimer()
        self.timer.setInterval(self.update_interval)
        self.timer.timeout.connect(self._draw_once)
        
    def start(self):
        if self._running:
            return
        self._running = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self._audio_data_provider.start()
        
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        self.timer.start()

    def stop(self):
        self._running = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.timer.stop()
    
    def _poll_loop(self):
        """
        Background thread: pull audio chunks quickly and append to rolling buffer.
        Do heavy CPU work here (FFT) and push only results to UI via a small queue.
        """
        while self._running:
            try:
                chunk = self._audio_data_provider.get_audio_chunk(self.chunk_samples)
                if chunk is None or len(chunk) == 0:
                    time.sleep(0.01)
                    continue
                chunk = np.asarray(chunk, dtype=np.float32)

                # append to circular buffer under lock
                with self._buffer_lock:
                    self._append_buffer_no_lock(chunk)

                # compute spectrogram column and push to queue
                if self.show_spectrogram:
                    self._fft_in[:] = chunk
                    self._fft_in *= self._hann
                    self._fft_out[:] = np.fft.rfft(self._fft_in)
                    self._mag[:] = np.abs(self._fft_out) * (1.0 / self.chunk_samples)
                    np.log10(self._mag + 1e-9, out=self._spec_db)
                    self._spec_db *= 20.0
                    
                    # push latest column (avoid blocking if queue full)
                    try:
                        self._spec_queue.put_nowait(self._spec_db.copy())
                    except queue.Full:
                        try:
                            _ = self._spec_queue.get_nowait()
                            self._spec_queue.put_nowait(self._spec_db.copy())
                        except Exception:
                            pass
            except Exception:
                time.sleep(0.05)

    def _append_buffer_no_lock(self, chunk: np.ndarray):
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

    def _draw_once(self):
        # copy buffer under lock to avoid reading while writer updates it
        with self._buffer_lock:
            if self._buf_pos == 0:
                data = self._buffer.copy()
            else:
                # reconstruct in correct order
                a = self._buffer[self._buf_pos:]
                b = self._buffer[:self._buf_pos]
                data = np.concatenate((a, b))

        self._line.setData(self._time_axis, data)

        mx = max(1e-3, np.max(np.abs(data)))
        #self.plot_widget.setYRange(-mx * 1.1, mx * 1.1)

        if self.show_spectrogram:
            drained = 0
            max_drain = 4
            while drained < max_drain:
                try:
                    col = self._spec_queue.get_nowait()
                except queue.Empty:
                    break
                # roll and insert; keep this operation cheap
                self._spec_data = np.roll(self._spec_data, -1, axis=0)
                bins = col.shape[0]
                #interpolate to log
                self._spec_data[-1 , :bins] = np.take(col, self._interp_index.astype(np.int32)) 
                drained += 1

            spec_display = np.transpose(self._spec_data.T)

            self._spec_img.setImage(spec_display, autoLevels=False)
            self._spec_img.setLevels([-125, -14])
            
            from PySide6.QtGui import QTransform
            tr = QTransform()
            tr.scale(self._buffer_seconds / self._spec_history, 
                     self.sample_rate / 2 / spec_display.shape[1])
            tr.translate(-self._spec_history, 0)
            self._spec_img.setTransform(tr)