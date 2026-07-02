"""File Transcription Studio — a standalone PySide6 window.

Opens a transcription project folder (created by
``TranscriptionController.transcribe_audio_file_to_project``) and lets you read
the transcript while playing the source audio: click any word to jump there,
and the word being spoken stays highlighted.

Run standalone::

    python -m app.ui.qt.transcription_studio "<project_folder>"
"""
import logging
import os
import sys

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QAction
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from app.projects import exporters, project_store
from app.ui.qt.transcript_view import TranscriptView

logger = logging.getLogger("app.ui.qt.studio")


def _fmt_time(ms: int) -> str:
    seconds = max(0, ms) // 1000
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


class TranscriptionStudio(QMainWindow):
    def __init__(self, folder: str):
        super().__init__()
        self._paths, self._document, self._meta = project_store.load_project(folder)

        # Timeline length floor. The player often underreports duration because
        # container metadata estimates are short, while meta.duration reflects
        # what faster-whisper actually decoded. We take the max of the two (and
        # grow it from playback position) so the slider/label cover real content.
        self._duration_ms = int(float(self._meta.get("duration") or 0) * 1000)

        title = self._meta.get("title") or os.path.basename(os.path.normpath(folder))
        self.setWindowTitle(f"WispLive — Transcription Studio — {title}")
        self.resize(900, 720)

        self._init_player()
        self._init_menu()
        self._init_ui()
        self._load_audio()

    # ---- setup ---------------------------------------------------------
    def _init_player(self) -> None:
        self._player = QMediaPlayer(self)
        self._audio_out = QAudioOutput(self)
        self._player.setAudioOutput(self._audio_out)
        self._player.positionChanged.connect(self._on_position)
        self._player.durationChanged.connect(self._on_duration)
        self._player.playbackStateChanged.connect(self._on_state_changed)

    def _init_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        for label, suffix, formatter in (
            ("Export as &TXT…", "txt", exporters.to_txt),
            ("Export as &SRT…", "srt", exporters.to_srt),
            ("Export as &VTT…", "vtt", exporters.to_vtt),
        ):
            action = QAction(label, self)
            action.triggered.connect(
                lambda checked=False, s=suffix, f=formatter: self._export(s, f)
            )
            file_menu.addAction(action)
        file_menu.addSeparator()
        close_action = QAction("&Close", self)
        close_action.triggered.connect(self.close)
        file_menu.addAction(close_action)

    def _init_ui(self) -> None:
        central = QWidget(self)
        layout = QVBoxLayout(central)

        # transport bar
        bar = QHBoxLayout()
        self._play_btn = QPushButton("▶ Play")
        self._play_btn.setFixedWidth(90)
        self._play_btn.clicked.connect(self._toggle_play)
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, self._duration_ms)
        self._slider.sliderMoved.connect(self._player.setPosition)
        self._time_label = QLabel(f"00:00 / {_fmt_time(self._duration_ms)}")
        bar.addWidget(self._play_btn)
        bar.addWidget(self._slider, 1)
        bar.addWidget(self._time_label)
        layout.addLayout(bar)

        # transcript
        self._view = TranscriptView(self._document, parent=central)
        self._view.word_seek_requested.connect(self._seek_seconds)
        layout.addWidget(self._view, 1)

        # status line
        status = QLabel(self._status_text())
        status.setStyleSheet("color: gray; padding: 2px;")
        layout.addWidget(status)

        self.setCentralWidget(central)

    def _status_text(self) -> str:
        bits = []
        if self._meta.get("model_size"):
            bits.append(f"model: {self._meta['model_size']}")
        if self._meta.get("language"):
            bits.append(f"lang: {self._meta['language']}")
        if self._meta.get("duration"):
            bits.append(f"duration: {_fmt_time(int(self._meta['duration'] * 1000))}")
        bits.append("amber = low confidence · click a word to jump")
        return "   |   ".join(bits)

    def _load_audio(self) -> None:
        audio_path = self._paths.audio_path
        if not audio_path or not os.path.isfile(audio_path):
            logger.warning("Audio file missing for project: %s", self._paths.folder)
            self._play_btn.setEnabled(False)
            self._play_btn.setText("no audio")
            return
        self._player.setSource(QUrl.fromLocalFile(audio_path))

    # ---- transport handlers --------------------------------------------
    def _toggle_play(self) -> None:
        if self._player.playbackState() == QMediaPlayer.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    def _seek_seconds(self, seconds: float) -> None:
        self._player.setPosition(int(seconds * 1000))
        if self._player.playbackState() != QMediaPlayer.PlayingState:
            self._player.play()

    def _on_position(self, ms: int) -> None:
        if ms > self._duration_ms:
            self._set_timeline(ms)
        if not self._slider.isSliderDown():
            self._slider.setValue(ms)
        self._update_time_label(ms)
        self._view.highlight_at_time(ms / 1000.0)

    def _on_duration(self, ms: int) -> None:
        # Never shrink below the meta/transcript length — the player commonly
        # reports a short duration for files whose metadata underreports.
        self._set_timeline(max(self._duration_ms, ms))
        self._update_time_label(self._player.position())

    def _set_timeline(self, ms: int) -> None:
        self._duration_ms = ms
        self._slider.setRange(0, ms)

    def _update_time_label(self, ms: int) -> None:
        self._time_label.setText(f"{_fmt_time(ms)} / {_fmt_time(self._duration_ms)}")

    def _on_state_changed(self, state) -> None:
        playing = state == QMediaPlayer.PlayingState
        self._play_btn.setText("⏸ Pause" if playing else "▶ Play")

    # ---- export --------------------------------------------------------
    def _export(self, suffix: str, formatter) -> None:
        # Default next to the project, named after the (filesystem-safe) folder.
        base = os.path.basename(os.path.normpath(self._paths.folder))
        default_path = os.path.join(self._paths.folder, f"{base}.{suffix}")
        path, _ = QFileDialog.getSaveFileName(
            self,
            f"Export as {suffix.upper()}",
            default_path,
            f"{suffix.upper()} files (*.{suffix});;All files (*.*)",
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(formatter(self._document))
        except OSError as e:
            logger.error("Export failed (%s): %s", path, e, exc_info=True)
            QMessageBox.critical(self, "Export failed", f"Could not write file:\n{e}")
            return
        logger.info("Exported %s to %s", suffix.upper(), path)
        self.statusBar().showMessage(f"Exported {os.path.basename(path)}", 5000)


def launch_studio(folder: str):
    """Launch the Studio in its own process (own Qt event loop).

    Prefers ``pythonw.exe`` so launching from the Tk app doesn't pop a console
    window. ``Popen`` with a list quotes arguments correctly, so folder paths
    containing spaces are passed through intact.
    """
    import subprocess

    if getattr(sys, "frozen", False):
        # Frozen (PyInstaller): ``sys.executable`` is the bundled app, not a
        # Python interpreter, so ``-m module`` is meaningless — the bootloader
        # would just relaunch the main app. Re-launch ourselves with a flag
        # the entry point dispatches to the Studio instead.
        return subprocess.Popen([sys.executable, "--studio", folder])

    exe = sys.executable
    gui_exe = os.path.join(os.path.dirname(exe), "pythonw.exe")
    if os.path.isfile(gui_exe):
        exe = gui_exe
    return subprocess.Popen([exe, "-m", "app.ui.qt.transcription_studio", folder])


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv
    if len(argv) < 2:
        print('usage: python -m app.ui.qt.transcription_studio "<project_folder>"')
        return 2
    app = QApplication(argv)
    window = TranscriptionStudio(argv[1])
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
