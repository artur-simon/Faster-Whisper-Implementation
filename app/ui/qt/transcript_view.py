"""Read-only transcript widget with click-to-seek and playback highlighting.

Each word is laid out as flowing text. Clicking a word emits its start time so
the player can seek there; :meth:`highlight_at_time` highlights the word being
spoken as playback advances. Low-confidence words are tinted so suspect
transcription is easy to spot — the core of the "read it, it looked off, let
me hear it" workflow.
"""
import bisect
from collections import namedtuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import QTextEdit

from app.projects.transcript_document import TranscriptDocument

# faster-whisper word probabilities sit ~0.9+ for confident words; below this
# we treat a word as worth a second listen.
LOW_CONFIDENCE_THRESHOLD = 0.6

WordSpan = namedtuple("WordSpan", ["cs", "ce", "ts", "te"])


class TranscriptView(QTextEdit):
    word_seek_requested = Signal(float)  # word start time, seconds

    def __init__(self, document: TranscriptDocument, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMouseTracking(True)
        self.setLineWrapMode(QTextEdit.WidgetWidth)

        self._spans: list[WordSpan] = []
        self._timed_starts: list[float] = []   # sorted word start times
        self._timed_index: list[int] = []      # span index per entry above
        self._active_index = -1
        self._active_color = QColor(120, 170, 255, 160)  # blue current-word
        self._low_color = QColor(255, 224, 138)          # amber low-confidence

        self._build(document)

    def _build(self, document: TranscriptDocument) -> None:
        cursor = self.textCursor()
        plain_fmt = QTextCharFormat()
        low_fmt = QTextCharFormat()
        low_fmt.setBackground(self._low_color)

        for segment in document.segments:
            for word in segment.words:
                start = cursor.position()
                is_low = word.probability is not None and word.probability < LOW_CONFIDENCE_THRESHOLD
                cursor.insertText(word.text, low_fmt if is_low else plain_fmt)
                end = cursor.position()
                self._spans.append(WordSpan(start, end, word.start, word.end))
            cursor.insertText("\n", plain_fmt)

        timed = sorted(
            (i for i, s in enumerate(self._spans) if s.ts is not None),
            key=lambda i: self._spans[i].ts,
        )
        self._timed_index = list(timed)
        self._timed_starts = [self._spans[i].ts for i in timed]

    # ---- click-to-seek -------------------------------------------------
    def mousePressEvent(self, event) -> None:
        super().mousePressEvent(event)
        if event.button() != Qt.LeftButton:
            return
        pos = self.cursorForPosition(event.position().toPoint()).position()
        for span in self._spans:
            if span.cs <= pos < span.ce and span.ts is not None:
                self.word_seek_requested.emit(span.ts)
                return

    # ---- playback highlight --------------------------------------------
    def highlight_at_time(self, seconds: float) -> None:
        idx = self._span_index_at_time(seconds)
        if idx == self._active_index:
            return
        self._active_index = idx
        self._apply_highlight(idx)

    def _span_index_at_time(self, t: float) -> int:
        if not self._timed_starts:
            return -1
        pos = bisect.bisect_right(self._timed_starts, t) - 1
        if pos < 0:
            return -1
        return self._timed_index[pos]

    def _apply_highlight(self, span_idx: int) -> None:
        selections = []
        if 0 <= span_idx < len(self._spans):
            span = self._spans[span_idx]
            sel = QTextEdit.ExtraSelection()
            fmt = QTextCharFormat()
            fmt.setBackground(self._active_color)
            sel.format = fmt
            cur = self.textCursor()
            cur.setPosition(span.cs)
            cur.setPosition(span.ce, QTextCursor.KeepAnchor)
            sel.cursor = cur
            selections.append(sel)
            self._auto_scroll(span.cs)
        self.setExtraSelections(selections)

    def _auto_scroll(self, pos: int) -> None:
        collapsed = self.textCursor()
        collapsed.setPosition(pos)
        rect = self.cursorRect(collapsed)
        viewport = self.viewport().rect()
        if rect.top() < viewport.top() or rect.bottom() > viewport.bottom():
            self.setTextCursor(collapsed)
            self.ensureCursorVisible()
