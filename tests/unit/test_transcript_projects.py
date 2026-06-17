import os
from datetime import datetime

import pytest

from app.models import Word, TranscriptionSegment
from app.projects import exporters, project_store
from app.projects.transcript_document import TranscriptDocument, SCHEMA_VERSION


def _sample_document():
    return TranscriptDocument.from_segments([
        TranscriptionSegment(
            text="Hello world",
            words=[
                Word(text="Hello", start=0.0, end=0.5, probability=0.99),
                Word(text=" world", start=0.5, end=1.2, probability=0.88),
            ],
            no_speech_prob=0.01,
        ),
        TranscriptionSegment(
            text="Segunda frase",
            words=[
                Word(text="Segunda", start=2.0, end=2.6, probability=0.95),
                Word(text=" frase", start=2.6, end=3.1, probability=0.91),
            ],
            no_speech_prob=0.02,
        ),
    ])


class TestTranscriptDocument:
    def test_json_round_trip_preserves_structure(self):
        doc = _sample_document()
        restored = TranscriptDocument.from_json(doc.to_json())

        assert restored.version == SCHEMA_VERSION
        assert restored.segments == doc.segments  # frozen dataclasses compare by value

    def test_to_dict_carries_word_timestamps_and_confidence(self):
        doc = _sample_document()
        data = doc.to_dict()
        first_word = data["segments"][0]["words"][0]

        assert first_word == {
            "text": "Hello",
            "start": 0.0,
            "end": 0.5,
            "probability": 0.99,
        }

    def test_duration_is_latest_word_end(self):
        assert _sample_document().duration() == 3.1

    def test_full_text_joins_segments_per_line(self):
        assert _sample_document().full_text() == "Hello world\nSegunda frase"

    def test_from_dict_tolerates_missing_word_timings(self):
        doc = TranscriptDocument.from_dict({
            "segments": [{"text": "x", "words": [{"text": "x"}]}]
        })
        assert doc.segments[0].words[0].start is None


class TestExporters:
    def test_to_txt(self):
        assert exporters.to_txt(_sample_document()) == "Hello world\nSegunda frase\n"

    def test_to_srt_format(self):
        srt = exporters.to_srt(_sample_document())
        assert "1\n00:00:00,000 --> 00:00:01,200\nHello world" in srt
        assert "2\n00:00:02,000 --> 00:00:03,100\nSegunda frase" in srt

    def test_to_vtt_format(self):
        vtt = exporters.to_vtt(_sample_document())
        assert vtt.startswith("WEBVTT")
        assert "00:00:00.000 --> 00:00:01.200\nHello world" in vtt

    def test_segments_without_timings_are_skipped_in_subtitles(self):
        doc = TranscriptDocument.from_segments([
            TranscriptionSegment(text="no timing", words=[], no_speech_prob=0.0)
        ])
        assert exporters.to_srt(doc) == ""
        assert exporters.to_vtt(doc).strip() == "WEBVTT"


class TestProbeDuration:
    def test_probe_real_fixture(self):
        av = pytest.importorskip("av")
        dur = project_store.probe_duration("tests/fixtures/longa_pt_2min.wav")
        assert dur is not None and 120 < dur < 130

    def test_probe_missing_file_returns_none(self):
        assert project_store.probe_duration("does/not/exist.wav") is None


class TestImportAudio:
    def test_remux_real_fixture_keeps_duration(self, tmp_path):
        pytest.importorskip("av")
        dest = str(tmp_path / "audio.wav")
        project_store._import_audio("tests/fixtures/longa_pt_2min.wav", dest)
        assert os.path.isfile(dest)
        dur = project_store.probe_duration(dest)
        assert dur is not None and 120 < dur < 130

    def test_falls_back_to_copy_for_non_audio(self, tmp_path):
        src = tmp_path / "junk.mp3"
        src.write_bytes(b"not really audio")
        dest = str(tmp_path / "audio.mp3")
        project_store._import_audio(str(src), dest)
        assert os.path.isfile(dest)
        with open(dest, "rb") as f:
            assert f.read() == b"not really audio"


class TestProjectStore:
    def test_create_project_copies_audio_and_writes_meta(self, tmp_path):
        src = tmp_path / "interview joão.mp3"
        src.write_bytes(b"fake-audio-bytes")

        paths = project_store.create_project(
            str(src),
            root=str(tmp_path / "projects"),
            meta={"model_size": "medium", "language": "pt"},
            now=datetime(2026, 6, 2, 10, 30),
        )

        assert os.path.basename(paths.folder) == "2026-06-02_interview-joão"
        assert os.path.isfile(paths.audio_path)
        assert paths.audio_path.endswith("audio.mp3")

        meta = project_store.read_meta(paths)
        assert meta["source_filename"] == "interview joão.mp3"
        assert meta["model_size"] == "medium"
        assert meta["language"] == "pt"

    def test_unique_folder_on_collision(self, tmp_path):
        src = tmp_path / "a.mp3"
        src.write_bytes(b"x")
        root = str(tmp_path / "projects")
        now = datetime(2026, 6, 2)

        first = project_store.create_project(str(src), root=root, now=now)
        second = project_store.create_project(str(src), root=root, now=now)

        assert first.folder != second.folder
        assert second.folder.endswith("-2")

    def test_save_and_load_round_trip(self, tmp_path):
        src = tmp_path / "talk.wav"
        src.write_bytes(b"x")
        paths = project_store.create_project(str(src), root=str(tmp_path / "p"))

        project_store.save_document(paths, _sample_document())
        loaded_paths, loaded_doc, meta = project_store.load_project(paths.folder)

        assert loaded_doc.segments == _sample_document().segments
        assert meta["duration"] == 3.1
        assert loaded_paths.audio_path.endswith("audio.wav")
