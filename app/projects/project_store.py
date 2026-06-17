"""On-disk layout for transcription projects.

A project folder looks like::

    <root>/2026-06-02_interview-joao/
        audio.mp3        # copy of the source audio (self-contained)
        transcript.json  # canonical document
        meta.json        # model, language, source name, timestamps, duration

The source audio is copied in so a project survives the original being moved
or deleted.
"""
import json
import logging
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

from app.projects.transcript_document import TranscriptDocument

logger = logging.getLogger("app.projects.store")

TRANSCRIPT_FILENAME = "transcript.json"
META_FILENAME = "meta.json"
_DEFAULT_ROOT_NAME = "WispLive Transcriptions"


@dataclass
class ProjectPaths:
    folder: str
    audio_path: str
    transcript_path: str
    meta_path: str


def default_projects_root() -> str:
    return os.path.join(os.path.expanduser("~"), _DEFAULT_ROOT_NAME)


def _slugify(name: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", name, flags=re.UNICODE).strip().lower()
    slug = re.sub(r"[\s_-]+", "-", slug)
    return slug.strip("-") or "transcription"


def _unique_folder(root: str, base_name: str) -> str:
    candidate = os.path.join(root, base_name)
    counter = 2
    while os.path.exists(candidate):
        candidate = os.path.join(root, f"{base_name}-{counter}")
        counter += 1
    return candidate


def create_project(
    audio_path: str,
    *,
    title: Optional[str] = None,
    root: Optional[str] = None,
    meta: Optional[dict] = None,
    now: Optional[datetime] = None,
) -> ProjectPaths:
    """Create a project folder and copy the source audio into it."""
    root = root or default_projects_root()
    now = now or datetime.now()
    os.makedirs(root, exist_ok=True)

    source_name = os.path.basename(audio_path)
    base_title = title or os.path.splitext(source_name)[0]
    folder_name = f"{now.strftime('%Y-%m-%d')}_{_slugify(base_title)}"
    folder = _unique_folder(root, folder_name)
    os.makedirs(folder)

    ext = os.path.splitext(source_name)[1] or ".audio"
    audio_dest = os.path.join(folder, f"audio{ext}")
    shutil.copy2(audio_path, audio_dest)

    paths = ProjectPaths(
        folder=folder,
        audio_path=audio_dest,
        transcript_path=os.path.join(folder, TRANSCRIPT_FILENAME),
        meta_path=os.path.join(folder, META_FILENAME),
    )

    base_meta = {
        "title": base_title,
        "source_filename": source_name,
        "audio_filename": os.path.basename(audio_dest),
        "created_at": now.isoformat(timespec="seconds"),
        "edited_at": now.isoformat(timespec="seconds"),
        "duration": 0.0,
    }
    base_meta.update(meta or {})
    _write_json(paths.meta_path, base_meta)

    logger.info(f"Created transcription project: {folder}")
    return paths


def save_document(
    paths: ProjectPaths,
    document: TranscriptDocument,
    now: Optional[datetime] = None,
) -> None:
    """Write the canonical transcript and refresh derived meta fields."""
    _write_text(paths.transcript_path, document.to_json())
    meta = read_meta(paths)
    meta["duration"] = document.duration()
    meta["edited_at"] = (now or datetime.now()).isoformat(timespec="seconds")
    _write_json(paths.meta_path, meta)
    logger.info(f"Saved transcript: {paths.transcript_path}")


def read_meta(paths: ProjectPaths) -> dict:
    if not os.path.exists(paths.meta_path):
        return {}
    with open(paths.meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_project(folder: str) -> Tuple[ProjectPaths, TranscriptDocument, dict]:
    """Load an existing project folder into its paths, document, and meta."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Not a transcription project folder: {folder}")
    transcript_path = os.path.join(folder, TRANSCRIPT_FILENAME)
    if not os.path.isfile(transcript_path):
        raise FileNotFoundError(f"No {TRANSCRIPT_FILENAME} in project folder: {folder}")

    meta_path = os.path.join(folder, META_FILENAME)
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    audio_filename = meta.get("audio_filename")
    audio_path = (
        os.path.join(folder, audio_filename)
        if audio_filename
        else _find_audio(folder)
    )

    with open(transcript_path, "r", encoding="utf-8") as f:
        document = TranscriptDocument.from_json(f.read())

    paths = ProjectPaths(
        folder=folder,
        audio_path=audio_path,
        transcript_path=transcript_path,
        meta_path=meta_path,
    )
    return paths, document, meta


def _find_audio(folder: str) -> Optional[str]:
    for name in sorted(os.listdir(folder)):
        if name.startswith("audio."):
            return os.path.join(folder, name)
    return None


def _write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _write_json(path: str, data: dict) -> None:
    _write_text(path, json.dumps(data, ensure_ascii=False, indent=2))
