from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi.testclient import TestClient

import main
from app.api.routes import captions as captions_routes
from app.database import get_db


async def _fake_db() -> AsyncIterator[object]:
    yield object()


@asynccontextmanager
async def _noop_lifespan(_app):
    yield


def test_get_captions_ok(monkeypatch):
    async def fake_get_clip_captions_payload(_db, *, project_id, local_key):
        assert project_id == 1
        assert local_key == "lk-1"
        return "ok", {
            "project_id": 1,
            "local_key": "lk-1",
            "clip_id": 99,
            "transcript_id": 55,
            "processing_status": "ready",
            "full_text": "Hello.",
            "sentences": [{"text": "Hello.", "start": 0.0, "end": 0.5, "words": []}],
            "meta": {"source_file": "a.mp4", "mime_type": "video/mp4", "extension": "mp4", "clip_meta": {}, "video_report": {}},
        }

    monkeypatch.setattr(captions_routes, "get_clip_captions_payload", fake_get_clip_captions_payload)
    main.app.dependency_overrides[get_db] = _fake_db
    original_lifespan = main.app.router.lifespan_context
    main.app.router.lifespan_context = _noop_lifespan

    try:
        with TestClient(main.app) as client:
            response = client.get("/captions", params={"project_id": 1, "local_key": "lk-1"})
    finally:
        main.app.router.lifespan_context = original_lifespan
        main.app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["clip_id"] == 99
    assert data["sentences"][0]["text"] == "Hello."


def test_get_captions_404_project(monkeypatch):
    async def fake_get_clip_captions_payload(_db, *, project_id, local_key):
        return "project_not_found", None

    monkeypatch.setattr(captions_routes, "get_clip_captions_payload", fake_get_clip_captions_payload)
    main.app.dependency_overrides[get_db] = _fake_db
    original_lifespan = main.app.router.lifespan_context
    main.app.router.lifespan_context = _noop_lifespan

    try:
        with TestClient(main.app) as client:
            response = client.get("/captions", params={"project_id": 999, "local_key": "x"})
    finally:
        main.app.router.lifespan_context = original_lifespan
        main.app.dependency_overrides.clear()

    assert response.status_code == 404


def test_get_captions_409_no_transcript(monkeypatch):
    async def fake_get_clip_captions_payload(_db, *, project_id, local_key):
        return "transcript_not_ready", {"clip_id": 1, "processing_status": "processing"}

    monkeypatch.setattr(captions_routes, "get_clip_captions_payload", fake_get_clip_captions_payload)
    main.app.dependency_overrides[get_db] = _fake_db
    original_lifespan = main.app.router.lifespan_context
    main.app.router.lifespan_context = _noop_lifespan

    try:
        with TestClient(main.app) as client:
            response = client.get("/captions", params={"project_id": 1, "local_key": "lk"})
    finally:
        main.app.router.lifespan_context = original_lifespan
        main.app.dependency_overrides.clear()

    assert response.status_code == 409
