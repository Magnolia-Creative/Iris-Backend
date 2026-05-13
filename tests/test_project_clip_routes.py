from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi.testclient import TestClient

import main
from app.database import get_db


async def _fake_db() -> AsyncIterator[object]:
    yield object()


@asynccontextmanager
async def _noop_lifespan(_app):
    yield


def test_create_sentence_transcription_route(monkeypatch):
    printed_lines: list[str] = []

    async def fake_transcribe_upload_to_sentences(audio_bytes, suffix=".m4a"):
        assert audio_bytes == b"audio"
        assert suffix == ".m4a"
        return {
            "transcript_id": "tr-123",
            "full_text": "Hello world.",
            "sentences": [
                {
                    "text": "Hello world.",
                    "start": 0.0,
                    "end": 1.2,
                    "confidence": 0.99,
                }
            ],
            "language_code": "en",
            "confidence": 0.99,
            "audio_duration": 1.2,
            "status": "completed",
            "meta": {"provider": "modal"},
        }

    async def fake_insert_sentence_upload(_db, result):
        assert result["full_text"] == "Hello world."
        return "aaaaaaaa-bbbb-4ccc-a123-456789abcdef"

    def fake_print(*args, **kwargs):
        printed_lines.append(" ".join(str(arg) for arg in args))

    monkeypatch.setattr(main, "transcribe_upload_to_sentences", fake_transcribe_upload_to_sentences)
    monkeypatch.setattr(main, "insert_sentence_upload_transcript", fake_insert_sentence_upload)
    monkeypatch.setattr("builtins.print", fake_print)
    original_lifespan = main.app.router.lifespan_context
    main.app.router.lifespan_context = _noop_lifespan
    main.app.dependency_overrides[get_db] = _fake_db

    try:
        with TestClient(main.app) as client:
            response = client.post(
                "/transcriptions/sentences",
                files={"audio": ("clip.m4a", b"audio", "audio/mp4")},
            )
    finally:
        main.app.router.lifespan_context = original_lifespan
        main.app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["transcript_id"] == "aaaaaaaa-bbbb-4ccc-a123-456789abcdef"
    assert payload["meta"]["provider_transcript_id"] == "tr-123"
    assert payload["full_text"] == "Hello world."
    assert payload["sentences"][0]["text"] == "Hello world."
    assert payload["meta"]["provider"] == "modal"
    assert any(
        "[TRANSCRIPT_RECEIVED][main_endpoint]" in line and "Hello world." in line
        for line in printed_lines
    )


def test_create_project_agent_session_route(monkeypatch):
    async def fake_create_agent_session(db, *, session_name=None, project_name=None):
        assert session_name is None
        assert project_name == "Launch Day"
        return {
            "session_id": 7,
            "session_name": "Launch Day",
            "session_status": "created",
            "project_id": 11,
            "project_name": "Launch Day",
            "uploaded_count": 0,
            "pending_clip_count": 0,
            "settled_clip_count": 0,
            "ready_for_websocket": False,
            "videos": [],
        }

    monkeypatch.setattr(main, "create_agent_session", fake_create_agent_session)
    main.app.dependency_overrides[get_db] = _fake_db
    original_lifespan = main.app.router.lifespan_context
    main.app.router.lifespan_context = _noop_lifespan

    try:
        with TestClient(main.app) as client:
            response = client.post("/projects/agent-sessions", json={"project_name": "Launch Day"})
    finally:
        main.app.router.lifespan_context = original_lifespan

    main.app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()["project_id"] == 11
    assert response.json()["session_name"] == "Launch Day"


def test_create_project_route(monkeypatch):
    async def fake_create_project(db, *, name=None):
        assert name == "My Doc"
        return {"project_id": 42, "project_name": "My Doc"}

    monkeypatch.setattr(main, "create_project", fake_create_project)
    main.app.dependency_overrides[get_db] = _fake_db
    original_lifespan = main.app.router.lifespan_context
    main.app.router.lifespan_context = _noop_lifespan

    try:
        with TestClient(main.app) as client:
            response = client.post("/projects", json={"name": "My Doc"})
    finally:
        main.app.router.lifespan_context = original_lifespan

    main.app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json() == {"project_id": 42, "project_name": "My Doc"}


def test_get_project_clips_status_route(monkeypatch):
    async def fake_get_persisted_project_data(db, project_id, *, include_ingest_details=False):
        assert project_id == 11
        assert include_ingest_details is False
        return {
            "session_id": None,
            "session_name": None,
            "session_status": "ready",
            "project_id": 11,
            "project_name": "P",
            "uploaded_count": 0,
            "pending_clip_count": 0,
            "settled_clip_count": 0,
            "ready_for_websocket": False,
            "videos": [],
        }

    monkeypatch.setattr(main, "get_persisted_project_data", fake_get_persisted_project_data)
    main.app.dependency_overrides[get_db] = _fake_db
    original_lifespan = main.app.router.lifespan_context
    main.app.router.lifespan_context = _noop_lifespan

    try:
        with TestClient(main.app) as client:
            response = client.get("/projects/11/clips/status")
    finally:
        main.app.router.lifespan_context = original_lifespan

    main.app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()["project_id"] == 11


def test_create_agent_session_for_project_route(monkeypatch):
    async def fake_create_agent_session_for_project(db, *, project_id, session_name=None):
        assert project_id == 11
        assert session_name == "Edit run"
        return {
            "session_id": 9,
            "session_name": "Edit run",
            "session_status": "created",
            "project_id": 11,
            "project_name": "Launch Day",
            "uploaded_count": 0,
            "pending_clip_count": 0,
            "settled_clip_count": 0,
            "ready_for_websocket": False,
            "videos": [],
        }

    monkeypatch.setattr(main, "create_agent_session_for_project", fake_create_agent_session_for_project)
    main.app.dependency_overrides[get_db] = _fake_db
    original_lifespan = main.app.router.lifespan_context
    main.app.router.lifespan_context = _noop_lifespan

    try:
        with TestClient(main.app) as client:
            response = client.post(
                "/projects/11/agent-sessions",
                json={"session_name": "Edit run"},
            )
    finally:
        main.app.router.lifespan_context = original_lifespan

    main.app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()["session_id"] == 9
    assert response.json()["project_id"] == 11


def test_process_project_clip_batch_route(monkeypatch):
    async def fake_process_project_clips(
        db,
        *,
        project_id,
        session_id,
        videos,
        local_keys,
        visual_frames_by_local_key=None,
    ):
        assert project_id == 11
        assert session_id is None
        assert local_keys == ["abc-123"]
        assert len(videos) == 1
        assert visual_frames_by_local_key is None
        return {
            "session_id": None,
            "session_name": None,
            "session_status": "processing",
            "project_id": 11,
            "project_name": "Launch Day",
            "uploaded_count": 1,
            "pending_clip_count": 1,
            "settled_clip_count": 0,
            "ready_for_websocket": False,
            "videos": [
                {
                    "index": 1,
                    "session_id": None,
                    "project_id": 11,
                    "clip_id": 99,
                    "transcript_id": None,
                    "local_key": "abc-123",
                    "file_name": "clip.m4a",
                    "mime_type": "audio/mp4",
                    "extension": ".m4a",
                    "processing_status": "processing",
                    "processing_error": None,
                }
            ],
            "vector_index": {"status": "scheduled", "scheduled_clip_count": 1},
        }

    monkeypatch.setattr(main, "process_project_clips", fake_process_project_clips)
    main.app.dependency_overrides[get_db] = _fake_db
    original_lifespan = main.app.router.lifespan_context
    main.app.router.lifespan_context = _noop_lifespan

    try:
        with TestClient(main.app) as client:
            response = client.post(
                "/projects/11/clips/process",
                data={"local_key": "abc-123"},
                files={"videos": ("clip.m4a", b"audio", "audio/mp4")},
            )
    finally:
        main.app.router.lifespan_context = original_lifespan

    main.app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()["ready_for_websocket"] is False
    assert response.json()["pending_clip_count"] == 1
    assert response.json()["videos"][0]["local_key"] == "abc-123"
    assert "file_size_bytes" not in response.json()["videos"][0]
    assert "transcript_segments" not in response.json()["videos"][0]
    assert "transcript_full_text" not in response.json()["videos"][0]
    assert "video_report" not in response.json()["videos"][0]
    assert "clip_meta" not in response.json()["videos"][0]


def test_get_session_status_route(monkeypatch):
    async def fake_get_persisted_session_data(db, session_id, *, include_ingest_details=False):
        assert session_id == 7
        assert include_ingest_details is False
        return {
            "session_id": 7,
            "session_name": "Launch Day",
            "session_status": "ready",
            "project_id": 11,
            "project_name": "Launch Day",
            "uploaded_count": 1,
            "pending_clip_count": 0,
            "settled_clip_count": 1,
            "ready_for_websocket": True,
            "videos": [
                {
                    "index": 1,
                    "session_id": 7,
                    "project_id": 11,
                    "clip_id": 99,
                    "transcript_id": 501,
                    "local_key": "abc-123",
                    "file_name": "clip.m4a",
                    "mime_type": "audio/mp4",
                    "extension": ".m4a",
                    "processing_status": "ready",
                    "processing_error": None,
                }
            ],
        }

    monkeypatch.setattr(main, "get_persisted_session_data", fake_get_persisted_session_data)
    main.app.dependency_overrides[get_db] = _fake_db
    original_lifespan = main.app.router.lifespan_context
    main.app.router.lifespan_context = _noop_lifespan

    try:
        with TestClient(main.app) as client:
            response = client.get("/sessions/7")
    finally:
        main.app.router.lifespan_context = original_lifespan

    main.app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()["session_status"] == "ready"
    assert response.json()["ready_for_websocket"] is True


def test_cancel_project_clip_route(monkeypatch):
    async def fake_cancel_clip_processing(db, *, project_id, session_id, local_key):
        assert project_id == 11
        assert session_id is None
        assert local_key == "abc-123"
        return {
            "session_id": None,
            "project_id": 11,
            "local_key": "abc-123",
            "task_cancelled": True,
            "deleted_clip_id": 99,
            "session_status": "created",
            "ready_for_websocket": False,
        }

    monkeypatch.setattr(main, "cancel_clip_processing", fake_cancel_clip_processing)
    main.app.dependency_overrides[get_db] = _fake_db
    original_lifespan = main.app.router.lifespan_context
    main.app.router.lifespan_context = _noop_lifespan

    try:
        with TestClient(main.app) as client:
            response = client.delete("/projects/11/clips/abc-123")
    finally:
        main.app.router.lifespan_context = original_lifespan

    main.app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()["task_cancelled"] is True
    assert response.json()["deleted_clip_id"] == 99


class _FakeExecResult:
    def __init__(self, row):
        self._row = row

    def scalar_one_or_none(self):
        return self._row


class _FakeSessionSemantic404:
    async def execute(self, *args, **kwargs):
        return _FakeExecResult(None)


async def _fake_db_semantic_404() -> AsyncIterator[object]:
    yield _FakeSessionSemantic404()


def test_semantic_search_project_not_found(monkeypatch):
    main.app.dependency_overrides[get_db] = _fake_db_semantic_404
    original_lifespan = main.app.router.lifespan_context
    main.app.router.lifespan_context = _noop_lifespan
    try:
        with TestClient(main.app) as client:
            response = client.post("/projects/40404/semantic-search", json={"query": "crash"})
    finally:
        main.app.router.lifespan_context = original_lifespan
        main.app.dependency_overrides.clear()
    assert response.status_code == 404


class _FakeSessionSemantic200:
    async def execute(self, *args, **kwargs):
        return _FakeExecResult(object())


async def _fake_db_semantic_200() -> AsyncIterator[object]:
    yield _FakeSessionSemantic200()


def test_semantic_search_project_route(monkeypatch):
    async def fake_search(db, *, project_id, query, limit=None):
        assert project_id == 11
        assert query == "car"
        assert limit == 3
        return {
            "matches": [
                {
                    "clip_id": 99,
                    "local_key": "abc",
                    "file_name": "x.m4a",
                    "start_time_seconds": 0.0,
                    "end_time_seconds": 4.0,
                    "confidence": 0.9,
                    "source": "audio",
                }
            ],
            "query": query,
        }

    monkeypatch.setattr(main, "search_project_semantic", fake_search)
    main.app.dependency_overrides[get_db] = _fake_db_semantic_200
    original_lifespan = main.app.router.lifespan_context
    main.app.router.lifespan_context = _noop_lifespan
    try:
        with TestClient(main.app) as client:
            response = client.post(
                "/projects/11/semantic-search",
                json={"query": "car", "limit": 3},
            )
    finally:
        main.app.router.lifespan_context = original_lifespan
        main.app.dependency_overrides.clear()
    assert response.status_code == 200
    body = response.json()
    assert body["matches"][0]["file_name"] == "x.m4a"
    assert body["matches"][0]["local_key"] == "abc"
