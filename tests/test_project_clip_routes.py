from collections.abc import AsyncIterator

from fastapi.testclient import TestClient

import main
from app.database import get_db


async def _fake_db() -> AsyncIterator[object]:
    yield object()


def test_create_sentence_transcription_route(monkeypatch):
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

    monkeypatch.setattr(main, "transcribe_upload_to_sentences", fake_transcribe_upload_to_sentences)

    with TestClient(main.app) as client:
        response = client.post(
            "/transcriptions/sentences",
            files={"audio": ("clip.m4a", b"audio", "audio/mp4")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["transcript_id"] == "tr-123"
    assert payload["full_text"] == "Hello world."
    assert payload["sentences"][0]["text"] == "Hello world."
    assert payload["meta"]["provider"] == "modal"


def test_create_project_agent_session_route(monkeypatch):
    async def fake_create_agent_session(db, *, session_name=None, project_name=None):
        assert session_name == "Launch Day"
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

    with TestClient(main.app) as client:
        response = client.post("/projects/agent-sessions", json={"project_name": "Launch Day"})

    main.app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()["project_id"] == 11
    assert response.json()["session_name"] == "Launch Day"


def test_process_project_clip_batch_route(monkeypatch):
    async def fake_process_project_clips(db, *, project_id, session_id, videos, local_keys):
        assert project_id == 11
        assert session_id == 7
        assert local_keys == ["abc-123"]
        assert len(videos) == 1
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

    monkeypatch.setattr(main, "process_project_clips", fake_process_project_clips)
    main.app.dependency_overrides[get_db] = _fake_db

    with TestClient(main.app) as client:
        response = client.post(
            "/projects/11/clips/process",
            data={"session_id": "7", "local_key": "abc-123"},
            files={"videos": ("clip.m4a", b"audio", "audio/mp4")},
        )

    main.app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()["ready_for_websocket"] is True
    assert response.json()["videos"][0]["local_key"] == "abc-123"
    assert "file_size_bytes" not in response.json()["videos"][0]
    assert "transcript_segments" not in response.json()["videos"][0]
    assert "transcript_full_text" not in response.json()["videos"][0]
    assert "video_report" not in response.json()["videos"][0]
    assert "clip_meta" not in response.json()["videos"][0]


def test_cancel_project_clip_route(monkeypatch):
    async def fake_cancel_clip_processing(db, *, project_id, session_id, local_key):
        assert project_id == 11
        assert session_id == 7
        assert local_key == "abc-123"
        return {
            "session_id": 7,
            "project_id": 11,
            "local_key": "abc-123",
            "task_cancelled": True,
            "deleted_clip_id": 99,
            "session_status": "created",
            "ready_for_websocket": False,
        }

    monkeypatch.setattr(main, "cancel_clip_processing", fake_cancel_clip_processing)
    main.app.dependency_overrides[get_db] = _fake_db

    with TestClient(main.app) as client:
        response = client.delete("/projects/11/clips/abc-123?session_id=7")

    main.app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()["task_cancelled"] is True
    assert response.json()["deleted_clip_id"] == 99
