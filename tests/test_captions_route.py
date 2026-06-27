from collections.abc import AsyncIterator

from app.api.routes import sources as sources_routes


async def _fake_db() -> AsyncIterator[object]:
    yield object()


def test_get_captions_ok(monkeypatch, app_client, set_db_override):
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

    monkeypatch.setattr(sources_routes, "get_clip_captions_payload", fake_get_clip_captions_payload)
    set_db_override(_fake_db)
    response = app_client.get("/projects/1/sources/lk-1/transcript")

    assert response.status_code == 200
    data = response.json()
    assert data["clip_id"] == 99
    assert data["sentences"][0]["text"] == "Hello."


def test_get_captions_404_project(monkeypatch, app_client, set_db_override):
    async def fake_get_clip_captions_payload(_db, *, project_id, local_key):
        return "project_not_found", None

    monkeypatch.setattr(sources_routes, "get_clip_captions_payload", fake_get_clip_captions_payload)
    set_db_override(_fake_db)
    response = app_client.get("/projects/999/sources/x/transcript")

    assert response.status_code == 404


def test_get_captions_409_no_transcript(monkeypatch, app_client, set_db_override):
    async def fake_get_clip_captions_payload(_db, *, project_id, local_key):
        return "transcript_not_ready", {"clip_id": 1, "processing_status": "processing"}

    monkeypatch.setattr(sources_routes, "get_clip_captions_payload", fake_get_clip_captions_payload)
    set_db_override(_fake_db)
    response = app_client.get("/projects/1/sources/lk/transcript")

    assert response.status_code == 409
