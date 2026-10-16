import asyncio
from io import BytesIO

from fastapi import UploadFile

from app.database import models
from app.services import session_ingest


def test_ready_clip_without_embeddings_is_reindexed_without_retranscription(monkeypatch):
    project = models.Project(id=4, name="Existing project")
    clip = models.Clip(
        id=5, project_id=4, local_key="clip-key", processing_status="ready"
    )

    class FakeDb:
        async def flush(self):
            pass

        async def commit(self):
            pass

        async def rollback(self):
            pass

        def add(self, _object):
            raise AssertionError("A ready clip must not be recreated")

    async def project_lookup(_db, *, project_id):
        assert project_id == 4
        return project

    async def clip_lookup(_db, *, project_id, local_key):
        assert (project_id, local_key) == (4, "clip-key")
        return clip

    async def count(_db, *, clip_id):
        assert clip_id == 5
        return 0

    async def record_transcription(*, project_id, pending_rows):
        assert project_id == 4
        assert pending_rows == []

    async def project_payload(_db, *, project_id):
        return {"project_id": project_id, "uploaded_count": 1}

    indexed = []

    def record_index(**kwargs):
        indexed.extend(kwargs["pending_rows"])
        return len(kwargs["pending_rows"])

    monkeypatch.setattr(session_ingest, "_require_project", project_lookup)
    monkeypatch.setattr(session_ingest, "_get_clip_by_project_and_local_key", clip_lookup)
    monkeypatch.setattr(session_ingest, "count_embeddings_for_clip", count)
    monkeypatch.setattr(session_ingest, "schedule_transcript_processing_tasks", record_transcription)
    monkeypatch.setattr(session_ingest, "schedule_vector_index_tasks", record_index)
    monkeypatch.setattr(session_ingest, "_build_project_payload", project_payload)
    monkeypatch.setattr(session_ingest.settings, "semantic_indexing_enabled", True)
    monkeypatch.setattr(session_ingest.gemini_embedding, "gemini_configured", lambda: True)

    result = asyncio.run(
        session_ingest.process_project_clips(
            FakeDb(),
            project_id=4,
            session_id=None,
            videos=[UploadFile(file=BytesIO(b"audio"), filename="clip.m4a")],
            local_keys=["clip-key"],
        )
    )

    assert [indexed_clip.id for _, indexed_clip in indexed] == [5]
    assert result["vector_index"] == {"status": "scheduled", "scheduled_clip_count": 1}
