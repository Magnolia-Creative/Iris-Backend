from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi.testclient import TestClient

import main
from app.api.routes import agent as agent_routes
from app.auth import ClerkPrincipal
from app.database import get_db


@asynccontextmanager
async def _noop_lifespan(_app):
    yield


async def _fake_db() -> AsyncIterator[object]:
    yield object()


async def _fake_require_clerk_websocket_user(_websocket):
    return ClerkPrincipal(user_id="user_test", session_id="sess_test", claims={"sub": "user_test"})


def test_agent_run_websocket_delegates_to_automake_graph(monkeypatch):
    async def fake_get_persisted_session_data(db, session_id, *, include_ingest_details=False):
        assert session_id == 7
        assert include_ingest_details is True
        return {
            "session_id": 7,
            "session_name": "Edit run",
            "session_status": "ready",
            "project_id": 11,
            "project_name": "Launch Day",
            "uploaded_count": 1,
            "pending_clip_count": 0,
            "settled_clip_count": 1,
            "ready_for_websocket": True,
            "videos": [],
        }

    async def fake_get_persisted_session_graph_state(*, db, session_id):
        assert session_id == 7
        return None

    async def fake_run_session_until_pause(*, state, db, event_handler):
        assert state["session_id"] == "7"
        assert state["user_prompt"] == "make a reel"
        await event_handler(
            {
                "type": "session_complete",
                "session_id": 7,
                "project_id": 11,
                "timeline": [],
            }
        )

    async def fake_require_owned_session(*args, **kwargs):
        return None

    monkeypatch.setattr(
        agent_routes,
        "require_clerk_websocket_user",
        _fake_require_clerk_websocket_user,
    )
    monkeypatch.setattr(agent_routes, "require_owned_session", fake_require_owned_session)
    monkeypatch.setattr(agent_routes, "get_persisted_session_data", fake_get_persisted_session_data)
    monkeypatch.setattr(
        agent_routes,
        "get_persisted_session_graph_state",
        fake_get_persisted_session_graph_state,
    )
    monkeypatch.setattr(agent_routes, "run_session_until_pause", fake_run_session_until_pause)
    monkeypatch.setattr(agent_routes, "record_session_event", lambda _session_id, payload: payload)
    main.app.dependency_overrides[get_db] = _fake_db
    original_lifespan = main.app.router.lifespan_context
    main.app.router.lifespan_context = _noop_lifespan
    try:
        with TestClient(main.app) as client:
            with client.websocket_connect("/agent/runs/7/stream") as websocket:
                websocket.send_json({"type": "start_session", "user_prompt": "make a reel"})
                assert websocket.receive_json()["type"] == "session_started"
                assert websocket.receive_json()["type"] == "session_complete"
                websocket.send_json({"type": "done"})
                assert websocket.receive_json()["type"] == "session_closed"
    finally:
        main.app.router.lifespan_context = original_lifespan
        main.app.dependency_overrides.pop(get_db, None)
