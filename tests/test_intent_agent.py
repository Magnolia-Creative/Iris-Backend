import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

from app.api.routes import agent as agent_routes
from app.agent.intent.editing.models import CompileSource, IntentCompileResult, IntentCompilerContext
from app.agent.intent.models import IntentAgentRequest, IntentAgentResponse
from app.agent.intent.service import IntentAgentService
from app.agent.intent.ui.models import IntentUIPlanRequest
from app.agent.intent.ui.planner import DeterministicIntentUIPlanner


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sample_context(**overrides) -> IntentCompilerContext:
    payload = {
        "timelineId": "timeline-test",
        "projectId": 1,
        "sessionId": 10,
        "selectedClipId": "clip-b",
        "selectedTrackId": "track-video",
        "playheadTimeUs": 10_000_000,
        "clipsById": {
            "clip-b": {
                "clip_id": "clip-b",
                "track_id": "track-video",
                "media_id": "media-b",
                "source_range": {"start": 0, "end": 10_000_000},
                "timeline_range": {"start": 5_000_000, "end": 15_000_000},
            }
        },
        "orderedClipIdsByTrackId": {"track-video": ["clip-b"]},
    }
    payload.update(overrides)
    return IntentCompilerContext.model_validate(payload)


def _edit_result() -> IntentCompileResult:
    return IntentCompileResult(
        actions=[],
        confidence=0.8,
        source=CompileSource.llm,
        unresolvedText=None,
        warnings=[],
        needsClarification=False,
        experimentalEffectOperations=[],
    )


def _ui_plan(prompt: str, context: IntentCompilerContext):
    return DeterministicIntentUIPlanner().plan(
        IntentUIPlanRequest(prompt=prompt, context=context)
    )


class _FakeIntentCompilerService:
    def __init__(self, *, started: asyncio.Event | None = None, peer_started: asyncio.Event | None = None):
        self.started = started
        self.peer_started = peer_started

    async def compile_prompt(self, *, prompt, context, event_handler=None):
        if self.started:
            self.started.set()
        if self.peer_started:
            await self.peer_started.wait()
        assert prompt == "make the audio louder"
        if event_handler:
            await event_handler({"type": "planner_started", "status": "Parsing prompt."})
        return _edit_result()


class _FakeIntentUIPlannerService:
    def __init__(self, *, started: asyncio.Event | None = None, peer_started: asyncio.Event | None = None):
        self.started = started
        self.peer_started = peer_started
        self.requests = []

    async def plan(self, request):
        if self.started:
            self.started.set()
        if self.peer_started:
            await self.peer_started.wait()
        self.requests.append(request)
        return _ui_plan(request.prompt, request.context)


def test_intent_agent_service_returns_edit_and_ui_outputs():
    context = _sample_context()
    ui_planner = _FakeIntentUIPlannerService()
    service = IntentAgentService(
        intent_compiler_service=_FakeIntentCompilerService(),
        ui_intent_planner_service=ui_planner,
    )

    result = asyncio.run(
        service.run(
            IntentAgentRequest(prompt="make the audio louder", context=context),
            context=context,
            hydration_meta={"hydrated_clip_ids": ["clip-b"]},
        )
    )

    assert result.edit.actions == []
    assert result.ui.workspaceId == "audio_loudness"
    assert result.meta.hydration == {"hydrated_clip_ids": ["clip-b"]}
    assert result.meta.edit_events[0]["type"] == "planner_started"
    assert ui_planner.requests[0].context == context


def test_intent_agent_graph_schedules_edit_and_ui_branches_concurrently():
    context = _sample_context()
    edit_started = asyncio.Event()
    ui_started = asyncio.Event()
    service = IntentAgentService(
        intent_compiler_service=_FakeIntentCompilerService(
            started=edit_started,
            peer_started=ui_started,
        ),
        ui_intent_planner_service=_FakeIntentUIPlannerService(
            started=ui_started,
            peer_started=edit_started,
        ),
    )

    result = asyncio.run(
        asyncio.wait_for(
            service.run(IntentAgentRequest(prompt="make the audio louder", context=context)),
            timeout=1,
        )
    )

    assert result.edit.source == CompileSource.llm
    assert result.ui.workspaceId == "audio_loudness"


def test_intent_agent_route_returns_combined_response(monkeypatch, app_client):
    context = _sample_context()
    owned_project = AsyncMock(return_value=None)
    owned_session = AsyncMock(return_value=None)

    async def fake_prepare_intent_transcript_context(*, prompt, context, db):
        assert prompt == "make the audio louder"
        return context, {"hydrated_clip_ids": ["clip-b"]}

    async def fake_run_intent_agent(request, *, context, hydration_meta=None):
        return IntentAgentResponse(
            edit=_edit_result(),
            ui=_ui_plan(request.prompt, context),
        )

    monkeypatch.setattr(agent_routes, "require_owned_project", owned_project)
    monkeypatch.setattr(agent_routes, "require_owned_session", owned_session)
    monkeypatch.setattr(agent_routes, "prepare_intent_transcript_context", fake_prepare_intent_transcript_context)
    monkeypatch.setattr(agent_routes, "run_intent_agent", fake_run_intent_agent)
    response = app_client.post(
        "/agent/runs",
        json={
            "kind": "intent",
            "prompt": "make the audio louder",
            "context": context.model_dump(mode="json", by_alias=True),
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["edit"]["actions"] == []
    assert body["ui"]["workspaceId"] == "audio_loudness"
    owned_project.assert_awaited_once()
    owned_session.assert_awaited_once()


def test_intent_agent_branch_modules_own_their_models():
    editing_dir = PROJECT_ROOT / "app" / "agent" / "intent" / "editing"
    editing_sources = "\n".join(
        path.read_text()
        for path in editing_dir.glob("*.py")
        if path.name not in {"__init__.py"}
    )
    assert "app.intent_compiler.models" not in editing_sources
    assert "app.intent_compiler.llm" not in editing_sources

    ui_models = (PROJECT_ROOT / "app" / "agent" / "intent" / "ui" / "models.py").read_text()
    assert "app.ui_workspace" not in ui_models
    assert "IntentUIWidgetPlacement" not in ui_models
