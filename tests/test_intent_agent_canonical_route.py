from unittest.mock import AsyncMock

from app.api.routes import agent as agent_routes
from app.agent.intent.editing.models import CompileSource, IntentCompileResult, IntentCompilerContext
from app.agent.intent.models import IntentAgentResponse
from app.agent.intent.ui.models import IntentUIPlanRequest
from app.agent.intent.ui.planner import DeterministicIntentUIPlanner


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


def test_agent_intent_route_covers_visual_workspace_planning(monkeypatch, app_client):
    context = _sample_context()
    owned_project = AsyncMock(return_value=None)
    owned_session = AsyncMock(return_value=None)

    async def fake_prepare_intent_transcript_context(*, prompt, context, db):
        assert prompt == "apply a vintage effect"
        return context, {"hydrated_clip_ids": ["clip-b"]}

    async def fake_run_intent_agent(request, *, context, hydration_meta=None):
        ui_plan = DeterministicIntentUIPlanner().plan(
            IntentUIPlanRequest(prompt=request.prompt, context=context)
        )
        return IntentAgentResponse(edit=_edit_result(), ui=ui_plan)

    monkeypatch.setattr(agent_routes, "require_owned_project", owned_project)
    monkeypatch.setattr(agent_routes, "require_owned_session", owned_session)
    monkeypatch.setattr(
        agent_routes,
        "prepare_intent_transcript_context",
        fake_prepare_intent_transcript_context,
    )
    monkeypatch.setattr(agent_routes, "run_intent_agent", fake_run_intent_agent)
    response = app_client.post(
        "/agent/runs",
        json={
            "kind": "intent",
            "prompt": "apply a vintage effect",
            "context": context.model_dump(mode="json", by_alias=True),
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ui"]["workspaceId"] == "visual_style"
    assert body["ui"]["currentSliceId"] == "visual_style"
    owned_project.assert_awaited_once()
    owned_session.assert_awaited_once()
