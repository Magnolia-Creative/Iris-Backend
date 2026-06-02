from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

import main
from app.intent_compiler.models import (
    ExperimentalEffectOperation,
    IntentCompileResult,
    IntentCompilerContext,
)
from app.ui_workspace.catalog import WIDGET_CATALOG, catalog_parameter_ids
from app.ui_workspace.models import (
    IntentSlice,
    LayoutNode,
    LayoutNodeType,
    ToolbarPlacement,
    UIEditorContext,
    UIWorkspacePlan,
    UIWorkspacePlanRequest,
    WidgetPlacement,
)
from app.ui_workspace.planner import DeterministicUIWorkspacePlanner, UIWorkspacePlannerService
from app.ui_workspace.validator import UIWorkspaceValidationError, validate_workspace_plan


@asynccontextmanager
async def _noop_lifespan(_app):
    yield


def _sample_context(**overrides) -> IntentCompilerContext:
    payload = {
        "timelineId": "timeline-test",
        "projectId": 1,
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


def test_catalog_contains_core_widgets():
    assert "playback.viewer" in WIDGET_CATALOG
    assert "toolbar.parameterControls" in WIDGET_CATALOG
    assert "vintageIntensity" in catalog_parameter_ids()


def test_default_workspace_includes_timeline_and_prompt():
    planner = DeterministicUIWorkspacePlanner()
    plan = planner.plan(
        UIWorkspacePlanRequest(
            prompt="",
            context=_sample_context(),
            editorContext=UIEditorContext(activeSpace="Edit", hasSelectedClip=False),
        )
    )
    assert plan.isDefaultWorkspace is True
    assert plan.toolbar.showPromptBar is True
    assert plan.toolbar.showNavigation is True
    widget_ids = _collect_widget_ids(plan.layout)
    assert "playback.viewer" in widget_ids
    assert "timeline.full" in widget_ids


def test_visual_and_audio_prompt_creates_two_slices():
    planner = DeterministicUIWorkspacePlanner()
    plan = planner.plan(
        UIWorkspacePlanRequest(
            prompt="apply a vintage effect and make the audio louder",
            context=_sample_context(),
            intentResult=IntentCompileResult.model_validate(
                {
                    "actions": [],
                    "confidence": 0.9,
                    "source": "llm",
                    "unresolvedText": None,
                    "warnings": [],
                    "needsClarification": False,
                    "experimentalEffectOperations": [
                        {
                            "operation": "setTemperature",
                            "sourceText": "vintage",
                            "confidence": 0.8,
                            "parameters": {"value": 0.2},
                        }
                    ],
                }
            ),
        )
    )
    assert len(plan.intentSlices) == 2
    assert plan.intentSlices[0].id == "visual_style"
    assert plan.currentSliceId == "visual_style"
    assert "playback.beforeAfterViewer" in _collect_widget_ids(plan.layout)
    assert plan.toolbar.showNavigation is True
    assert plan.toolbar.showPromptBar is False


def test_audio_slice_workspace_uses_levels_meter():
    planner = DeterministicUIWorkspacePlanner()
    plan = planner.plan(
        UIWorkspacePlanRequest(
            prompt="make the audio louder",
            context=_sample_context(),
            currentWorkspaceId="audio_loudness",
        )
    )
    assert plan.currentSliceId == "audio_loudness"
    widget_ids = _collect_widget_ids(plan.layout)
    assert "audio.levelsMeter" in widget_ids
    toolbar_widget_ids = [widget.widgetId for widget in plan.toolbar.widgets]
    assert "toolbar.parameterControls" in toolbar_widget_ids


def test_validator_rejects_unknown_widget():
    plan = UIWorkspacePlan(
        workspaceId="bad",
        intentSummary="test",
        layout=LayoutNode(
            type=LayoutNodeType.widget,
            widget=WidgetPlacement(
                widgetId="unknown.widget",
                intentSliceId="default",
            ),
        ),
        toolbar=ToolbarPlacement(),
    )
    try:
        validate_workspace_plan(plan)
        raised = False
    except UIWorkspaceValidationError:
        raised = True
    assert raised


def test_ui_workspace_plan_accepts_swift_shaped_vintage_intent_result():
    """Regression: iOS re-posts websocket intent results to ui-workspace-plan."""
    intent_result = {
        "actions": [
            {
                "action_id": "8025399e-5821-45ae-824e-1ac3f406ead2",
                "timeline_id": "timeline-test",
                "created_at": "2026-06-01T22:30:04Z",
                "type": "UPDATE_EFFECT_PARAMS",
                "payload": {
                    "updateClipColorFilter": {
                        "clipId": "clip-b",
                        "adjustments": {
                            "saturation": -0.25,
                            "contrast": -0.15,
                            "temperature": 0.18,
                        },
                    }
                },
                "group_id": None,
            }
        ],
        "confidence": 0.85,
        "source": "llm",
        "unresolvedText": None,
        "warnings": ["unsupportedAction"],
        "needsClarification": False,
        "experimentalEffectOperations": [
            {
                "operation": "addGrain",
                "sourceText": "Apply a vintage effect.",
                "intention": "add vintage film grain",
                "target": {"type": "selectedClip", "clipId": None, "value": None, "track": None},
                "confidence": 0.85,
                "parameters": {"amount": 0.45},
                "parameterNotes": {"amount": "Adds visible film grain."},
            },
            {
                "operation": "setSaturation",
                "sourceText": "Apply a vintage effect.",
                "intention": "fade colors slightly",
                "target": {"type": "selectedClip", "clipId": None, "value": None, "track": None},
                "confidence": 0.85,
                "parameters": {"value": -0.25},
                "parameterNotes": {"value": "Desaturates the clip."},
            },
        ],
    }
    request = UIWorkspacePlanRequest(
        prompt="Apply a vintage effect.",
        context=_sample_context(projectId=1),
        intentResult=IntentCompileResult.model_validate(intent_result),
    )
    plan = DeterministicUIWorkspacePlanner().plan(request)
    assert plan.workspaceId == "visual_style"
    assert plan.toolbar.showPromptBar is False
    toolbar_widget_ids = [widget.widgetId for widget in plan.toolbar.widgets]
    assert "toolbar.parameterControls" in toolbar_widget_ids


def test_ui_workspace_plan_route_returns_plan():
    main.app.router.lifespan_context = _noop_lifespan
    with patch("main.require_owned_project", new_callable=AsyncMock) as owned_project:
        owned_project.return_value = object()
        with patch(
            "main.UIWorkspacePlannerService.plan",
            new_callable=AsyncMock,
        ) as plan_mock:
            plan_mock.return_value = DeterministicUIWorkspacePlanner().plan(
                UIWorkspacePlanRequest(
                    prompt="apply vintage look",
                    context=_sample_context(projectId=1),
                )
            )
            with TestClient(main.app) as client:
                with patch(
                    "app.auth.clerk.require_clerk_user",
                    return_value=type("P", (), {"user_id": "user_test"})(),
                ):
                    response = client.post(
                        "/projects/1/ui-workspace-plan",
                        json={
                            "prompt": "apply vintage look",
                            "context": _sample_context(projectId=1).model_dump(),
                        },
                    )
    assert response.status_code == 200
    body = response.json()
    assert body["plan"]["workspaceId"] == "visual_style"


def _collect_widget_ids(node: LayoutNode) -> list[str]:
    ids: list[str] = []
    if node.widget is not None:
        ids.append(node.widget.widgetId)
    for child in node.children:
        ids.extend(_collect_widget_ids(child))
    return ids
