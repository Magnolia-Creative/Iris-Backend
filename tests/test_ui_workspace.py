from app.agent.intent.editing.models import IntentCompileResult, IntentCompilerContext
from app.agent.intent.ui.models import IntentUIEditorContext, IntentUIPlanRequest
from app.agent.intent.ui.planner import (
    PARAMETER_CATALOG,
    WIDGET_CATALOG,
    DeterministicIntentUIPlanner,
)


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


def _collect_widget_ids(node: dict) -> list[str]:
    ids: list[str] = []
    widget = node.get("widget")
    if isinstance(widget, dict) and widget.get("widgetId"):
        ids.append(widget["widgetId"])
    for child in node.get("children") or []:
        ids.extend(_collect_widget_ids(child))
    return ids


def test_catalog_contains_core_widgets():
    assert "playback.viewer" in WIDGET_CATALOG
    assert "toolbar.parameterControls" in WIDGET_CATALOG
    assert "vintageIntensity" in PARAMETER_CATALOG


def test_default_workspace_includes_timeline_and_prompt():
    planner = DeterministicIntentUIPlanner()
    plan = planner.plan(
        IntentUIPlanRequest(
            prompt="",
            context=_sample_context(),
            editorContext=IntentUIEditorContext(activeSpace="Edit", hasSelectedClip=False),
        )
    )

    assert plan.isDefaultWorkspace is True
    assert plan.toolbar["showPromptBar"] is True
    assert plan.toolbar["showNavigation"] is True
    widget_ids = _collect_widget_ids(plan.layout)
    assert "playback.viewer" in widget_ids
    assert "timeline.full" in widget_ids


def test_visual_and_audio_prompt_creates_two_slices():
    planner = DeterministicIntentUIPlanner()
    plan = planner.plan(
        IntentUIPlanRequest(
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
    assert plan.intentSlices[0]["id"] == "visual_style"
    assert plan.currentSliceId == "visual_style"
    assert "playback.beforeAfterViewer" in _collect_widget_ids(plan.layout)
    assert plan.toolbar["showNavigation"] is True
    assert plan.toolbar["showPromptBar"] is False


def test_audio_slice_workspace_uses_levels_meter():
    planner = DeterministicIntentUIPlanner()
    plan = planner.plan(
        IntentUIPlanRequest(
            prompt="make the audio louder",
            context=_sample_context(),
            currentWorkspaceId="audio_loudness",
        )
    )

    assert plan.currentSliceId == "audio_loudness"
    widget_ids = _collect_widget_ids(plan.layout)
    assert "audio.levelsMeter" in widget_ids
    toolbar_widget_ids = [widget["widgetId"] for widget in plan.toolbar["widgets"]]
    assert "toolbar.parameterControls" in toolbar_widget_ids


def test_swift_shaped_vintage_intent_result_drives_parameter_controls():
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
    request = IntentUIPlanRequest(
        prompt="Apply a vintage effect.",
        context=_sample_context(projectId=1),
        intentResult=IntentCompileResult.model_validate(intent_result),
    )
    plan = DeterministicIntentUIPlanner().plan(request)

    assert plan.workspaceId == "visual_style"
    assert plan.toolbar["showPromptBar"] is False
    toolbar_widget_ids = [widget["widgetId"] for widget in plan.toolbar["widgets"]]
    assert "toolbar.parameterControls" in toolbar_widget_ids
