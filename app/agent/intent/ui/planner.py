from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import re
from time import perf_counter
from typing import Any

from langchain_openai import ChatOpenAI

from app.agent.intent.editing.capabilities import DEFAULT_EFFECT_CAPABILITIES
from app.agent.intent.editing.models import IntentCompileResult
from app.agent.intent.ui.models import (
    IntentUIEditorContext,
    IntentUIPlan,
    IntentUIPlanRequest,
    LLMIntentUIPlanDraft,
)
from app.config import settings

logger = logging.getLogger(__name__)

CATALOG_VERSION = "1"

WIDGET_CATALOG: dict[str, dict[str, Any]] = {
    "playback.viewer": {"variants": ["compact", "large"], "contributions": ["preview", "scrub"]},
    "playback.beforeAfterViewer": {"variants": ["large", "split"], "contributions": ["compare_before_after", "preview"]},
    "timeline.full": {"variants": ["expanded", "compressed"], "contributions": ["navigate_timeline", "edit_structure"]},
    "timeline.primaryTrack": {"variants": ["affectedRangeOnly", "full"], "contributions": ["focus_clip", "trim_range"]},
    "timeline.focusedClipStrip": {"variants": ["compact", "expanded"], "contributions": ["focus_clip", "scrub"]},
    "audio.levelsMeter": {"variants": ["compact", "expanded"], "contributions": ["monitor_audio"]},
    "toolbar.parameterControls": {"variants": ["sliderGroup", "compact"], "contributions": ["tune_parameter"]},
    "toolbar.reviewActions": {"variants": ["standard"], "contributions": ["approve_edit", "reject_edit", "refine_prompt"]},
    "toolbar.promptBar": {"variants": ["idle", "typing", "recording"], "contributions": ["enter_prompt", "refine_prompt"]},
    "toolbar.clipTools": {"variants": ["collapsed", "expanded"], "contributions": ["split_clip", "delete_clip", "color_clip", "volume_clip"]},
    "panel.importBrowser": {"variants": ["standard"], "contributions": ["browse_media", "import_media"]},
    "panel.exportSettings": {"variants": ["standard"], "contributions": ["configure_export"]},
}


@dataclass(frozen=True)
class ParameterSpec:
    id: str
    label: str
    min_value: float
    max_value: float
    default_value: float


def _elapsed_ms(started_at: float) -> int:
    return round((perf_counter() - started_at) * 1000)


def _build_parameter_catalog() -> dict[str, ParameterSpec]:
    parameters: dict[str, ParameterSpec] = {}
    operation_labels = {
        "addGrain": ("grain", "Grain"),
        "setTemperature": ("temperature", "Temperature"),
        "setSaturation": ("saturation", "Saturation"),
        "setContrast": ("contrast", "Contrast"),
        "setExposure": ("exposure", "Exposure"),
        "setHighlights": ("highlights", "Highlights"),
        "setShadows": ("shadows", "Shadows"),
    }
    for capability in DEFAULT_EFFECT_CAPABILITIES:
        mapping = operation_labels.get(capability.operation)
        if mapping is None or not capability.parameters:
            continue
        param_id, label = mapping
        param_spec = capability.parameters[0]
        minimum = float(param_spec.minimum if param_spec.minimum is not None else -1)
        maximum = float(param_spec.maximum if param_spec.maximum is not None else 1)
        parameters[param_id] = ParameterSpec(param_id, label, minimum, maximum, 0.0)
    parameters["vintageIntensity"] = ParameterSpec("vintageIntensity", "Vintage", 0.0, 1.0, 0.7)
    parameters["volumeGain"] = ParameterSpec("volumeGain", "Volume", 0.0, 2.0, 1.0)
    return parameters


PARAMETER_CATALOG = _build_parameter_catalog()


def _parameter_control(parameter_id: str) -> dict[str, Any]:
    spec = PARAMETER_CATALOG.get(parameter_id)
    if spec is None:
        return {"parameterId": parameter_id, "control": "slider"}
    return {
        "parameterId": spec.id,
        "control": "slider",
        "label": spec.label,
        "minValue": spec.min_value,
        "maxValue": spec.max_value,
        "defaultValue": spec.default_value,
    }


def _widget(
    widget_id: str,
    *,
    intent_slice_id: str,
    variant: str | None = None,
    prominence: str = "supporting",
    reason: str | None = None,
    size: dict[str, Any] | None = None,
    controls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "widgetId": widget_id,
        "variant": variant,
        "prominence": prominence,
        "intentSliceId": intent_slice_id,
        "reason": reason,
        "size": size,
        "controls": controls or [],
        "props": {},
    }


def _layout_node(
    node_type: str,
    *,
    widget: dict[str, Any] | None = None,
    children: list[dict[str, Any]] | None = None,
    size: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {"type": node_type, "children": children or [], "widget": widget, "size": size}


def _toolbar_from_parameters(parameter_ids: list[str], *, intent_slice_id: str) -> dict[str, Any]:
    return {
        "widgets": [
            _widget(
                "toolbar.parameterControls",
                variant="sliderGroup",
                prominence="primary",
                intent_slice_id=intent_slice_id,
                reason="Expose the parameters most relevant to this intent",
                controls=[_parameter_control(parameter_id) for parameter_id in parameter_ids],
            )
        ],
        "showNavigation": False,
        "showPromptBar": False,
    }


def _prompt_mentions_audio(prompt: str) -> bool:
    return bool(re.search(r"\b(audio|volume|loud|louder|quieter|sound|gain|levels?)\b", prompt.lower()))


def _prompt_mentions_visual_style(prompt: str) -> bool:
    return bool(
        re.search(
            r"\b(vintage|color|colour|warm|cool|grain|fade|saturation|contrast|exposure|look|style|mood|cinematic)\b",
            prompt.lower(),
        )
    )


def _effect_parameter_ids(result: IntentCompileResult | None) -> list[str]:
    if result is None:
        return []
    mapping = {
        "addGrain": "grain",
        "setTemperature": "temperature",
        "setSaturation": "saturation",
        "setContrast": "contrast",
        "setExposure": "exposure",
        "setHighlights": "highlights",
        "setShadows": "shadows",
    }
    ids: list[str] = []
    for operation in result.experimentalEffectOperations:
        param_id = mapping.get(operation.operation)
        if param_id and param_id not in ids:
            ids.append(param_id)
    if ids and "vintageIntensity" not in ids:
        ids.insert(0, "vintageIntensity")
    return ids


def _intent_slices_from_request(request: IntentUIPlanRequest) -> list[dict[str, Any]]:
    prompt = request.prompt.strip()
    slices: list[dict[str, Any]] = []
    visual_params = _effect_parameter_ids(request.intentResult)
    if _prompt_mentions_visual_style(prompt) or visual_params:
        slices.append(
            {
                "id": "visual_style",
                "title": "Visual style",
                "goal": "Preview and tune visual appearance",
                "modality": "visual",
                "parameterIds": visual_params or ["vintageIntensity", "temperature", "grain", "saturation"],
            }
        )
    if _prompt_mentions_audio(prompt):
        slices.append(
            {
                "id": "audio_loudness",
                "title": "Audio loudness",
                "goal": "Adjust loudness and monitor levels",
                "modality": "audio",
                "parameterIds": ["volumeGain"],
            }
        )
    if not slices:
        slices.append({"id": "general_edit", "title": "Edit", "goal": "Review and refine the proposed edit", "modality": "mixed", "parameterIds": []})
    return slices


class DeterministicIntentUIPlanner:
    def plan(self, request: IntentUIPlanRequest) -> IntentUIPlan:
        editor = request.editorContext or IntentUIEditorContext()
        if not request.prompt.strip() and not request.intentResult:
            return self._default_workspace(editor)

        slices = _intent_slices_from_request(request)
        current_index = 0
        if request.currentWorkspaceId:
            for index, slice_ in enumerate(slices):
                if request.currentWorkspaceId == slice_["id"]:
                    current_index = index
                    break
        return self._workspace_for_slice(request, slices, slices[current_index], current_index)

    def _default_workspace(self, editor: IntentUIEditorContext) -> IntentUIPlan:
        toolbar_widgets = [
            _widget(
                "toolbar.promptBar",
                variant="idle",
                prominence="primary",
                intent_slice_id="default",
                reason="Default prompt entry for general editing",
            )
        ]
        if editor.hasSelectedClip:
            toolbar_widgets.append(
                _widget(
                    "toolbar.clipTools",
                    variant="collapsed",
                    intent_slice_id="default",
                    reason="Basic clip operations when a clip is selected",
                )
            )

        layout_children = [
            _layout_node(
                "widget",
                widget=_widget(
                    "playback.viewer",
                    variant="compact",
                    prominence="primary",
                    intent_slice_id="default",
                    reason="Continuous visual feedback during editing",
                    size={"weight": 0.35, "importance": "primary"},
                ),
            ),
            _layout_node(
                "widget",
                widget=_widget(
                    "timeline.full",
                    variant="expanded" if editor.activeSpace == "Edit" else "compressed",
                    prominence="primary",
                    intent_slice_id="default",
                    reason="Timeline representation for navigation and direct edits",
                    size={"weight": 0.45, "importance": "primary"},
                ),
            ),
        ]
        hidden = []
        if editor.activeSpace == "Import":
            layout_children.append(
                _layout_node(
                    "widget",
                    widget=_widget("panel.importBrowser", variant="standard", intent_slice_id="default", reason="Import flow requires media browsing", size={"weight": 0.2}),
                )
            )
        else:
            hidden.append("panel.importBrowser")
        if editor.activeSpace == "Export":
            layout_children.append(
                _layout_node(
                    "widget",
                    widget=_widget("panel.exportSettings", variant="standard", intent_slice_id="default", reason="Export flow requires output settings", size={"weight": 0.2}),
                )
            )
        else:
            hidden.append("panel.exportSettings")

        return IntentUIPlan(
            catalogVersion=CATALOG_VERSION,
            workspaceId="default",
            intentSummary="Default editor workspace",
            intentSlices=[{"id": "default", "title": "Edit", "goal": "General timeline editing", "modality": "mixed", "parameterIds": []}],
            currentSliceId="default",
            currentSliceIndex=0,
            layout=_layout_node("vstack", children=layout_children),
            toolbar={"widgets": toolbar_widgets, "showNavigation": True, "showPromptBar": True},
            isDefaultWorkspace=True,
            restoreDefaultOnComplete=False,
            hiddenBecauseIrrelevant=hidden,
        )

    def _workspace_for_slice(
        self,
        request: IntentUIPlanRequest,
        slices: list[dict[str, Any]],
        current_slice: dict[str, Any],
        current_index: int,
    ) -> IntentUIPlan:
        current_slice_id = str(current_slice["id"])
        modality = current_slice.get("modality")
        hidden = ["panel.importBrowser", "panel.exportSettings", "toolbar.promptBar"]
        layout_children: list[dict[str, Any]] = []
        toolbar_widgets = [
            _widget(
                "toolbar.reviewActions",
                variant="standard",
                intent_slice_id=current_slice_id,
                reason="Allow apply, cancel, or refine during intent review",
            )
        ]

        if modality == "visual":
            layout_children.extend(
                [
                    _layout_node(
                        "widget",
                        widget=_widget(
                            "playback.beforeAfterViewer",
                            variant="large",
                            prominence="primary",
                            intent_slice_id=current_slice_id,
                            reason="Visual intent requires before/after comparison",
                            size={"weight": 0.58, "minHeight": 220, "importance": "primary"},
                        ),
                    ),
                    _layout_node(
                        "hstack",
                        size={"weight": 0.27},
                        children=[
                            _layout_node(
                                "widget",
                                widget=_widget(
                                    "timeline.focusedClipStrip",
                                    variant="affectedRangeOnly",
                                    intent_slice_id=current_slice_id,
                                    reason="Show only the affected clip region",
                                    size={"weight": 0.45},
                                ),
                            )
                        ],
                    ),
                ]
            )
            parameter_ids = list(current_slice.get("parameterIds") or [])
            if parameter_ids:
                toolbar_widgets.insert(0, _toolbar_from_parameters(parameter_ids, intent_slice_id=current_slice_id)["widgets"][0])
        elif modality == "audio":
            layout_children.extend(
                [
                    _layout_node(
                        "widget",
                        widget=_widget(
                            "playback.viewer",
                            variant="large",
                            prominence="primary",
                            intent_slice_id=current_slice_id,
                            reason="Audio changes still need playback feedback",
                            size={"weight": 0.5, "importance": "primary"},
                        ),
                    ),
                    _layout_node(
                        "widget",
                        widget=_widget(
                            "audio.levelsMeter",
                            variant="expanded",
                            intent_slice_id=current_slice_id,
                            reason="Levels help validate loudness changes",
                            size={"weight": 0.2},
                        ),
                    ),
                ]
            )
            toolbar_widgets.insert(
                0,
                _toolbar_from_parameters(list(current_slice.get("parameterIds") or ["volumeGain"]), intent_slice_id=current_slice_id)["widgets"][0],
            )
            hidden.append("timeline.full")
        else:
            layout_children.extend(
                [
                    _layout_node(
                        "widget",
                        widget=_widget(
                            "playback.viewer",
                            variant="large",
                            prominence="primary",
                            intent_slice_id=current_slice_id,
                            reason="Preview proposed edit",
                            size={"weight": 0.55},
                        ),
                    ),
                    _layout_node(
                        "widget",
                        widget=_widget(
                            "timeline.primaryTrack",
                            variant="affectedRangeOnly",
                            intent_slice_id=current_slice_id,
                            reason="Focused timeline for the affected clip",
                            size={"weight": 0.25},
                        ),
                    ),
                ]
            )

        return IntentUIPlan(
            catalogVersion=CATALOG_VERSION,
            workspaceId=current_slice_id,
            intentSummary=request.prompt.strip() or "Intent workspace",
            intentSlices=slices,
            currentSliceId=current_slice_id,
            currentSliceIndex=current_index,
            layout=_layout_node("vstack", children=layout_children),
            toolbar={"widgets": toolbar_widgets, "showNavigation": len(slices) > 1, "showPromptBar": False},
            hiddenBecauseIrrelevant=hidden,
            isDefaultWorkspace=False,
            restoreDefaultOnComplete=True,
        )


class LLMIntentUIPlanner:
    def __init__(self, llm: Any | None = None) -> None:
        self.llm = llm or ChatOpenAI(api_key=settings.openai_api_key, model=settings.intent_ui_openai_model, temperature=0)

    async def plan(self, request: IntentUIPlanRequest) -> IntentUIPlan:
        llm = self.llm.with_structured_output(LLMIntentUIPlanDraft, method="function_calling")
        started_at = perf_counter()
        draft = await llm.ainvoke(
            [
                (
                    "system",
                    "You plan focused just-in-time editor UI workspaces. Return compact JSON using only known widget ids, variants, and parameters.",
                ),
                ("human", self._prompt(request)),
            ]
        )
        logger.info("[intent-ui] LLM planner completed slices=%s elapsed_ms=%s", len(draft.intentSlices), _elapsed_ms(started_at))
        if not draft.intentSlices:
            return DeterministicIntentUIPlanner().plan(request)

        current_slice_id = draft.currentSliceId or str(draft.intentSlices[0].get("id") or "workspace")
        current_index = next((index for index, slice_ in enumerate(draft.intentSlices) if slice_.get("id") == current_slice_id), 0)
        return IntentUIPlan(
            catalogVersion=CATALOG_VERSION,
            workspaceId=current_slice_id,
            intentSummary=draft.intentSummary,
            intentSlices=draft.intentSlices,
            currentSliceId=current_slice_id,
            currentSliceIndex=current_index,
            layout=draft.layout,
            toolbar=draft.toolbar,
            hiddenBecauseIrrelevant=draft.hiddenBecauseIrrelevant,
            transitions=draft.transitions,
            isDefaultWorkspace=False,
            restoreDefaultOnComplete=True,
        )

    def _prompt(self, request: IntentUIPlanRequest) -> str:
        catalog = {
            "widgets": [
                {"id": widget_id, "variants": spec["variants"], "contributions": spec["contributions"]}
                for widget_id, spec in sorted(WIDGET_CATALOG.items())
            ],
            "parameters": [
                {"id": spec.id, "label": spec.label, "control": "slider", "min": spec.min_value, "max": spec.max_value}
                for spec in PARAMETER_CATALOG.values()
            ],
        }
        payload = {
            "prompt": request.prompt,
            "context": request.context.model_dump(),
            "editorContext": request.editorContext.model_dump() if request.editorContext else {},
            "intentResult": request.intentResult.model_dump() if request.intentResult else None,
            "catalog": catalog,
        }
        return json.dumps(payload, default=str)


class IntentUIPlannerService:
    def __init__(
        self,
        *,
        deterministic: DeterministicIntentUIPlanner | None = None,
        llm_planner: LLMIntentUIPlanner | None = None,
        use_llm: bool | None = None,
    ) -> None:
        self.deterministic = deterministic or DeterministicIntentUIPlanner()
        self.llm_planner = llm_planner
        self.use_llm = use_llm if use_llm is not None else bool(settings.openai_api_key)

    async def plan(self, request: IntentUIPlanRequest) -> IntentUIPlan:
        if not request.prompt.strip() and not request.intentResult:
            return self.deterministic.plan(request)
        if not self.use_llm:
            return self.deterministic.plan(request)
        try:
            planner = self.llm_planner or LLMIntentUIPlanner()
            return await planner.plan(request)
        except Exception:
            logger.exception("[intent-ui] LLM planner failed; using deterministic fallback")
            return self.deterministic.plan(request)
