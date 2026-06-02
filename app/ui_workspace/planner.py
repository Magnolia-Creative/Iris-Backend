from __future__ import annotations

import json
import logging
import re
from time import perf_counter
from typing import Any

from langchain_openai import ChatOpenAI

from app.config import settings
from app.intent_compiler.models import IntentCompileResult, IntentCompilerContext
from app.ui_workspace.catalog import CATALOG_VERSION, PARAMETER_CATALOG, WIDGET_CATALOG
from app.ui_workspace.models import (
    IntentSlice,
    LayoutNode,
    LayoutNodeType,
    LLMWorkspaceDraft,
    Prominence,
    ToolbarPlacement,
    UIEditorContext,
    UIWorkspacePlan,
    UIWorkspacePlanRequest,
    WidgetPlacement,
    WidgetSizeHint,
)
from app.ui_workspace.validator import (
    parameter_control_from_catalog,
    toolbar_from_parameters,
    validate_workspace_plan,
)

logger = logging.getLogger(__name__)


def _elapsed_ms(started_at: float) -> int:
    return round((perf_counter() - started_at) * 1000)


def _get_llm() -> Any:
    return ChatOpenAI(
        api_key=settings.openai_api_key,
        model=settings.intent_openai_model,
        temperature=0,
    )


def _prompt_mentions_audio(prompt: str) -> bool:
    lowered = prompt.lower()
    return bool(
        re.search(
            r"\b(audio|volume|loud|louder|quieter|sound|gain|levels?)\b",
            lowered,
        )
    )


def _prompt_mentions_visual_style(prompt: str) -> bool:
    lowered = prompt.lower()
    return bool(
        re.search(
            r"\b(vintage|color|colour|warm|cool|grain|fade|saturation|contrast|exposure|look|style|mood|cinematic)\b",
            lowered,
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


def _intent_slices_from_request(request: UIWorkspacePlanRequest) -> list[IntentSlice]:
    prompt = request.prompt.strip()
    slices: list[IntentSlice] = []
    visual_params = _effect_parameter_ids(request.intentResult)
    if _prompt_mentions_visual_style(prompt) or visual_params:
        if not visual_params:
            visual_params = ["vintageIntensity", "temperature", "grain", "saturation"]
        slices.append(
            IntentSlice(
                id="visual_style",
                title="Visual style",
                goal="Preview and tune visual appearance",
                modality="visual",
                parameterIds=visual_params,
            )
        )
    if _prompt_mentions_audio(prompt):
        slices.append(
            IntentSlice(
                id="audio_loudness",
                title="Audio loudness",
                goal="Adjust loudness and monitor levels",
                modality="audio",
                parameterIds=["volumeGain"],
            )
        )
    if not slices:
        slices.append(
            IntentSlice(
                id="general_edit",
                title="Edit",
                goal="Review and refine the proposed edit",
                modality="mixed",
                parameterIds=[],
            )
        )
    return slices


class DeterministicUIWorkspacePlanner:
    def plan(self, request: UIWorkspacePlanRequest) -> UIWorkspacePlan:
        editor = request.editorContext or UIEditorContext()
        if not request.prompt.strip() and not request.intentResult:
            return self._default_workspace(editor)

        slices = _intent_slices_from_request(request)
        current_index = 0
        if request.currentWorkspaceId:
            for index, slice_ in enumerate(slices):
                if request.currentWorkspaceId == slice_.id:
                    current_index = index
                    break
        current_slice = slices[current_index]
        return self._workspace_for_slice(
            request=request,
            slices=slices,
            current_slice=current_slice,
            current_index=current_index,
            editor=editor,
        )

    def _default_workspace(self, editor: UIEditorContext) -> UIWorkspacePlan:
        toolbar_widgets: list[WidgetPlacement] = [
            WidgetPlacement(
                widgetId="toolbar.promptBar",
                variant="idle",
                prominence=Prominence.primary,
                intentSliceId="default",
                reason="Default prompt entry for general editing",
            )
        ]
        if editor.hasSelectedClip:
            toolbar_widgets.append(
                WidgetPlacement(
                    widgetId="toolbar.clipTools",
                    variant="collapsed",
                    prominence=Prominence.supporting,
                    intentSliceId="default",
                    reason="Basic clip operations when a clip is selected",
                )
            )
        layout_children: list[LayoutNode] = [
            LayoutNode(
                type=LayoutNodeType.widget,
                widget=WidgetPlacement(
                    widgetId="playback.viewer",
                    variant="compact",
                    prominence=Prominence.primary,
                    intentSliceId="default",
                    reason="Continuous visual feedback during editing",
                    size=WidgetSizeHint(weight=0.35, importance=Prominence.primary),
                ),
            ),
            LayoutNode(
                type=LayoutNodeType.widget,
                widget=WidgetPlacement(
                    widgetId="timeline.full",
                    variant="expanded" if editor.activeSpace == "Edit" else "compressed",
                    prominence=Prominence.primary,
                    intentSliceId="default",
                    reason="Timeline representation for navigation and direct edits",
                    size=WidgetSizeHint(weight=0.45, importance=Prominence.primary),
                ),
            ),
        ]
        show_nav = True
        if editor.activeSpace == "Import":
            layout_children.append(
                LayoutNode(
                    type=LayoutNodeType.widget,
                    widget=WidgetPlacement(
                        widgetId="panel.importBrowser",
                        variant="standard",
                        prominence=Prominence.supporting,
                        intentSliceId="default",
                        reason="Import flow requires media browsing",
                        size=WidgetSizeHint(weight=0.2),
                    ),
                )
            )
        if editor.activeSpace == "Export":
            layout_children.append(
                LayoutNode(
                    type=LayoutNodeType.widget,
                    widget=WidgetPlacement(
                        widgetId="panel.exportSettings",
                        variant="standard",
                        prominence=Prominence.supporting,
                        intentSliceId="default",
                        reason="Export flow requires output settings",
                        size=WidgetSizeHint(weight=0.2),
                    ),
                )
            )

        return validate_workspace_plan(
            UIWorkspacePlan(
                catalogVersion=CATALOG_VERSION,
                workspaceId="default",
                intentSummary="Default editor workspace",
                intentSlices=[
                    IntentSlice(
                        id="default",
                        title="Edit",
                        goal="General timeline editing",
                        modality="mixed",
                    )
                ],
                currentSliceId="default",
                currentSliceIndex=0,
                layout=LayoutNode(type=LayoutNodeType.vstack, children=layout_children),
                toolbar=ToolbarPlacement(
                    widgets=toolbar_widgets,
                    showNavigation=show_nav,
                    showPromptBar=True,
                ),
                isDefaultWorkspace=True,
                restoreDefaultOnComplete=False,
                hiddenBecauseIrrelevant=[
                    item
                    for item in (
                        "panel.importBrowser" if editor.activeSpace != "Import" else None,
                        "panel.exportSettings" if editor.activeSpace != "Export" else None,
                    )
                    if item
                ],
            )
        )

    def _workspace_for_slice(
        self,
        *,
        request: UIWorkspacePlanRequest,
        slices: list[IntentSlice],
        current_slice: IntentSlice,
        current_index: int,
        editor: UIEditorContext,
    ) -> UIWorkspacePlan:
        hidden: list[str] = [
            "panel.importBrowser",
            "panel.exportSettings",
            "toolbar.promptBar",
        ]
        layout_children: list[LayoutNode] = []
        toolbar_widgets: list[WidgetPlacement] = [
            WidgetPlacement(
                widgetId="toolbar.reviewActions",
                variant="standard",
                prominence=Prominence.supporting,
                intentSliceId=current_slice.id,
                reason="Allow apply, cancel, or refine during intent review",
            )
        ]

        if current_slice.modality == "visual":
            layout_children.extend(
                [
                    LayoutNode(
                        type=LayoutNodeType.widget,
                        widget=WidgetPlacement(
                            widgetId="playback.beforeAfterViewer",
                            variant="large",
                            prominence=Prominence.primary,
                            intentSliceId=current_slice.id,
                            reason="Visual intent requires before/after comparison",
                            size=WidgetSizeHint(weight=0.58, minHeight=220, importance=Prominence.primary),
                        ),
                    ),
                    LayoutNode(
                        type=LayoutNodeType.hstack,
                        size=WidgetSizeHint(weight=0.27),
                        children=[
                            LayoutNode(
                                type=LayoutNodeType.widget,
                                widget=WidgetPlacement(
                                    widgetId="timeline.focusedClipStrip",
                                    variant="affectedRangeOnly",
                                    prominence=Prominence.supporting,
                                    intentSliceId=current_slice.id,
                                    reason="Show only the affected clip region",
                                    size=WidgetSizeHint(weight=0.45),
                                ),
                            ),
                        ],
                    ),
                ]
            )
            if current_slice.parameterIds:
                toolbar_widgets.insert(
                    0,
                    toolbar_from_parameters(
                        current_slice.parameterIds,
                        intent_slice_id=current_slice.id,
                    ).widgets[0],
                )
        elif current_slice.modality == "audio":
            layout_children.extend(
                [
                    LayoutNode(
                        type=LayoutNodeType.widget,
                        widget=WidgetPlacement(
                            widgetId="playback.viewer",
                            variant="large",
                            prominence=Prominence.primary,
                            intentSliceId=current_slice.id,
                            reason="Audio changes still need playback feedback",
                            size=WidgetSizeHint(weight=0.5, importance=Prominence.primary),
                        ),
                    ),
                    LayoutNode(
                        type=LayoutNodeType.widget,
                        widget=WidgetPlacement(
                            widgetId="audio.levelsMeter",
                            variant="expanded",
                            prominence=Prominence.supporting,
                            intentSliceId=current_slice.id,
                            reason="Levels help validate loudness changes",
                            size=WidgetSizeHint(weight=0.2),
                        ),
                    ),
                ]
            )
            toolbar_widgets.insert(
                0,
                toolbar_from_parameters(
                    current_slice.parameterIds or ["volumeGain"],
                    intent_slice_id=current_slice.id,
                ).widgets[0],
            )
            hidden.append("timeline.full")
        else:
            layout_children.append(
                LayoutNode(
                    type=LayoutNodeType.widget,
                    widget=WidgetPlacement(
                        widgetId="playback.viewer",
                        variant="large",
                        prominence=Prominence.primary,
                        intentSliceId=current_slice.id,
                        reason="Preview proposed edit",
                        size=WidgetSizeHint(weight=0.55),
                    ),
                )
            )
            layout_children.append(
                LayoutNode(
                    type=LayoutNodeType.widget,
                    widget=WidgetPlacement(
                        widgetId="timeline.primaryTrack",
                        variant="affectedRangeOnly",
                        prominence=Prominence.supporting,
                        intentSliceId=current_slice.id,
                        reason="Focused timeline for the affected clip",
                        size=WidgetSizeHint(weight=0.25),
                    ),
                )
            )

        return validate_workspace_plan(
            UIWorkspacePlan(
                catalogVersion=CATALOG_VERSION,
                workspaceId=current_slice.id,
                intentSummary=request.prompt.strip() or "Intent workspace",
                intentSlices=slices,
                currentSliceId=current_slice.id,
                currentSliceIndex=current_index,
                layout=LayoutNode(type=LayoutNodeType.vstack, children=layout_children),
                toolbar=ToolbarPlacement(
                    widgets=toolbar_widgets,
                    showNavigation=len(slices) > 1,
                    showPromptBar=False,
                ),
                hiddenBecauseIrrelevant=[item for item in hidden if item],
                isDefaultWorkspace=False,
                restoreDefaultOnComplete=True,
            )
        )


class UIWorkspaceLLMPlanner:
    def __init__(self, llm: Any | None = None) -> None:
        self.llm = llm or _get_llm()

    async def plan(self, request: UIWorkspacePlanRequest) -> UIWorkspacePlan:
        llm = self.llm.with_structured_output(LLMWorkspaceDraft, method="function_calling")
        started_at = perf_counter()
        draft = await llm.ainvoke(
            [
                (
                    "system",
                    "You plan just-in-time editor UI workspaces. Choose only known widget ids, "
                    "variants, and parameters from the catalog. Every visible widget must serve "
                    "the active intent slice. Prefer focused widgets over general ones. "
                    "Put actionable controls in the toolbar. Hide navigation unless import/export "
                    "or multi-slice stepping requires it.",
                ),
                ("human", self._prompt(request)),
            ]
        )
        logger.info(
            "[ui-workspace] LLM planner completed slices=%s elapsed_ms=%s",
            len(draft.intentSlices),
            _elapsed_ms(started_at),
        )
        plan = UIWorkspacePlan(
            catalogVersion=CATALOG_VERSION,
            workspaceId=draft.currentSliceId or draft.intentSlices[0].id if draft.intentSlices else "workspace",
            intentSummary=draft.intentSummary,
            intentSlices=draft.intentSlices,
            currentSliceId=draft.currentSliceId,
            currentSliceIndex=0,
            layout=draft.layout,
            toolbar=draft.toolbar,
            hiddenBecauseIrrelevant=draft.hiddenBecauseIrrelevant,
            transitions=draft.transitions,
            isDefaultWorkspace=False,
            restoreDefaultOnComplete=True,
        )
        return validate_workspace_plan(plan)

    def _prompt(self, request: UIWorkspacePlanRequest) -> str:
        catalog = {
            "widgets": [
                {
                    "id": widget.id,
                    "variants": sorted(widget.variants),
                    "contributions": sorted(widget.intent_contributions),
                }
                for widget in sorted(WIDGET_CATALOG.values(), key=lambda item: item.id)
            ],
            "parameters": [
                {
                    "id": param.id,
                    "label": param.label,
                    "control": param.control,
                    "min": param.min_value,
                    "max": param.max_value,
                }
                for param in PARAMETER_CATALOG.values()
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


class UIWorkspacePlannerService:
    def __init__(
        self,
        *,
        deterministic: DeterministicUIWorkspacePlanner | None = None,
        llm_planner: UIWorkspaceLLMPlanner | None = None,
        use_llm: bool | None = None,
    ) -> None:
        self.deterministic = deterministic or DeterministicUIWorkspacePlanner()
        self.llm_planner = llm_planner or UIWorkspaceLLMPlanner()
        self.use_llm = use_llm if use_llm is not None else bool(settings.openai_api_key)

    async def plan(self, request: UIWorkspacePlanRequest) -> UIWorkspacePlan:
        if not request.prompt.strip() and not request.intentResult:
            return self.deterministic.plan(request)
        if not self.use_llm:
            return self.deterministic.plan(request)
        try:
            return await self.llm_planner.plan(request)
        except Exception:
            logger.exception("[ui-workspace] LLM planner failed; using deterministic fallback")
            return self.deterministic.plan(request)
