from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.intent_compiler.models import IntentCompileResult, IntentCompilerContext


class UIWorkspaceBaseModel(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class WidgetRole(StrEnum):
    monitor = "monitor"
    representation = "representation"
    inspector = "inspector"
    tool = "tool"
    navigator = "navigator"
    review = "review"


class ControlType(StrEnum):
    slider = "slider"
    toggle = "toggle"
    segmented = "segmented"
    button = "button"
    textField = "textField"


class Prominence(StrEnum):
    primary = "primary"
    supporting = "supporting"
    compact = "compact"
    hidden = "hidden"


class LayoutNodeType(StrEnum):
    vstack = "vstack"
    hstack = "hstack"
    zstack = "zstack"
    widget = "widget"
    toolbar = "toolbar"


class WidgetSizeHint(UIWorkspaceBaseModel):
    weight: float | None = None
    minHeight: float | None = None
    maxHeight: float | None = None
    importance: Prominence | None = None
    collapsible: bool = False


class ParameterControl(UIWorkspaceBaseModel):
    parameterId: str
    control: ControlType
    label: str | None = None
    minValue: float | None = None
    maxValue: float | None = None
    defaultValue: float | None = None


class WidgetPlacement(UIWorkspaceBaseModel):
    widgetId: str
    variant: str | None = None
    prominence: Prominence = Prominence.supporting
    size: WidgetSizeHint | None = None
    intentSliceId: str
    reason: str | None = None
    controls: list[ParameterControl] = Field(default_factory=list)
    props: dict[str, Any] = Field(default_factory=dict)


class LayoutNode(UIWorkspaceBaseModel):
    type: LayoutNodeType
    children: list[LayoutNode] = Field(default_factory=list)
    widget: WidgetPlacement | None = None
    size: WidgetSizeHint | None = None


class IntentSlice(UIWorkspaceBaseModel):
    id: str
    title: str
    goal: str
    modality: Literal["visual", "audio", "timeline", "caption", "export", "import", "mixed"] = "mixed"
    parameterIds: list[str] = Field(default_factory=list)


class ToolbarPlacement(UIWorkspaceBaseModel):
    widgets: list[WidgetPlacement] = Field(default_factory=list)
    showNavigation: bool = False
    showPromptBar: bool = False


class TransitionHint(UIWorkspaceBaseModel):
    fromWidgetId: str | None = None
    toWidgetId: str | None = None
    style: Literal["persist", "resize", "replace", "enter", "exit"] = "replace"


class UIWorkspacePlan(UIWorkspaceBaseModel):
    catalogVersion: str = "1"
    workspaceId: str
    intentSummary: str
    intentSlices: list[IntentSlice] = Field(default_factory=list)
    currentSliceId: str | None = None
    currentSliceIndex: int = 0
    layout: LayoutNode
    toolbar: ToolbarPlacement
    transitions: list[TransitionHint] = Field(default_factory=list)
    hiddenBecauseIrrelevant: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    isDefaultWorkspace: bool = False
    restoreDefaultOnComplete: bool = True


class UIEditorContext(UIWorkspaceBaseModel):
    """Lightweight editor facts for UI planning."""

    activeSpace: str | None = None
    hasSelectedClip: bool = False
    hasSelectedCaption: bool = False
    hasVideoClips: bool = False
    hasAudioClips: bool = False
    isReviewActive: bool = False
    isPromptActionReviewActive: bool = False
    isCaptionsChromeActive: bool = False
    clientCatalogVersion: str | None = None


class UIWorkspacePlanRequest(UIWorkspaceBaseModel):
    prompt: str
    context: IntentCompilerContext
    editorContext: UIEditorContext | None = None
    intentResult: IntentCompileResult | None = None
    currentWorkspaceId: str | None = None


class UIWorkspacePlanResponse(UIWorkspaceBaseModel):
    plan: UIWorkspacePlan


class LLMWorkspaceDraft(UIWorkspaceBaseModel):
    """Structured output from the LLM workspace planner."""

    intentSummary: str
    intentSlices: list[IntentSlice] = Field(default_factory=list)
    currentSliceId: str | None = None
    layout: LayoutNode
    toolbar: ToolbarPlacement
    hiddenBecauseIrrelevant: list[str] = Field(default_factory=list)
    transitions: list[TransitionHint] = Field(default_factory=list)
