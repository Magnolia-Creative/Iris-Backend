from __future__ import annotations

from typing import Any, NotRequired, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from app.agent.intent.editing.models import IntentCompileResult, IntentCompilerContext


class IntentUIBaseModel(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class IntentUIEditorContext(IntentUIBaseModel):
    activeSpace: str | None = None
    hasSelectedClip: bool = False


class IntentUIPlanRequest(IntentUIBaseModel):
    prompt: str
    context: IntentCompilerContext
    editorContext: IntentUIEditorContext | None = None
    intentResult: IntentCompileResult | None = None
    currentWorkspaceId: str | None = None


class IntentUIPlan(IntentUIBaseModel):
    catalogVersion: str = "1"
    workspaceId: str
    intentSummary: str
    intentSlices: list[dict[str, Any]] = Field(default_factory=list)
    currentSliceId: str | None = None
    currentSliceIndex: int = 0
    layout: dict[str, Any] = Field(default_factory=dict)
    toolbar: dict[str, Any] = Field(default_factory=dict)
    transitions: list[dict[str, Any]] = Field(default_factory=list)
    hiddenBecauseIrrelevant: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    isDefaultWorkspace: bool = False
    restoreDefaultOnComplete: bool = True


class LLMIntentUIPlanDraft(IntentUIBaseModel):
    intentSummary: str
    intentSlices: list[dict[str, Any]] = Field(default_factory=list)
    currentSliceId: str | None = None
    layout: dict[str, Any] = Field(default_factory=dict)
    toolbar: dict[str, Any] = Field(default_factory=dict)
    hiddenBecauseIrrelevant: list[str] = Field(default_factory=list)
    transitions: list[dict[str, Any]] = Field(default_factory=list)


class UIIntentGraphState(TypedDict, total=False):
    prompt: str
    context: IntentCompilerContext
    editor_context: IntentUIEditorContext | None
    current_workspace_id: str | None
    intent_result: IntentCompileResult | None
    plan: NotRequired[IntentUIPlan]
    events: NotRequired[list[dict[str, Any]]]
    elapsed_ms: NotRequired[int]
