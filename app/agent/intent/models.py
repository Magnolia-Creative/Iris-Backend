from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.agent.intent.editing.models import IntentCompileResult, IntentCompilerContext
from app.agent.intent.ui.models import IntentUIEditorContext, IntentUIPlan


class IntentAgentBaseModel(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class IntentAgentRequest(IntentAgentBaseModel):
    prompt: str
    context: IntentCompilerContext
    editorContext: IntentUIEditorContext | None = None
    currentWorkspaceId: str | None = None


class IntentAgentTiming(IntentAgentBaseModel):
    branch: Literal["edit", "ui", "total"]
    elapsed_ms: int


class IntentAgentMeta(IntentAgentBaseModel):
    hydration: dict[str, Any] = Field(default_factory=dict)
    edit_events: list[dict[str, Any]] = Field(default_factory=list)
    ui_events: list[dict[str, Any]] = Field(default_factory=list)
    timings: list[IntentAgentTiming] = Field(default_factory=list)


class IntentAgentResponse(IntentAgentBaseModel):
    edit: IntentCompileResult
    ui: IntentUIPlan
    meta: IntentAgentMeta = Field(default_factory=IntentAgentMeta)
