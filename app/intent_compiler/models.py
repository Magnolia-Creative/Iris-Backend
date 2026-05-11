from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


JSONValue = Any


class IntentCompilerBaseModel(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class TimeRange(IntentCompilerBaseModel):
    start: int
    end: int

    @property
    def duration(self) -> int:
        return self.end - self.start


class Clip(IntentCompilerBaseModel):
    clipId: str = Field(alias="clip_id")
    trackId: str = Field(alias="track_id")
    mediaId: str = Field(alias="media_id")
    sourceRange: TimeRange = Field(alias="source_range")
    timelineRange: TimeRange = Field(alias="timeline_range")


class IntentCompilerContext(IntentCompilerBaseModel):
    timelineId: str
    selectedClipId: str | None = None
    selectedTrackId: str | None = None
    selectedRange: TimeRange | None = None
    playheadTimeUs: int | None = None
    clipsById: dict[str, Clip] = Field(default_factory=dict)
    orderedClipIdsByTrackId: dict[str, list[str]] = Field(default_factory=dict)

    def clip(self, clip_id: str | None) -> Clip | None:
        if clip_id is None:
            return None
        return self.clipsById.get(clip_id)


class CompileSource(StrEnum):
    deterministic = "deterministic"
    embedding = "embedding"
    llm = "llm"
    mixed = "mixed"


class IntentCompileWarning(StrEnum):
    missingSelectedClip = "missingSelectedClip"
    missingPlayhead = "missingPlayhead"
    missingSelectedTrack = "missingSelectedTrack"
    clipNotFound = "clipNotFound"
    trackNotFound = "trackNotFound"
    splitTimeOutsideClip = "splitTimeOutsideClip"
    invalidTrimRange = "invalidTrimRange"
    invalidMoveOrder = "invalidMoveOrder"
    unsupportedAction = "unsupportedAction"
    unsupportedIntent = "unsupportedIntent"
    ambiguousTarget = "ambiguousTarget"
    noActionProduced = "noActionProduced"
    embeddingUnavailable = "embeddingUnavailable"
    llmUnavailable = "llmUnavailable"
    invalidLLMResponse = "invalidLLMResponse"
    lowConfidence = "lowConfidence"
    destructiveActionNeedsClarification = "destructiveActionNeedsClarification"


class IntentEditType(StrEnum):
    splitClip = "splitClip"
    removeClip = "removeClip"
    trimClip = "trimClip"
    moveClip = "moveClip"
    replaceTrackClips = "replaceTrackClips"
    unknown = "unknown"


class ActionType(StrEnum):
    addClip = "ADD_CLIP"
    removeClip = "REMOVE_CLIP"
    trimClip = "TRIM_CLIP"
    splitClip = "SPLIT_CLIP"
    moveClip = "MOVE_CLIP"
    replaceTrackClips = "REPLACE_TRACK_CLIPS"


class Action(IntentCompilerBaseModel):
    actionId: str = Field(alias="action_id")
    timelineId: str = Field(alias="timeline_id")
    createdAt: float = Field(alias="created_at")
    type: ActionType
    payload: dict[str, Any]
    groupId: str | None = Field(default=None, alias="group_id")


class SemanticTrackReference(IntentCompilerBaseModel):
    type: Literal["selectedTrack", "trackId"]
    trackId: str | None = None


class SemanticClipReference(IntentCompilerBaseModel):
    type: Literal["selectedClip", "clipId", "sameAsPrevious", "ordinal", "currentClipAtPlayhead"]
    clipId: str | None = None
    value: str | None = None
    track: SemanticTrackReference | None = None


SemanticEditTarget = SemanticClipReference | SemanticTrackReference


class SemanticEditOperation(IntentCompilerBaseModel):
    type: IntentEditType
    sourceText: str
    target: SemanticEditTarget | None = None
    parameters: dict[str, JSONValue] = Field(default_factory=dict)
    confidence: float | None = None

    @field_validator("confidence", mode="after")
    @classmethod
    def default_confidence(cls, value: float | None, info: Any) -> float:
        if value is not None:
            return value
        op_type = info.data.get("type")
        return 0.1 if op_type == IntentEditType.unknown else 0.8


class SemanticEffectRequest(IntentCompilerBaseModel):
    sourceText: str
    target: SemanticEditTarget | None = None
    intent: str | None = None
    attributes: list[str] = Field(default_factory=list)
    confidence: float = 0.7

    @field_validator("intent", mode="after")
    @classmethod
    def default_intent(cls, value: str | None, info: Any) -> str:
        return value or info.data.get("sourceText") or ""


class EffectCapabilityParameter(IntentCompilerBaseModel):
    name: str
    valueType: str
    minimum: float | None = None
    maximum: float | None = None
    description: str


class EffectCapability(IntentCompilerBaseModel):
    operation: str
    description: str
    parameters: list[EffectCapabilityParameter]
    retrievalText: str
    examples: list[str]


class RelevantEffectCapability(IntentCompilerBaseModel):
    capability: EffectCapability
    score: float


class ExperimentalEffectOperation(IntentCompilerBaseModel):
    operation: str
    sourceText: str
    target: SemanticEditTarget | None = None
    confidence: float
    parameters: dict[str, JSONValue] = Field(default_factory=dict)


class ExperimentalEffectPlan(IntentCompilerBaseModel):
    operations: list[ExperimentalEffectOperation] = Field(default_factory=list)
    rationale: str | None = None


class SemanticEditPlan(IntentCompilerBaseModel):
    operations: list[SemanticEditOperation] = Field(default_factory=list)
    effectRequests: list[SemanticEffectRequest] = Field(default_factory=list)
    experimentalEffectOperations: list[ExperimentalEffectOperation] = Field(default_factory=list)
    needsClarification: bool = False
    clarificationQuestion: str | None = None


class IntentCompileResult(IntentCompilerBaseModel):
    actions: list[Action]
    confidence: float
    source: CompileSource
    unresolvedText: str | None
    warnings: list[IntentCompileWarning]
    needsClarification: bool
    experimentalEffectOperations: list[ExperimentalEffectOperation] = Field(default_factory=list)


class IntentCompileRequest(IntentCompilerBaseModel):
    prompt: str
    context: IntentCompilerContext


class IntentRunCreateResponse(IntentCompilerBaseModel):
    run_id: str
    websocket_url: str

