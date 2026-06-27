from typing import Any, Literal

from pydantic import BaseModel, Field
from typing_extensions import NotRequired, TypedDict


NextAction = Literal["hydrate_transcripts", "clip_cleanup", "timeline_planner", "finish"]
SESSION_GRAPH_STATE_VERSION = 1


class RetrievalPlan(TypedDict, total=False):
    clip_ids: list[str]
    reason: str
    query_focus: list[str]


class EditPlan(TypedDict, total=False):
    objective: str
    target_style: str
    target_clips: list[str]
    constraints: list[str]


class CleanupPlan(TypedDict, total=False):
    selected_clip_ids: list[str]
    dropped_clip_ids: list[str]
    trim_suggestions: list[dict[str, Any]]
    cleanup_notes: list[str]


class TimelineEntry(TypedDict, total=False):
    clip_id: str
    local_key: str | None
    in_sec: float
    out_sec: float
    rationale: str


class ClipState(TypedDict, total=False):
    clip_id: str
    local_key: str | None
    transcript_id: int
    summary: str
    metadata: dict[str, Any]
    transcript_cached: bool
    transcript_cache_key: str | None


class SessionGraphState(TypedDict, total=False):
    graph_state_version: int
    session_id: str
    project_id: int
    user_prompt: str
    clips: list[ClipState]
    next_action: NotRequired[NextAction | None]
    retrieval_plan: NotRequired[RetrievalPlan | None]
    edit_plan: NotRequired[EditPlan | None]
    cleanup_plan: NotRequired[CleanupPlan | None]
    timeline: NotRequired[list[TimelineEntry]]
    timeline_notes: NotRequired[list[str]]
    waiting_for_user: bool
    force_reconsider: NotRequired[bool]
    iteration_count: int
    notes: list[str]
    errors: list[str]
    status_message: NotRequired[str]
    status_details: NotRequired[dict[str, Any]]


def normalize_session_graph_state(
    graph_state: dict[str, Any],
    *,
    session_id: int | str,
) -> SessionGraphState:
    normalized: SessionGraphState = dict(graph_state)
    normalized["graph_state_version"] = int(
        normalized.get("graph_state_version") or SESSION_GRAPH_STATE_VERSION
    )
    normalized["session_id"] = str(session_id)
    normalized.setdefault("waiting_for_user", False)
    normalized.setdefault("iteration_count", 0)
    normalized.setdefault("notes", [])
    normalized.setdefault("errors", [])
    normalized.setdefault("clips", [])
    return normalized


class RetrievalPlanModel(BaseModel):
    clip_ids: list[str] = Field(default_factory=list)
    reason: str = ""
    query_focus: list[str] = Field(default_factory=list)


class EditPlanModel(BaseModel):
    objective: str = ""
    target_style: str = ""
    target_clips: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)


class DecisionAgentOutput(BaseModel):
    next_action: NextAction
    retrieval_plan: RetrievalPlanModel = Field(default_factory=RetrievalPlanModel)
    edit_plan: EditPlanModel = Field(default_factory=EditPlanModel)
    reasoning_notes: list[str] = Field(default_factory=list)


class TrimSuggestionModel(BaseModel):
    clip_id: str
    in_sec: float
    out_sec: float
    reason: str


class ClipCleanupOutput(BaseModel):
    selected_clip_ids: list[str] = Field(default_factory=list)
    dropped_clip_ids: list[str] = Field(default_factory=list)
    trim_suggestions: list[TrimSuggestionModel] = Field(default_factory=list)
    cleanup_notes: list[str] = Field(default_factory=list)


class TimelinePlannerEntryModel(BaseModel):
    clip_id: str
    local_key: str | None = None
    in_sec: float
    out_sec: float
    rationale: str


class TimelinePlannerOutput(BaseModel):
    timeline: list[TimelinePlannerEntryModel] = Field(default_factory=list)
    timeline_notes: list[str] = Field(default_factory=list)


class ValidatedTimelineEntry(BaseModel):
    clip_id: str
    local_key: str | None = None
    in_sec: float
    out_sec: float
    rationale: str = ""


class TimelineValidationResult(BaseModel):
    is_valid: bool
    normalized_timeline: list[ValidatedTimelineEntry] = Field(default_factory=list)
    validation_notes: list[str] = Field(default_factory=list)
    validation_errors: list[str] = Field(default_factory=list)
