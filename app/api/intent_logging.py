from typing import Any

from app.agent.intent.editing.models import IntentCompilerContext


def intent_context_log_summary(
    context: IntentCompilerContext,
    *,
    hydration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "timeline_id": context.timelineId,
        "project_id": context.projectId,
        "session_id": context.sessionId,
        "selected_clip_id": context.selectedClipId,
        "selected_track_id": context.selectedTrackId,
        "clip_count": len(context.clipsById),
        "track_count": len(context.orderedClipIdsByTrackId),
        "transcript_context_count": len(context.transcriptContextsByClipId),
    }
    if hydration:
        summary["transcript_hydration"] = hydration
    return summary


def intent_result_log_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "action_count": len(result.get("actions") or []),
        "experimental_effect_count": len(result.get("experimentalEffectOperations") or []),
        "warnings": result.get("warnings") or [],
        "needs_clarification": result.get("needsClarification"),
        "source": result.get("source"),
    }
