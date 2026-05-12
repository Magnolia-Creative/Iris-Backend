from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import re
from typing import Any, Literal
from uuid import uuid4

from app.intent_compiler.models import (
    Action,
    ActionType,
    Clip,
    CompileSource,
    IntentCompileResult,
    IntentCompileWarning,
    IntentCompilerContext,
    IntentEditType,
    SemanticClipReference,
    SemanticEditOperation,
    SemanticEditPlan,
    SemanticEditTarget,
    SemanticTrackReference,
    TimeRange,
)


DurationUnit = Literal["microsecond", "millisecond", "second", "minute"]


@dataclass(frozen=True)
class DurationExpression:
    kind: Literal["duration", "percentage", "vague"]
    value: float | None = None
    unit: DurationUnit | None = None
    phrase: str | None = None


@dataclass(frozen=True)
class TimeExpression:
    kind: Literal["playhead", "absoluteTimelineTime", "fractionOfClip", "afterStart", "beforeEnd"]
    value: float | None = None
    unit: DurationUnit | None = None
    amount: DurationExpression | None = None


@dataclass(frozen=True)
class ResolvedOperation:
    type: IntentEditType
    sourceText: str
    targetClipId: str | None
    targetTrackId: str | None
    confidence: float
    parameters: dict[str, Any]


@dataclass(frozen=True)
class Resolution:
    status: Literal["success", "clarification", "unsupported"]
    operation: ResolvedOperation | None = None
    warning: IntentCompileWarning | None = None
    question: str | None = None


class IntentTimelineSimulator:
    def __init__(self, context: IntentCompilerContext) -> None:
        self.clips_by_id = {clip_id: clip.model_copy(deep=True) for clip_id, clip in context.clipsById.items()}
        self.ordered_clip_ids_by_track_id = {
            track_id: list(clip_ids) for track_id, clip_ids in context.orderedClipIdsByTrackId.items()
        }

    def clip(self, clip_id: str) -> Clip | None:
        return self.clips_by_id.get(clip_id)

    def ordered_clip_ids(self, track_id: str) -> list[str]:
        return self.ordered_clip_ids_by_track_id.get(track_id, [])

    def clip_at(self, time_us: int, preferred_track_id: str | None) -> Clip | None:
        if preferred_track_id:
            for clip_id in self.ordered_clip_ids(preferred_track_id):
                clip = self.clip(clip_id)
                if clip and clip.timelineRange.start <= time_us < clip.timelineRange.end:
                    return clip
        for track_id in sorted(self.ordered_clip_ids_by_track_id):
            for clip_id in self.ordered_clip_ids(track_id):
                clip = self.clip(clip_id)
                if clip and clip.timelineRange.start <= time_us < clip.timelineRange.end:
                    return clip
        return None

    def apply(self, operation: ResolvedOperation) -> None:
        params = operation.parameters
        if operation.type == IntentEditType.splitClip and operation.targetClipId:
            self._apply_split(operation.targetClipId, int(params["atTimeUs"]))
        elif operation.type == IntentEditType.removeClip and operation.targetClipId:
            self._apply_remove(operation.targetClipId)
        elif operation.type == IntentEditType.trimClip and operation.targetClipId:
            self._apply_trim(
                operation.targetClipId,
                TimeRange.model_validate(params["sourceRange"]),
            )
        elif operation.type == IntentEditType.removeClipRanges and operation.targetClipId:
            self._apply_remove_ranges(
                operation.targetClipId,
                [TimeRange.model_validate(value) for value in params["sourceRanges"]],
            )
        elif operation.type == IntentEditType.moveClip and operation.targetClipId:
            clip = self.clip(operation.targetClipId)
            if clip:
                self.ordered_clip_ids_by_track_id[clip.trackId] = list(params["orderedClipIds"])
        elif operation.type == IntentEditType.replaceTrackClips and operation.targetTrackId:
            clips = [Clip.model_validate(clip) for clip in params["clips"]]
            for clip in clips:
                self.clips_by_id[clip.clipId] = clip
            self.ordered_clip_ids_by_track_id[operation.targetTrackId] = [clip.clipId for clip in clips]

    def _apply_split(self, clip_id: str, at_time_us: int) -> None:
        clip = self.clip(clip_id)
        if clip is None:
            return
        left_duration = at_time_us - clip.timelineRange.start
        source_mid = clip.sourceRange.start + left_duration
        left_clip = clip.model_copy(
            update={
                "sourceRange": TimeRange(start=clip.sourceRange.start, end=source_mid),
                "timelineRange": TimeRange(start=clip.timelineRange.start, end=at_time_us),
            }
        )
        right_clip_id = f"{clip_id}-split-right"
        right_clip = Clip(
            clip_id=right_clip_id,
            track_id=clip.trackId,
            media_id=clip.mediaId,
            source_range=TimeRange(start=source_mid, end=clip.sourceRange.end),
            timeline_range=TimeRange(start=at_time_us, end=clip.timelineRange.end),
        )
        self.clips_by_id[clip_id] = left_clip
        self.clips_by_id[right_clip_id] = right_clip
        order = self.ordered_clip_ids(clip.trackId)
        if clip_id in order:
            index = order.index(clip_id)
            self.ordered_clip_ids_by_track_id[clip.trackId] = order[: index + 1] + [right_clip_id] + order[index + 1 :]

    def _apply_remove(self, clip_id: str) -> None:
        clip = self.clip(clip_id)
        if clip is None:
            return
        self.ordered_clip_ids_by_track_id[clip.trackId] = [
            candidate for candidate in self.ordered_clip_ids(clip.trackId) if candidate != clip_id
        ]

    def _apply_trim(self, clip_id: str, source_range: TimeRange) -> None:
        clip = self.clip(clip_id)
        if clip is None or source_range.duration <= 0:
            return
        self.clips_by_id[clip_id] = clip.model_copy(
            update={
                "sourceRange": source_range,
                "timelineRange": TimeRange(
                    start=clip.timelineRange.start,
                    end=clip.timelineRange.start + source_range.duration,
                ),
            }
        )
        self._pack_track(clip.trackId)

    def _apply_remove_ranges(self, clip_id: str, source_ranges: list[TimeRange]) -> None:
        clip = self.clip(clip_id)
        if clip is None:
            return
        removal_ranges = _normalized_remove_ranges(source_ranges, clip.sourceRange)
        if not removal_ranges:
            return
        survivor_ranges = _survivor_ranges(clip.sourceRange, removal_ranges)
        order = self.ordered_clip_ids(clip.trackId)
        if not survivor_ranges:
            self._apply_remove(clip_id)
            return

        replacement_ids: list[str] = []
        for index, source_range in enumerate(survivor_ranges):
            survivor_id = clip_id if index == 0 else f"{clip_id}-range-{index + 1}"
            replacement_ids.append(survivor_id)
            self.clips_by_id[survivor_id] = Clip(
                clip_id=survivor_id,
                track_id=clip.trackId,
                media_id=clip.mediaId,
                source_range=source_range,
                timeline_range=TimeRange(start=0, end=source_range.duration),
            )
        if clip_id in order:
            clip_index = order.index(clip_id)
            self.ordered_clip_ids_by_track_id[clip.trackId] = order[:clip_index] + replacement_ids + order[clip_index + 1 :]
        self._pack_track(clip.trackId)

    def _pack_track(self, track_id: str) -> None:
        cursor = 0
        for clip_id in self.ordered_clip_ids(track_id):
            clip = self.clip(clip_id)
            if clip is None:
                continue
            duration = clip.timelineRange.duration
            self.clips_by_id[clip_id] = clip.model_copy(
                update={"timelineRange": TimeRange(start=cursor, end=cursor + duration)}
            )
            cursor += duration


class IntentCompiler:
    def compile(
        self,
        plan: SemanticEditPlan,
        *,
        original_prompt: str,
        context: IntentCompilerContext,
    ) -> IntentCompileResult:
        if plan.needsClarification:
            return self._clarification(
                original_prompt,
                plan.clarificationQuestion,
                [IntentCompileWarning.ambiguousTarget],
                plan,
            )

        simulator = IntentTimelineSimulator(context)
        previous_clip_id: str | None = None
        previous_track_id: str | None = None
        actions: list[Action] = []
        confidences: list[float] = []
        warnings: list[IntentCompileWarning] = []

        for operation in plan.operations:
            resolved = self._resolve_operation(operation, context, simulator, previous_clip_id, previous_track_id)
            if resolved.status == "success" and resolved.operation:
                action = self._make_action(resolved.operation, context)
                if action is None:
                    warnings.append(IntentCompileWarning.unsupportedIntent)
                    continue
                actions.append(action)
                confidences.append(resolved.operation.confidence)
                simulator.apply(resolved.operation)
                previous_clip_id = resolved.operation.targetClipId or previous_clip_id
                previous_track_id = resolved.operation.targetTrackId or previous_track_id
            elif resolved.status == "clarification":
                return self._clarification(
                    original_prompt,
                    resolved.question,
                    [resolved.warning or IntentCompileWarning.ambiguousTarget],
                    plan,
                )
            else:
                warnings.append(IntentCompileWarning.unsupportedIntent)

        if plan.experimentalEffectOperations:
            warnings.append(IntentCompileWarning.unsupportedAction)

        if not actions:
            return IntentCompileResult(
                actions=[],
                confidence=0,
                source=CompileSource.llm,
                unresolvedText=original_prompt,
                warnings=self._unique_warnings(warnings) or [IntentCompileWarning.noActionProduced],
                needsClarification=False,
                experimentalEffectOperations=plan.experimentalEffectOperations,
            )

        return self._validated_result(
            IntentCompileResult(
                actions=actions,
                confidence=sum(confidences) / len(confidences),
                source=CompileSource.llm,
                unresolvedText=None,
                warnings=self._unique_warnings(warnings),
                needsClarification=False,
                experimentalEffectOperations=plan.experimentalEffectOperations,
            ),
            context,
        )

    def _resolve_operation(
        self,
        operation: SemanticEditOperation,
        context: IntentCompilerContext,
        simulator: IntentTimelineSimulator,
        previous_clip_id: str | None,
        previous_track_id: str | None,
    ) -> Resolution:
        match operation.type:
            case IntentEditType.splitClip:
                return self._resolve_split(operation, context, simulator, previous_clip_id)
            case IntentEditType.removeClip:
                return self._resolve_remove(operation, context, simulator, previous_clip_id)
            case IntentEditType.trimClip:
                return self._resolve_trim(operation, context, simulator, previous_clip_id)
            case IntentEditType.removeClipRanges:
                return self._resolve_remove_ranges(operation, context, simulator, previous_clip_id)
            case IntentEditType.moveClip:
                return self._resolve_move(operation, context, simulator, previous_clip_id)
            case IntentEditType.replaceTrackClips:
                return self._resolve_replace_track(operation, context, simulator, previous_track_id)
            case _:
                return Resolution(status="unsupported")

    def _resolve_split(
        self,
        operation: SemanticEditOperation,
        context: IntentCompilerContext,
        simulator: IntentTimelineSimulator,
        previous_clip_id: str | None,
    ) -> Resolution:
        clip = self._resolve_clip(operation.target, context, simulator, previous_clip_id)
        if clip is None:
            return self._needs(IntentCompileWarning.missingSelectedClip, "Which clip do you want to split?")
        position = self._time_expression(operation.parameters.get("position"))
        if position is None:
            return self._needs(IntentCompileWarning.missingPlayhead, "Where do you want to split the clip?")
        at_time_us = self._resolve_time_us(position, clip, context, operation.sourceText)
        if at_time_us is None:
            return self._needs(IntentCompileWarning.missingPlayhead, "Where do you want to split the clip?")
        if at_time_us <= clip.timelineRange.start or at_time_us >= clip.timelineRange.end:
            return self._needs(
                IntentCompileWarning.splitTimeOutsideClip,
                "The split point is outside the clip. Where should I split it?",
            )
        return self._success(operation, clip_id=clip.clipId, parameters={"atTimeUs": at_time_us})

    def _resolve_remove(
        self,
        operation: SemanticEditOperation,
        context: IntentCompilerContext,
        simulator: IntentTimelineSimulator,
        previous_clip_id: str | None,
    ) -> Resolution:
        clip = self._resolve_clip(operation.target, context, simulator, previous_clip_id)
        if clip is None:
            return self._needs(IntentCompileWarning.missingSelectedClip, "Which clip do you want to delete?")
        return self._success(operation, clip_id=clip.clipId, parameters={})

    def _resolve_trim(
        self,
        operation: SemanticEditOperation,
        context: IntentCompilerContext,
        simulator: IntentTimelineSimulator,
        previous_clip_id: str | None,
    ) -> Resolution:
        clip = self._resolve_clip(operation.target, context, simulator, previous_clip_id)
        if clip is None:
            return self._needs(IntentCompileWarning.missingSelectedClip, "Which clip do you want to trim?")
        edge = str(operation.parameters.get("edge") or "").lower() or self._infer_trim_edge(operation.sourceText)
        amount = self._duration_expression(operation.parameters.get("amount"))
        duration_us = self._resolve_duration_us(amount, clip, operation.sourceText) if amount else None
        if duration_us is None:
            duration_us = self._explicit_duration_us(operation.sourceText)
        if duration_us is None or duration_us <= 0 or duration_us >= clip.timelineRange.duration:
            return self._needs(IntentCompileWarning.invalidTrimRange, "How much do you want to trim?")
        if edge in {"start", "beginning"}:
            source_range = TimeRange(start=clip.sourceRange.start + duration_us, end=clip.sourceRange.end)
        elif edge in {"end", "final"}:
            source_range = TimeRange(start=clip.sourceRange.start, end=clip.sourceRange.end - duration_us)
        else:
            return self._needs(IntentCompileWarning.invalidTrimRange, "Should I trim the start or end of the clip?")
        return self._success(
            operation,
            clip_id=clip.clipId,
            parameters={"sourceRange": source_range.model_dump()},
        )

    def _resolve_remove_ranges(
        self,
        operation: SemanticEditOperation,
        context: IntentCompilerContext,
        simulator: IntentTimelineSimulator,
        previous_clip_id: str | None,
    ) -> Resolution:
        clip = self._resolve_clip(operation.target, context, simulator, previous_clip_id)
        if clip is None:
            return self._needs(IntentCompileWarning.missingSelectedClip, "Which clip do you want to remove dead space from?")

        source_ranges = self._source_ranges(operation.parameters.get("sourceRanges"))
        if not source_ranges and _dead_space_terms(operation.sourceText):
            source_ranges = [
                TimeRange(start=pause.startUs, end=pause.endUs)
                for pause in context.transcriptContextsByClipId.get(clip.clipId, None).pauseRanges
            ] if context.transcriptContextsByClipId.get(clip.clipId) else []
        if not source_ranges:
            return self._needs(
                IntentCompileWarning.missingTranscriptContext,
                "I need transcript timing for this clip before I can remove the dead space.",
            )

        removal_ranges = _normalized_remove_ranges(source_ranges, clip.sourceRange)
        if not removal_ranges:
            return self._needs(IntentCompileWarning.invalidRemoveRange, "Which source ranges should I remove?")
        return self._success(
            operation,
            clip_id=clip.clipId,
            parameters={"sourceRanges": [source_range.model_dump() for source_range in removal_ranges]},
        )

    def _resolve_move(
        self,
        operation: SemanticEditOperation,
        context: IntentCompilerContext,
        simulator: IntentTimelineSimulator,
        previous_clip_id: str | None,
    ) -> Resolution:
        clip = self._resolve_clip(operation.target, context, simulator, previous_clip_id)
        if clip is None:
            return self._needs(IntentCompileWarning.missingSelectedClip, "Which clip do you want to move?")
        original_order = simulator.ordered_clip_ids(clip.trackId)
        ordered_clip_ids = operation.parameters.get("orderedClipIds")
        if isinstance(ordered_clip_ids, list):
            new_order = [str(value) for value in ordered_clip_ids if isinstance(value, str)]
        else:
            placement = str(operation.parameters.get("placement") or "").lower()
            new_order = [candidate for candidate in original_order if candidate != clip.clipId]
            if placement in {"beginning", "start", "first"}:
                new_order.insert(0, clip.clipId)
            elif placement in {"end", "last"}:
                new_order.append(clip.clipId)
            else:
                return self._needs(IntentCompileWarning.ambiguousTarget, "Where should this clip move?")
        if set(original_order) != set(new_order) or len(original_order) != len(new_order) or len(set(new_order)) != len(new_order):
            return self._needs(IntentCompileWarning.invalidMoveOrder, "The requested clip order is not valid.")
        return self._success(operation, clip_id=clip.clipId, parameters={"orderedClipIds": new_order})

    def _resolve_replace_track(
        self,
        operation: SemanticEditOperation,
        context: IntentCompilerContext,
        simulator: IntentTimelineSimulator,
        previous_track_id: str | None,
    ) -> Resolution:
        track_id = self._resolve_track_id(operation.target, context, previous_track_id)
        if track_id is None:
            return self._needs(IntentCompileWarning.missingSelectedTrack, "Which track do you want to reorder?")
        ordered_value = operation.parameters.get("orderedClipIds")
        if not isinstance(ordered_value, list):
            return self._needs(IntentCompileWarning.invalidMoveOrder, "What clip order should I use?")
        ordered_clip_ids = [str(value) for value in ordered_value if isinstance(value, str)]
        existing_order = simulator.ordered_clip_ids(track_id)
        if (
            set(existing_order) != set(ordered_clip_ids)
            or len(existing_order) != len(ordered_clip_ids)
            or len(set(ordered_clip_ids)) != len(ordered_clip_ids)
        ):
            return self._needs(IntentCompileWarning.invalidMoveOrder, "The requested clip order is not valid.")
        clips = [simulator.clip(clip_id) for clip_id in ordered_clip_ids]
        if any(clip is None for clip in clips):
            return self._needs(IntentCompileWarning.clipNotFound, "One of those clips is not available.")
        return self._success(
            operation,
            track_id=track_id,
            parameters={"clips": [clip.model_dump(by_alias=True) for clip in clips if clip is not None]},
        )

    def _resolve_clip(
        self,
        target: SemanticEditTarget | None,
        context: IntentCompilerContext,
        simulator: IntentTimelineSimulator,
        previous_clip_id: str | None,
    ) -> Clip | None:
        if target is None:
            return simulator.clip(context.selectedClipId) if context.selectedClipId else None
        if not isinstance(target, SemanticClipReference):
            return None
        if target.type == "selectedClip":
            return simulator.clip(context.selectedClipId) if context.selectedClipId else None
        if target.type == "clipId":
            return simulator.clip(target.clipId) if target.clipId else None
        if target.type == "sameAsPrevious":
            return simulator.clip(previous_clip_id) if previous_clip_id else None
        if target.type == "currentClipAtPlayhead":
            return simulator.clip_at(context.playheadTimeUs, context.selectedTrackId) if context.playheadTimeUs is not None else None
        if target.type == "ordinal":
            track_id = self._resolve_track_ref(target.track, context) or context.selectedTrackId
            if track_id is None:
                return None
            ordered = simulator.ordered_clip_ids(track_id)
            index = {"first": 0, "second": 1, "third": 2}.get((target.value or "").lower())
            if (target.value or "").lower() == "last":
                index = len(ordered) - 1
            if index is None or index < 0 or index >= len(ordered):
                return None
            return simulator.clip(ordered[index])
        return None

    def _resolve_track_id(
        self,
        target: SemanticEditTarget | None,
        context: IntentCompilerContext,
        previous_track_id: str | None,
    ) -> str | None:
        if target is None:
            return context.selectedTrackId or previous_track_id
        if not isinstance(target, SemanticTrackReference):
            return None
        return self._resolve_track_ref(target, context) or previous_track_id

    @staticmethod
    def _resolve_track_ref(reference: SemanticTrackReference | None, context: IntentCompilerContext) -> str | None:
        if reference is None:
            return None
        if reference.type == "selectedTrack":
            return context.selectedTrackId
        return reference.trackId

    def _make_action(self, operation: ResolvedOperation, context: IntentCompilerContext) -> Action | None:
        created_at = _swift_reference_date_seconds()
        if operation.type == IntentEditType.splitClip and operation.targetClipId:
            return Action(
                action_id=str(uuid4()),
                timeline_id=context.timelineId,
                created_at=created_at,
                type=ActionType.splitClip,
                payload={"splitClip": {"clipId": operation.targetClipId, "atTimeUs": operation.parameters["atTimeUs"]}},
            )
        if operation.type == IntentEditType.removeClip and operation.targetClipId:
            return Action(
                action_id=str(uuid4()),
                timeline_id=context.timelineId,
                created_at=created_at,
                type=ActionType.removeClip,
                payload={"removeClip": {"clipId": operation.targetClipId}},
            )
        if operation.type == IntentEditType.trimClip and operation.targetClipId:
            return Action(
                action_id=str(uuid4()),
                timeline_id=context.timelineId,
                created_at=created_at,
                type=ActionType.trimClip,
                payload={
                    "trimClip": {
                        "clipId": operation.targetClipId,
                        "sourceRange": operation.parameters["sourceRange"],
                    }
                },
            )
        if operation.type == IntentEditType.removeClipRanges and operation.targetClipId:
            return Action(
                action_id=str(uuid4()),
                timeline_id=context.timelineId,
                created_at=created_at,
                type=ActionType.removeClipRanges,
                payload={
                    "removeClipRanges": {
                        "clipId": operation.targetClipId,
                        "sourceRanges": operation.parameters["sourceRanges"],
                    }
                },
            )
        if operation.type == IntentEditType.moveClip and operation.targetClipId:
            return Action(
                action_id=str(uuid4()),
                timeline_id=context.timelineId,
                created_at=created_at,
                type=ActionType.moveClip,
                payload={
                    "moveClip": {
                        "clipId": operation.targetClipId,
                        "orderedClipIds": operation.parameters["orderedClipIds"],
                    }
                },
            )
        if operation.type == IntentEditType.replaceTrackClips and operation.targetTrackId:
            return Action(
                action_id=str(uuid4()),
                timeline_id=context.timelineId,
                created_at=created_at,
                type=ActionType.replaceTrackClips,
                payload={
                    "replaceTrackClips": {
                        "trackId": operation.targetTrackId,
                        "clips": operation.parameters["clips"],
                    }
                },
            )
        return None

    def _validated_result(self, result: IntentCompileResult, context: IntentCompilerContext) -> IntentCompileResult:
        warnings = list(result.warnings)
        valid_actions: list[Action] = []
        for action in result.actions:
            action_warnings = self._validate_action(action, context)
            if action_warnings:
                warnings.extend(action_warnings)
            else:
                valid_actions.append(action)
        return IntentCompileResult(
            actions=valid_actions,
            confidence=result.confidence if valid_actions else 0,
            source=result.source,
            unresolvedText=result.unresolvedText if valid_actions else (result.unresolvedText or "Invalid timeline action"),
            warnings=self._unique_warnings(warnings),
            needsClarification=result.needsClarification or not valid_actions,
            experimentalEffectOperations=result.experimentalEffectOperations,
        )

    def _validate_action(self, action: Action, context: IntentCompilerContext) -> list[IntentCompileWarning]:
        if action.timelineId != context.timelineId:
            return [IntentCompileWarning.unsupportedAction]
        payload = action.payload
        if "splitClip" in payload:
            body = payload["splitClip"]
            clip = context.clip(body.get("clipId"))
            at_time_us = body.get("atTimeUs")
            if clip is None:
                return [IntentCompileWarning.clipNotFound]
            if at_time_us <= clip.timelineRange.start or at_time_us >= clip.timelineRange.end:
                return [IntentCompileWarning.splitTimeOutsideClip]
        if "removeClip" in payload and context.clip(payload["removeClip"].get("clipId")) is None:
            return [IntentCompileWarning.clipNotFound]
        if "trimClip" in payload:
            body = payload["trimClip"]
            if context.clip(body.get("clipId")) is None:
                return [IntentCompileWarning.clipNotFound]
            source = TimeRange.model_validate(body.get("sourceRange"))
            if source.duration <= 0:
                return [IntentCompileWarning.invalidTrimRange]
        if "removeClipRanges" in payload:
            body = payload["removeClipRanges"]
            clip = context.clip(body.get("clipId"))
            if clip is None:
                return [IntentCompileWarning.clipNotFound]
            source_ranges = self._source_ranges(body.get("sourceRanges"))
            if _invalid_remove_ranges(source_ranges, clip.sourceRange):
                return [IntentCompileWarning.invalidRemoveRange]
        if "moveClip" in payload:
            body = payload["moveClip"]
            clip = context.clip(body.get("clipId"))
            if clip is None:
                return [IntentCompileWarning.clipNotFound]
            existing_order = context.orderedClipIdsByTrackId.get(clip.trackId, [])
            ordered = body.get("orderedClipIds") or []
            if set(existing_order) != set(ordered) or len(existing_order) != len(ordered) or len(set(ordered)) != len(ordered):
                return [IntentCompileWarning.invalidMoveOrder]
        return []

    def _success(
        self,
        operation: SemanticEditOperation,
        *,
        clip_id: str | None = None,
        track_id: str | None = None,
        parameters: dict[str, Any],
    ) -> Resolution:
        return Resolution(
            status="success",
            operation=ResolvedOperation(
                type=operation.type,
                sourceText=operation.sourceText,
                targetClipId=clip_id,
                targetTrackId=track_id,
                confidence=operation.confidence or 0,
                parameters=parameters,
            ),
        )

    @staticmethod
    def _needs(warning: IntentCompileWarning, question: str) -> Resolution:
        return Resolution(status="clarification", warning=warning, question=question)

    @staticmethod
    def _clarification(
        original_prompt: str,
        question: str | None,
        warnings: list[IntentCompileWarning],
        plan: SemanticEditPlan,
    ) -> IntentCompileResult:
        return IntentCompileResult(
            actions=[],
            confidence=0,
            source=CompileSource.llm,
            unresolvedText=question or original_prompt,
            warnings=warnings,
            needsClarification=True,
            experimentalEffectOperations=plan.experimentalEffectOperations,
        )

    @staticmethod
    def _unique_warnings(warnings: list[IntentCompileWarning]) -> list[IntentCompileWarning]:
        seen: set[IntentCompileWarning] = set()
        unique: list[IntentCompileWarning] = []
        for warning in warnings:
            if warning not in seen:
                seen.add(warning)
                unique.append(warning)
        return unique

    def _duration_expression(self, value: Any) -> DurationExpression | None:
        if not isinstance(value, dict):
            return None
        raw_value = _float_value(value.get("value"))
        raw_unit = value.get("unit")
        unit = self._unit(str(raw_unit)) if raw_unit is not None else None
        if raw_value is not None and unit is not None:
            return DurationExpression(kind="duration", value=raw_value, unit=unit)
        match value.get("type"):
            case "duration":
                return DurationExpression(kind="duration", value=raw_value, unit=unit) if raw_value is not None and unit else None
            case "percentage":
                return DurationExpression(kind="percentage", value=raw_value) if raw_value is not None else None
            case "vague":
                return DurationExpression(kind="vague", phrase=str(value.get("phrase") or ""))
            case _:
                return None

    def _time_expression(self, value: Any) -> TimeExpression | None:
        if not isinstance(value, dict):
            return None
        match value.get("type"):
            case "playhead":
                return TimeExpression(kind="playhead")
            case "absoluteTimelineTime":
                raw_value = _float_value(value.get("value"))
                raw_unit = value.get("unit")
                unit = self._unit(str(raw_unit)) if raw_unit is not None else None
                return TimeExpression(kind="absoluteTimelineTime", value=raw_value, unit=unit) if raw_value is not None and unit else None
            case "fractionOfClip":
                raw_value = _float_value(value.get("value"))
                return TimeExpression(kind="fractionOfClip", value=raw_value) if raw_value is not None else None
            case "afterStart" | "beforeEnd" as kind:
                amount = self._duration_expression(value.get("amount"))
                return TimeExpression(kind=kind, amount=amount) if amount else None
            case _:
                return None

    def _source_ranges(self, value: Any) -> list[TimeRange]:
        if not isinstance(value, list):
            return []
        ranges: list[TimeRange] = []
        for item in value:
            try:
                ranges.append(TimeRange.model_validate(item))
            except Exception:
                continue
        return ranges

    def _resolve_duration_us(self, expression: DurationExpression, clip: Clip, source_text: str | None) -> int | None:
        if expression.kind == "duration" and expression.value is not None and expression.unit is not None:
            explicit = self._explicit_duration_us(source_text, expression.value)
            return explicit if explicit is not None else self._microseconds(expression.value, expression.unit)
        if expression.kind == "percentage" and expression.value is not None:
            return round(clip.timelineRange.duration * expression.value / 100)
        if expression.kind == "vague":
            return self._explicit_duration_us(source_text)
        return None

    def _resolve_time_us(
        self,
        expression: TimeExpression,
        clip: Clip,
        context: IntentCompilerContext,
        source_text: str | None,
    ) -> int | None:
        if expression.kind == "playhead":
            return context.playheadTimeUs
        if expression.kind == "absoluteTimelineTime" and expression.value is not None and expression.unit is not None:
            explicit = self._explicit_duration_us(source_text, expression.value)
            return explicit if explicit is not None else self._microseconds(expression.value, expression.unit)
        if expression.kind == "fractionOfClip" and expression.value is not None:
            return clip.timelineRange.start + round(clip.timelineRange.duration * expression.value)
        if expression.kind == "afterStart" and expression.amount:
            duration = self._resolve_duration_us(expression.amount, clip, source_text)
            return clip.timelineRange.start + duration if duration is not None else None
        if expression.kind == "beforeEnd" and expression.amount:
            duration = self._resolve_duration_us(expression.amount, clip, source_text)
            return clip.timelineRange.end - duration if duration is not None else None
        return None

    def _explicit_duration_us(self, text: str | None, expected_value: float | None = None) -> int | None:
        if not text:
            return None
        pattern = re.compile(
            r"\b(\d+(?:\.\d+)?|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s*"
            r"(microseconds?|usec|us|milliseconds?|msec|ms|seconds?|secs?|sec|s|minutes?|mins?|min|m)\b",
            re.IGNORECASE,
        )
        for match in pattern.finditer(text):
            numeric_value = _float_value(match.group(1))
            value = numeric_value if numeric_value is not None else _spoken_number_value(match.group(1))
            if value is None:
                continue
            if expected_value is not None and abs(value - expected_value) > 0.000_001:
                continue
            unit = self._unit(match.group(2))
            if unit:
                return self._microseconds(value, unit)
        return None

    @staticmethod
    def _infer_trim_edge(source_text: str | None) -> str:
        text = (source_text or "").lower()
        if re.search(r"\b(first|start|beginning|front|opening)\b", text):
            return "start"
        if re.search(r"\b(last|end|ending|tail|final)\b", text):
            return "end"
        return ""

    @staticmethod
    def _unit(raw: str) -> DurationUnit | None:
        match raw.lower():
            case "microsecond" | "microseconds" | "usec" | "us":
                return "microsecond"
            case "millisecond" | "milliseconds" | "msec" | "ms":
                return "millisecond"
            case "second" | "seconds" | "sec" | "secs" | "s":
                return "second"
            case "minute" | "minutes" | "min" | "mins" | "m":
                return "minute"
            case _:
                return None

    @staticmethod
    def _microseconds(value: float, unit: DurationUnit) -> int:
        match unit:
            case "microsecond":
                return round(value)
            case "millisecond":
                return round(value * 1_000)
            case "second":
                return round(value * 1_000_000)
            case "minute":
                return round(value * 60_000_000)


def _float_value(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalized_remove_ranges(source_ranges: list[TimeRange], clip_source_range: TimeRange) -> list[TimeRange]:
    bounded: list[TimeRange] = []
    for source_range in source_ranges:
        start = max(clip_source_range.start, source_range.start)
        end = min(clip_source_range.end, source_range.end)
        if end > start:
            bounded.append(TimeRange(start=start, end=end))

    bounded.sort(key=lambda item: (item.start, item.end))
    merged: list[TimeRange] = []
    for source_range in bounded:
        if not merged or source_range.start > merged[-1].end:
            merged.append(source_range)
        else:
            merged[-1].end = max(merged[-1].end, source_range.end)
    return merged


def _survivor_ranges(clip_source_range: TimeRange, removal_ranges: list[TimeRange]) -> list[TimeRange]:
    survivors: list[TimeRange] = []
    cursor = clip_source_range.start
    for removal_range in removal_ranges:
        if removal_range.start > cursor:
            survivors.append(TimeRange(start=cursor, end=removal_range.start))
        cursor = max(cursor, removal_range.end)
    if cursor < clip_source_range.end:
        survivors.append(TimeRange(start=cursor, end=clip_source_range.end))
    return [source_range for source_range in survivors if source_range.duration > 0]


def _invalid_remove_ranges(source_ranges: list[TimeRange], clip_source_range: TimeRange) -> bool:
    if not source_ranges:
        return True
    previous_end: int | None = None
    for source_range in sorted(source_ranges, key=lambda item: (item.start, item.end)):
        if source_range.duration <= 0:
            return True
        if source_range.start < clip_source_range.start or source_range.end > clip_source_range.end:
            return True
        if previous_end is not None and source_range.start < previous_end:
            return True
        previous_end = source_range.end
    return False


def _dead_space_terms(text: str | None) -> bool:
    return bool(re.search(r"\b(dead\s*space|silence|silent|pause|pauses|gap|gaps)\b", text or "", re.IGNORECASE))


def _spoken_number_value(value: str) -> float | None:
    numbers = {
        "a": 1,
        "an": 1,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
    }
    return numbers.get(value.lower())


def _swift_reference_date_seconds() -> float:
    reference = datetime(2001, 1, 1, tzinfo=UTC)
    return (datetime.now(tz=UTC) - reference).total_seconds()

