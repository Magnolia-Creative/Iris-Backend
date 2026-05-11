from __future__ import annotations

from collections.abc import Awaitable, Callable
import json
from typing import Any

from langchain_openai import ChatOpenAI

from app.config import settings
from app.services.intent_compiler.capabilities import DEFAULT_EFFECT_CAPABILITIES
from app.services.intent_compiler.compiler import IntentCompiler
from app.services.intent_compiler.models import (
    EffectCapability,
    ExperimentalEffectOperation,
    ExperimentalEffectPlan,
    IntentCompileResult,
    IntentCompileWarning,
    IntentCompilerContext,
    RelevantEffectCapability,
    SemanticEffectRequest,
    SemanticEditPlan,
)


IntentEventHandler = Callable[[dict[str, Any]], Awaitable[None]]


def _get_llm() -> Any:
    return ChatOpenAI(
        api_key=settings.openai_api_key,
        model=settings.intent_openai_model,
        temperature=0,
    )


class IntentLLMCompiler:
    def __init__(self, llm: Any | None = None) -> None:
        self.llm = llm or _get_llm()

    async def make_semantic_plan(self, prompt: str, context: IntentCompilerContext) -> SemanticEditPlan:
        llm = self.llm.with_structured_output(SemanticEditPlan)
        return await llm.ainvoke(
            [
                (
                    "system",
                    "You are a strict JSON planner for a video timeline intent compiler. "
                    "Return only values matching the provided structured schema. Never copy schema "
                    "placeholder strings into output. Never invent clip ids, track ids, time ranges, "
                    "or operation names.",
                ),
                ("human", self._semantic_prompt(prompt, context)),
            ]
        )

    async def plan_experimental_effects(
        self,
        *,
        original_prompt: str,
        effect_request: SemanticEffectRequest,
        relevant_capabilities: list[RelevantEffectCapability],
        context: IntentCompilerContext,
    ) -> ExperimentalEffectPlan:
        llm = self.llm.with_structured_output(ExperimentalEffectPlan)
        return await llm.ainvoke(
            [
                (
                    "system",
                    "You convert one abstract visual style request into conservative effect operations. "
                    "Use only retrieved capability operation names and parameter schemas.",
                ),
                (
                    "human",
                    self._effect_prompt(
                        original_prompt=original_prompt,
                        effect_request=effect_request,
                        relevant_capabilities=relevant_capabilities,
                        context=context,
                    ),
                ),
            ]
        )

    def _semantic_prompt(self, prompt: str, context: IntentCompilerContext) -> str:
        return (
            "Parse one user prompt into ordered concrete timeline operations and/or abstract visual "
            "effect requests.\n"
            "Rules:\n"
            "- Style-only prompts must produce operations=[] and effectRequests with the style intent.\n"
            "- Concrete timeline edits use only splitClip, removeClip, trimClip, moveClip, replaceTrackClips, unknown.\n"
            "- Abstract visual look, mood, color grade, texture, vintage, cinematic, warmer, colder, faded, "
            "dreamy, grainy, or moody requests go in effectRequests, not operations.\n"
            "- Captions, audio, transitions, generative media, or unsupported non-visual effects should be unknown.\n"
            "- Use target objects only, never plain target strings. Use only ids in ctx.\n"
            "- If required target/time/order is missing, set needsClarification=true and ask a short question.\n"
            "- Preserve explicit units from user text, e.g. '2 seconds' means unit=second.\n"
            "Target shapes:\n"
            '{"type":"selectedClip"}, {"type":"clipId","clipId":"existing"}, '
            '{"type":"sameAsPrevious"}, {"type":"ordinal","value":"first|second|third|last","track":{"type":"selectedTrack"}}, '
            '{"type":"currentClipAtPlayhead"}, {"type":"selectedTrack"}, {"type":"trackId","trackId":"existing"}.\n'
            "Operation parameter shapes:\n"
            "- splitClip: parameters.position is playhead, absoluteTimelineTime, fractionOfClip, afterStart, or beforeEnd.\n"
            "- trimClip: parameters={\"edge\":\"start|end\",\"amount\":duration|percentage|vague}.\n"
            "- moveClip: parameters.placement is beginning|start|first|end|last, or orderedClipIds.\n"
            "- replaceTrackClips: parameters.orderedClipIds.\n"
            f"ctx={_json(_editor_context(context))}\n"
            f"user={prompt}"
        )

    def _effect_prompt(
        self,
        *,
        original_prompt: str,
        effect_request: SemanticEffectRequest,
        relevant_capabilities: list[RelevantEffectCapability],
        context: IntentCompilerContext,
    ) -> str:
        return (
            "Convert this one visual effect request into experimental effect operations.\n"
            "Rules:\n"
            "- Use only operation names from retrievedCapabilities.\n"
            "- Do not invent operations or parameters.\n"
            "- Preserve the request target exactly unless null.\n"
            "- Prefer a small combination of complementary effects over many weak effects.\n"
            "- Use conservative numeric values unless the prompt asks for an extreme look.\n"
            "- Parameters must stay within min/max ranges.\n"
            "- If no capability is useful, return operations=[] with rationale.\n"
            f"originalUserPrompt={original_prompt}\n"
            f"effectRequest={effect_request.model_dump_json()}\n"
            f"retrievedCapabilities={_json([cap.model_dump() for cap in relevant_capabilities])}\n"
            f"ctx={_json(_editor_context(context))}"
        )


class IntentCompilerService:
    def __init__(
        self,
        llm_compiler: IntentLLMCompiler | None = None,
        compiler: IntentCompiler | None = None,
        capabilities: list[EffectCapability] | None = None,
    ) -> None:
        self.llm_compiler = llm_compiler or IntentLLMCompiler()
        self.compiler = compiler or IntentCompiler()
        self.capabilities = capabilities or DEFAULT_EFFECT_CAPABILITIES

    async def compile_prompt(
        self,
        *,
        prompt: str,
        context: IntentCompilerContext,
        event_handler: IntentEventHandler | None = None,
    ) -> IntentCompileResult:
        await _emit(event_handler, {"type": "planner_started", "status": "Parsing prompt."})
        semantic_plan = await self.llm_compiler.make_semantic_plan(prompt, context)
        await _emit(
            event_handler,
            {
                "type": "planner_completed",
                "operations": len(semantic_plan.operations),
                "effect_requests": len(semantic_plan.effectRequests),
            },
        )

        effect_operations: list[ExperimentalEffectOperation] = []
        warnings: list[IntentCompileWarning] = []
        for effect_request in semantic_plan.effectRequests:
            relevant = self._relevant_capabilities(effect_request)
            if not relevant:
                warnings.append(IntentCompileWarning.unsupportedIntent)
                continue
            await _emit(
                event_handler,
                {
                    "type": "effect_planner_started",
                    "intent": effect_request.intent,
                    "capabilities": [cap.capability.operation for cap in relevant],
                },
            )
            effect_plan = await self.llm_compiler.plan_experimental_effects(
                original_prompt=prompt,
                effect_request=effect_request,
                relevant_capabilities=relevant,
                context=context,
            )
            validated = self._validate_effect_operations(effect_plan.operations, relevant)
            effect_operations.extend(validated[0])
            warnings.extend(validated[1])
            await _emit(
                event_handler,
                {
                    "type": "effect_planner_completed",
                    "intent": effect_request.intent,
                    "operations": [operation.model_dump() for operation in validated[0]],
                },
            )

        merged_plan = SemanticEditPlan(
            operations=semantic_plan.operations,
            effectRequests=semantic_plan.effectRequests,
            experimentalEffectOperations=effect_operations,
            needsClarification=semantic_plan.needsClarification,
            clarificationQuestion=semantic_plan.clarificationQuestion,
        )
        result = self.compiler.compile(merged_plan, original_prompt=prompt, context=context)
        result = _with_additional_warnings(result, warnings)
        await _emit(
            event_handler,
            {
                "type": "validation_completed",
                "actions": len(result.actions),
                "experimental_effects": len(result.experimentalEffectOperations),
                "warnings": [warning.value for warning in result.warnings],
            },
        )
        return result

    def _relevant_capabilities(self, effect_request: SemanticEffectRequest) -> list[RelevantEffectCapability]:
        query_parts = [effect_request.sourceText, effect_request.intent or "", *effect_request.attributes]
        query = " ".join(query_parts).lower()
        scored: list[RelevantEffectCapability] = []
        for capability in self.capabilities:
            haystack = " ".join(
                [capability.operation, capability.description, capability.retrievalText, *capability.examples]
            ).lower()
            score = _keyword_score(query, haystack)
            if score > 0:
                scored.append(RelevantEffectCapability(capability=capability, score=score))
        if not scored and query.strip():
            scored = [RelevantEffectCapability(capability=capability, score=0.01) for capability in self.capabilities]
        return sorted(scored, key=lambda item: item.score, reverse=True)[:5]

    @staticmethod
    def _validate_effect_operations(
        operations: list[ExperimentalEffectOperation],
        relevant_capabilities: list[RelevantEffectCapability],
    ) -> tuple[list[ExperimentalEffectOperation], list[IntentCompileWarning]]:
        capabilities_by_operation = {
            relevant.capability.operation: relevant.capability for relevant in relevant_capabilities
        }
        valid: list[ExperimentalEffectOperation] = []
        warnings: list[IntentCompileWarning] = []
        for operation in operations:
            capability = capabilities_by_operation.get(operation.operation)
            if capability is None:
                warnings.append(IntentCompileWarning.unsupportedAction)
                continue
            params = dict(operation.parameters)
            for parameter in capability.parameters:
                if parameter.valueType != "number" or parameter.name not in params:
                    continue
                try:
                    value = float(params[parameter.name])
                except (TypeError, ValueError):
                    continue
                minimum = parameter.minimum if parameter.minimum is not None else value
                maximum = parameter.maximum if parameter.maximum is not None else value
                params[parameter.name] = min(max(value, minimum), maximum)
            valid.append(operation.model_copy(update={"parameters": params}))
        return valid, warnings


async def _emit(event_handler: IntentEventHandler | None, payload: dict[str, Any]) -> None:
    if event_handler is not None:
        await event_handler(payload)


def _with_additional_warnings(
    result: IntentCompileResult,
    additional_warnings: list[IntentCompileWarning],
) -> IntentCompileResult:
    if not additional_warnings:
        return result
    seen: set[IntentCompileWarning] = set()
    warnings = [warning for warning in [*result.warnings, *additional_warnings] if not (warning in seen or seen.add(warning))]
    return result.model_copy(update={"warnings": warnings})


def _keyword_score(query: str, haystack: str) -> float:
    terms = {term for term in re_split_words(query) if len(term) > 2}
    if not terms:
        return 0
    matches = sum(1 for term in terms if term in haystack)
    return matches / len(terms)


def re_split_words(text: str) -> list[str]:
    return [part for part in "".join(char if char.isalnum() else " " for char in text.lower()).split() if part]


def _editor_context(context: IntentCompilerContext) -> dict[str, Any]:
    current_clip_id = _current_clip_at_playhead_id(context)
    current_clip = context.clip(context.selectedClipId) or context.clip(current_clip_id)
    return {
        "selectedClipId": context.selectedClipId,
        "selectedTrackId": context.selectedTrackId,
        "playheadTimeUs": context.playheadTimeUs,
        "currentClipAtPlayheadId": current_clip_id,
        "currentClip": (
            {
                "clipId": current_clip.clipId,
                "durationUs": current_clip.timelineRange.duration,
                "trackId": current_clip.trackId,
            }
            if current_clip
            else None
        ),
        "orderedClipIdsByTrackId": context.orderedClipIdsByTrackId,
    }


def _current_clip_at_playhead_id(context: IntentCompilerContext) -> str | None:
    if context.playheadTimeUs is None:
        return None
    track_ids = []
    if context.selectedTrackId:
        track_ids.append(context.selectedTrackId)
    track_ids.extend(track_id for track_id in sorted(context.orderedClipIdsByTrackId) if track_id not in track_ids)
    for track_id in track_ids:
        for clip_id in context.orderedClipIdsByTrackId.get(track_id, []):
            clip = context.clip(clip_id)
            if clip and clip.timelineRange.start <= context.playheadTimeUs < clip.timelineRange.end:
                return clip.clipId
    return None


def _json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)

