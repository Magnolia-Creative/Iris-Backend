from __future__ import annotations

from collections.abc import Awaitable, Callable
import json
import logging
from math import sqrt
import re
from typing import Any, Protocol

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.config import settings
from app.intent_compiler.capabilities import DEFAULT_EFFECT_CAPABILITIES
from app.intent_compiler.compiler import IntentCompiler
from app.intent_compiler.models import (
    EffectCapability,
    EffectCapabilityParameter,
    ExperimentalEffectOperation,
    ExperimentalEffectPlan,
    IntentCompileResult,
    IntentCompileWarning,
    IntentCompilerContext,
    IntentEditType,
    JSONValue,
    RelevantEffectCapability,
    SemanticClipReference,
    SemanticEffectRequest,
    SemanticEditOperation,
    SemanticEditPlan,
)


IntentEventHandler = Callable[[dict[str, Any]], Awaitable[None]]
EFFECT_EMBEDDING_MODEL = "text-embedding-3-small"

logger = logging.getLogger(__name__)


class EffectEmbeddingClient(Protocol):
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        ...


def _get_llm() -> Any:
    return ChatOpenAI(
        api_key=settings.openai_api_key,
        model=settings.intent_openai_model,
        temperature=0,
    )


def _get_effect_embeddings() -> EffectEmbeddingClient:
    return OpenAIEmbeddings(api_key=settings.openai_api_key, model=EFFECT_EMBEDDING_MODEL)


class IntentLLMCompiler:
    def __init__(self, llm: Any | None = None) -> None:
        self.llm = llm or _get_llm()

    async def make_semantic_plan(self, prompt: str, context: IntentCompilerContext) -> SemanticEditPlan:
        llm = self.llm.with_structured_output(SemanticEditPlan, method="function_calling")
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
        llm = self.llm.with_structured_output(ExperimentalEffectPlan, method="function_calling")
        return await llm.ainvoke(
            [
                (
                    "system",
                    "You convert one abstract visual style request into conservative effect operations. "
                    "Use only retrieved capability operation names and parameter schemas. "
                    "Every operation must include concrete parameter values, an intention, and notes explaining those values.",
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
            "- Concrete timeline edits use only splitClip, removeClip, trimClip, removeClipRanges, moveClip, "
            "replaceTrackClips, unknown.\n"
            "- Abstract visual look, mood, color grade, texture, vintage, cinematic, warmer, colder, faded, "
            "dreamy, grainy, or moody requests go in effectRequests, not operations.\n"
            "- Captions, audio, transitions, generative media, or unsupported non-visual effects should be unknown.\n"
            "- Use target objects only, never plain target strings. Use only ids in ctx.\n"
            "- If ctx.selectedClipId is non-null, treat targetless clip edits and phrases like "
            "'this clip', 'the clip', 'selected clip', 'current clip', and 'it' as {\"type\":\"selectedClip\"}.\n"
            "- Prioritize ctx.selectedClipId for clip edits unless the user explicitly names another clip. "
            "Phrases like 'first two seconds', 'first part', or 'beginning' describe the selected clip's trim edge, "
            "not the first clip on the timeline.\n"
            "- If no clip is selected but ctx.currentClipAtPlayheadId is non-null, use "
            "{\"type\":\"currentClipAtPlayhead\"} for targetless clip edits.\n"
            "- For 'split in half', 'split down the middle', or similar, use "
            "parameters.position={\"type\":\"fractionOfClip\",\"value\":0.5}.\n"
            "- Phrases like 'here', 'right here', 'split here', 'cut here', 'at the playhead', or 'this point' "
            "refer to the timeline playhead: use parameters.position={\"type\":\"playhead\"} whenever "
            "ctx.playheadTimeUs lies strictly between the target clip's timeline start and end.\n"
            "- If a split position is omitted and ctx.playheadTimeUs is inside the target clip, use "
            "parameters.position={\"type\":\"playhead\"}.\n"
            "- Always include parameters.position for splitClip; use lowercase type strings exactly: "
            "playhead, absoluteTimelineTime, fractionOfClip, afterStart, beforeEnd.\n"
            "- For requests to cut out dead space, silence, long pauses, or transcript gaps inside a clip, use "
            "removeClipRanges with sourceRanges copied from ctx.transcriptContext.pauseRanges. Do not use trimClip "
            "for an interior gap.\n"
            "- Only set needsClarification=true when required target/time/order cannot be inferred from ctx.\n"
            "- Preserve explicit units from user text, e.g. '2 seconds' means unit=second.\n"
            "Target shapes:\n"
            '{"type":"selectedClip"}, {"type":"clipId","clipId":"existing"}, '
            '{"type":"sameAsPrevious"}, {"type":"ordinal","value":"first|second|third|last","track":{"type":"selectedTrack"}}, '
            '{"type":"currentClipAtPlayhead"}, {"type":"selectedTrack"}, {"type":"trackId","trackId":"existing"}.\n'
            "Operation parameter shapes:\n"
            "- splitClip: parameters.position is playhead, absoluteTimelineTime, fractionOfClip, afterStart, or beforeEnd.\n"
            "- trimClip: parameters={\"edge\":\"start|end\",\"amount\":duration|percentage|vague}.\n"
            "- removeClipRanges: parameters.sourceRanges is an array of source-time ranges in microseconds, "
            "e.g. [{\"start\":1200000,\"end\":2200000}].\n"
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
            "- Every selected operation must include a concrete value for each capability parameter.\n"
            "- Every selected operation must include intention: a short lower-case imperative describing that "
            "operation's visual goal, such as \"make it warmer\" for positive setTemperature.\n"
            "- Add parameterNotes with one short note per parameter explaining what that chosen value does.\n"
            '- Example: intention="make it cooler", parameters={"value":-0.25}, '
            'parameterNotes={"value":"Makes temperature cooler."}.\n'
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
        embedding_client: EffectEmbeddingClient | None = None,
    ) -> None:
        self.llm_compiler = llm_compiler or IntentLLMCompiler()
        self.compiler = compiler or IntentCompiler()
        self.capabilities = capabilities or DEFAULT_EFFECT_CAPABILITIES
        self.embedding_client = embedding_client or _get_effect_embeddings()
        self._capability_embedding_cache: tuple[tuple[str, ...], list[list[float]]] | None = None

    async def compile_prompt(
        self,
        *,
        prompt: str,
        context: IntentCompilerContext,
        event_handler: IntentEventHandler | None = None,
    ) -> IntentCompileResult:
        _log_intent_compile_incoming_context(prompt=prompt, context=context)
        await _emit(event_handler, {"type": "planner_started", "status": "Parsing prompt."})
        semantic_plan = await self.llm_compiler.make_semantic_plan(prompt, context)
        semantic_plan = _apply_contextual_assumptions(prompt, context, semantic_plan)
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
            try:
                relevant = await self._relevant_capabilities(effect_request)
            except Exception:
                warnings.append(IntentCompileWarning.embeddingUnavailable)
                continue
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
            validated_operations = _assume_effect_operation_targets(validated[0], context)
            effect_operations.extend(validated_operations)
            warnings.extend(validated[1])
            await _emit(
                event_handler,
                {
                    "type": "effect_planner_completed",
                    "intent": effect_request.intent,
                    "operations": [operation.model_dump() for operation in validated_operations],
                },
            )

        merged_plan = SemanticEditPlan(
            operations=semantic_plan.operations,
            effectRequests=semantic_plan.effectRequests,
            experimentalEffectOperations=_assume_effect_operation_targets(effect_operations, context),
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

    async def _relevant_capabilities(
        self,
        effect_request: SemanticEffectRequest,
    ) -> list[RelevantEffectCapability]:
        query_parts = [effect_request.sourceText, effect_request.intent or "", *effect_request.attributes]
        query = " ".join(part for part in query_parts if part).strip()
        if not query or not self.capabilities:
            return []

        query_embedding = (await self.embedding_client.aembed_documents([query]))[0]
        capability_embeddings = await self._capability_embeddings()
        scored = [
            RelevantEffectCapability(
                capability=capability,
                score=_cosine_similarity(query_embedding, capability_embedding),
            )
            for capability, capability_embedding in zip(self.capabilities, capability_embeddings, strict=True)
        ]
        return sorted(scored, key=lambda item: item.score, reverse=True)[:5]

    async def _capability_embeddings(self) -> list[list[float]]:
        cache_key = tuple(_capability_embedding_text(capability) for capability in self.capabilities)
        if self._capability_embedding_cache and self._capability_embedding_cache[0] == cache_key:
            return self._capability_embedding_cache[1]

        embeddings = await self.embedding_client.aembed_documents(list(cache_key))
        self._capability_embedding_cache = (cache_key, embeddings)
        return embeddings

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
            parameter_notes = dict(operation.parameterNotes)
            for parameter in capability.parameters:
                if parameter.valueType != "number":
                    continue
                if parameter.name not in params:
                    inferred = _inferred_effect_parameter_value(operation, parameter)
                    if inferred is None:
                        continue
                    params[parameter.name] = inferred
                try:
                    value = float(params[parameter.name])
                except (TypeError, ValueError):
                    inferred = _inferred_effect_parameter_value(operation, parameter)
                    if inferred is None:
                        continue
                    value = inferred
                minimum = parameter.minimum if parameter.minimum is not None else value
                maximum = parameter.maximum if parameter.maximum is not None else value
                params[parameter.name] = min(max(value, minimum), maximum)
                parameter_notes[parameter.name] = _effect_parameter_note(
                    operation.operation,
                    parameter.name,
                    float(params[parameter.name]),
                    parameter.description,
                )
            valid.append(
                operation.model_copy(
                    update={
                        "intention": _effect_operation_intention(operation, params),
                        "parameters": params,
                        "parameterNotes": parameter_notes,
                    }
                )
            )
        return valid, warnings


_CLIP_EDIT_TYPES = {
    IntentEditType.splitClip,
    IntentEditType.removeClip,
    IntentEditType.trimClip,
    IntentEditType.removeClipRanges,
    IntentEditType.moveClip,
}

_SPLIT_TERMS = re.compile(r"\b(split|cut|slice)\b", re.IGNORECASE)
_HALF_TERMS = re.compile(r"\b(half|middle|midpoint|center|centre)\b", re.IGNORECASE)
_REMOVE_TERMS = re.compile(r"\b(delete|remove)\b", re.IGNORECASE)
_SELECTED_REFERENCE_TERMS = re.compile(r"\b(this|that|the|selected|current|clip|it)\b", re.IGNORECASE)
_BROAD_TARGET_TERMS = re.compile(r"\b(all|every|everything|entire timeline)\b", re.IGNORECASE)
_EXPLICIT_NON_SELECTED_CLIP_TERMS = re.compile(
    r"\b(first|second|third|last)\s+(clip|video|shot)\b"
    r"|\bclip[\s-]+[a-z0-9]+\b"
    r"|\b(at|under)\s+(the\s+)?playhead\b",
    re.IGNORECASE,
)


def _apply_contextual_assumptions(
    prompt: str,
    context: IntentCompilerContext,
    plan: SemanticEditPlan,
) -> SemanticEditPlan:
    target = _preferred_clip_target(context)
    if target is None:
        return plan

    fallback = _fallback_selected_clip_plan(prompt, target)
    if fallback and plan.needsClarification and not (plan.operations or plan.effectRequests or plan.experimentalEffectOperations):
        return fallback

    operations = [_assume_operation_target(operation, target) for operation in plan.operations]
    effect_requests = [
        request.model_copy(update={"target": target, "confidence": max(float(request.confidence), 0.85)})
        if request.target is None
        else request
        for request in plan.effectRequests
    ]
    effect_operations = _assume_effect_operation_targets(plan.experimentalEffectOperations, context)

    needs_clarification = plan.needsClarification
    clarification_question = plan.clarificationQuestion
    if plan.needsClarification and (
        any(operation.type in _CLIP_EDIT_TYPES for operation in operations) or effect_requests or effect_operations
    ):
        needs_clarification = False
        clarification_question = None

    return plan.model_copy(
        update={
            "operations": operations,
            "effectRequests": effect_requests,
            "experimentalEffectOperations": effect_operations,
            "needsClarification": needs_clarification,
            "clarificationQuestion": clarification_question,
        }
    )


def _assume_operation_target(
    operation: SemanticEditOperation,
    target: SemanticClipReference,
) -> SemanticEditOperation:
    if operation.type not in _CLIP_EDIT_TYPES:
        return operation
    if operation.target is not None:
        if _should_prefer_selected_clip_target(operation):
            return operation.model_copy(
                update={
                    "target": target,
                    "confidence": max(float(operation.confidence or 0), 0.85),
                }
            )
        return operation
    return operation.model_copy(
        update={
            "target": target,
            "confidence": max(float(operation.confidence or 0), 0.85),
        }
    )


def _should_prefer_selected_clip_target(operation: SemanticEditOperation) -> bool:
    if not isinstance(operation.target, SemanticClipReference):
        return False
    if operation.target.type == "selectedClip":
        return False
    return _EXPLICIT_NON_SELECTED_CLIP_TERMS.search(operation.sourceText) is None


def _assume_effect_operation_targets(
    operations: list[ExperimentalEffectOperation],
    context: IntentCompilerContext,
) -> list[ExperimentalEffectOperation]:
    target = _preferred_clip_target(context)
    if target is None:
        return operations
    return [
        operation.model_copy(
            update={
                "target": target,
                "confidence": max(float(operation.confidence), 0.85),
            }
        )
        if operation.target is None
        else operation
        for operation in operations
    ]


def _preferred_clip_target(context: IntentCompilerContext) -> SemanticClipReference | None:
    if context.selectedClipId and context.clip(context.selectedClipId):
        return SemanticClipReference(type="selectedClip")
    if _current_clip_at_playhead_id(context):
        return SemanticClipReference(type="currentClipAtPlayhead")
    return None


def _fallback_selected_clip_plan(prompt: str, target: SemanticClipReference) -> SemanticEditPlan | None:
    if _BROAD_TARGET_TERMS.search(prompt):
        return None
    if _SPLIT_TERMS.search(prompt):
        position = {"type": "fractionOfClip", "value": 0.5} if _HALF_TERMS.search(prompt) else {"type": "playhead"}
        return SemanticEditPlan(
            operations=[
                SemanticEditOperation(
                    type=IntentEditType.splitClip,
                    sourceText=prompt,
                    target=target,
                    parameters={"position": position},
                    confidence=0.9,
                )
            ],
            needsClarification=False,
        )
    if _REMOVE_TERMS.search(prompt) and _SELECTED_REFERENCE_TERMS.search(prompt):
        return SemanticEditPlan(
            operations=[
                SemanticEditOperation(
                    type=IntentEditType.removeClip,
                    sourceText=prompt,
                    target=target,
                    parameters={},
                    confidence=0.9,
                )
            ],
            needsClarification=False,
        )
    return None


def _inferred_effect_parameter_value(
    operation: ExperimentalEffectOperation,
    parameter: EffectCapabilityParameter,
) -> float | None:
    minimum = parameter.minimum if parameter.minimum is not None else -1
    maximum = parameter.maximum if parameter.maximum is not None else 1
    source = operation.sourceText.lower()
    intensity = _effect_intensity(operation.operation, source)

    if minimum >= 0:
        return min(max(intensity, minimum), maximum)

    direction = _effect_direction(operation.operation, source)
    if direction == 0:
        direction = _default_effect_direction(operation.operation)
    return min(max(direction * intensity, minimum), maximum)


def _effect_intensity(operation: str, source: str) -> float:
    base_by_operation = {
        "addGrain": 0.18,
        "setTemperature": 0.4,
        "setSaturation": 0.18,
        "setContrast": 0.28,
        "setExposure": 0.16,
        "setHighlights": 0.2,
        "setShadows": 0.22,
    }
    value = base_by_operation.get(operation, 0.25)

    if "cinematic" in source:
        cinematic_values = {
            "addGrain": 0.12,
            "setTemperature": 0.35,
            "setSaturation": 0.12,
            "setContrast": 0.32,
            "setExposure": 0.1,
            "setHighlights": 0.18,
            "setShadows": 0.28,
        }
        value = cinematic_values.get(operation, value)
    if any(term in source for term in ["vintage", "film", "analog", "2016", "la vibe"]):
        vintage_values = {
            "addGrain": 0.22,
            "setTemperature": 0.3,
            "setSaturation": 0.16,
            "setContrast": 0.22,
            "setExposure": 0.08,
            "setHighlights": 0.16,
            "setShadows": 0.2,
        }
        value = vintage_values.get(operation, value)
    if any(term in source for term in ["extreme", "super", "very", "really", "heavy", "strong", "intense"]):
        return min(value * 1.75, 0.8)
    if any(term in source for term in ["slight", "subtle", "little", "soft", "gentle"]):
        return min(value, 0.18)
    return value


def _effect_direction(operation: str, source: str) -> int:
    negative_terms_by_operation = {
        "setTemperature": ["cool", "cold", "blue", "icy"],
        "setSaturation": ["desaturat", "faded", "muted", "washed", "bleached"],
        "setContrast": ["soft", "flat", "faded", "matte", "less harsh"],
        "setExposure": ["dark", "dim", "underexposed", "moody"],
        "setHighlights": ["recover", "blown", "soften"],
        "setShadows": ["deepen", "crush", "dark"],
    }
    positive_terms_by_operation = {
        "setTemperature": ["warm", "golden", "sunset", "orange", "la", "vintage"],
        "setSaturation": ["saturat", "vibrant", "pop", "colorful", "rich"],
        "setContrast": ["contrast", "punchy", "dramatic", "cinematic", "moody"],
        "setExposure": ["bright", "airy", "light", "overexposed"],
        "setHighlights": ["glow", "lift", "bright"],
        "setShadows": ["lift", "faded", "matte"],
    }
    if any(term in source for term in negative_terms_by_operation.get(operation, [])):
        return -1
    if any(term in source for term in positive_terms_by_operation.get(operation, [])):
        return 1
    return 0


def _default_effect_direction(operation: str) -> int:
    defaults = {
        "setTemperature": 1,
        "setSaturation": 1,
        "setContrast": 1,
        "setExposure": 1,
        "setHighlights": 1,
        "setShadows": 1,
    }
    return defaults.get(operation, 1)


def _effect_parameter_note(
    operation: str,
    parameter_name: str,
    value: float,
    description: str,
) -> str:
    if operation == "addGrain" and parameter_name == "amount":
        return "Adds visible film grain." if value > 0 else "Leaves film grain unchanged."
    if operation == "setTemperature" and parameter_name == "value":
        return _signed_note(value, "Makes temperature cooler.", "Makes temperature warmer.")
    if operation == "setSaturation" and parameter_name == "value":
        return _signed_note(value, "Desaturates the clip.", "Increases color saturation.")
    if operation == "setContrast" and parameter_name == "value":
        return _signed_note(value, "Softens contrast.", "Increases contrast.")
    if operation == "setExposure" and parameter_name == "value":
        return _signed_note(value, "Darkens the clip.", "Brightens the clip.")
    if operation == "setHighlights" and parameter_name == "value":
        return _signed_note(value, "Recovers bright highlights.", "Lifts bright highlights.")
    if operation == "setShadows" and parameter_name == "value":
        return _signed_note(value, "Deepens shadows.", "Lifts shadows.")
    return description.rstrip(".") + "."


def _effect_operation_intention(
    operation: ExperimentalEffectOperation,
    parameters: dict[str, JSONValue],
) -> str:
    existing = (operation.intention or "").strip()
    if existing:
        return existing

    if operation.operation == "addGrain":
        amount = _numeric_effect_parameter(parameters, "amount")
        return "add film grain" if amount is None or amount > 0 else "keep film grain unchanged"

    value = _numeric_effect_parameter(parameters, "value")
    if operation.operation == "setTemperature":
        return _signed_intention(value, "make it cooler", "make it warmer", "keep temperature unchanged")
    if operation.operation == "setSaturation":
        return _signed_intention(value, "make it more faded", "boost color saturation", "keep saturation unchanged")
    if operation.operation == "setContrast":
        return _signed_intention(value, "soften contrast", "add contrast", "keep contrast unchanged")
    if operation.operation == "setExposure":
        return _signed_intention(value, "darken the clip", "brighten the clip", "keep exposure unchanged")
    if operation.operation == "setHighlights":
        return _signed_intention(value, "recover highlights", "lift highlights", "keep highlights unchanged")
    if operation.operation == "setShadows":
        return _signed_intention(value, "deepen shadows", "lift shadows", "keep shadows unchanged")

    return operation.sourceText


def _numeric_effect_parameter(parameters: dict[str, JSONValue], name: str) -> float | None:
    value = parameters.get(name)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _signed_intention(value: float | None, negative: str, positive: str, neutral: str) -> str:
    if value is None:
        return positive
    if value < 0:
        return negative
    if value > 0:
        return positive
    return neutral


def _signed_note(value: float, negative_note: str, positive_note: str) -> str:
    if value < 0:
        return negative_note
    if value > 0:
        return positive_note
    return "Leaves this parameter unchanged."


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


def _capability_embedding_text(capability: EffectCapability) -> str:
    return "\n".join(
        [
            f"Operation: {capability.operation}",
            f"Description: {capability.description}",
            f"Retrieval text: {capability.retrievalText}",
            f"Examples: {'; '.join(capability.examples)}",
            "Parameters: "
            + "; ".join(
                f"{parameter.name} ({parameter.valueType}): {parameter.description}"
                for parameter in capability.parameters
            ),
        ]
    )


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0
    dot = sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0
    return dot / (left_norm * right_norm)


def _editor_context(context: IntentCompilerContext) -> dict[str, Any]:
    current_clip_id = _current_clip_at_playhead_id(context)
    current_clip = context.clip(context.selectedClipId) or context.clip(current_clip_id)
    transcript_context = _compact_transcript_context(context, current_clip.clipId if current_clip else None)
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
        "transcriptContext": transcript_context,
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


def _compact_transcript_context(context: IntentCompilerContext, clip_id: str | None) -> dict[str, Any] | None:
    if not clip_id:
        return None
    transcript = context.transcriptContextsByClipId.get(clip_id)
    if transcript is None:
        return None
    return {
        "clipId": transcript.clipId,
        "transcriptId": transcript.transcriptId,
        "fullTextExcerpt": (transcript.fullText or "")[:500],
        "pauseRanges": [
            {
                "start": pause.startUs,
                "end": pause.endUs,
                "duration": pause.durationUs,
                "beforeWord": pause.beforeWord,
                "afterWord": pause.afterWord,
            }
            for pause in transcript.pauseRanges[:20]
        ],
        "wordTimeline": [
            {"word": word.word, "start": word.startUs, "end": word.endUs}
            for word in transcript.words[:120]
        ],
    }


def _json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _log_intent_compile_incoming_context(*, prompt: str, context: IntentCompilerContext) -> None:
    """Emit full compiler context for each intent compile (prompt + API context + planner ctx)."""
    payload: dict[str, Any] = {
        "tag": "intent_compile_incoming",
        "prompt": prompt,
        "fullContext": context.model_dump(mode="json", by_alias=True),
        "plannerEditorContext": _editor_context(context),
    }
    try:
        line = json.dumps(payload, default=str)
    except (TypeError, ValueError):
        logger.exception("[intent-compile] Failed to serialize intent compile context")
        return
    logger.info("[intent-compile] %s", line)

