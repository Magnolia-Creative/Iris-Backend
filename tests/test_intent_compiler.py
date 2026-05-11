import asyncio
from contextlib import asynccontextmanager

from fastapi.testclient import TestClient

import main
from app.intent_compiler.capabilities import DEFAULT_EFFECT_CAPABILITIES
from app.intent_compiler.compiler import IntentCompiler
from app.intent_compiler.llm import (
    IntentCompilerService,
    IntentLLMCompiler,
    _capability_embedding_text,
)
from app.intent_compiler.models import (
    CompileSource,
    EffectCapability,
    EffectCapabilityParameter,
    ExperimentalEffectOperation,
    ExperimentalEffectPlan,
    IntentCompileResult,
    IntentCompileWarning,
    IntentCompilerContext,
    IntentEditType,
    RelevantEffectCapability,
    SemanticEffectRequest,
    SemanticEditOperation,
    SemanticEditPlan,
)
from app.services.realtime_transcription import DEFAULT_TRANSCRIBE_MODEL


@asynccontextmanager
async def _noop_lifespan(_app):
    yield


def _sample_context() -> dict:
    return {
        "timelineId": "timeline-test",
        "selectedClipId": "clip-b",
        "selectedTrackId": "track-video",
        "selectedRange": None,
        "playheadTimeUs": 10_000_000,
        "clipsById": {
            "clip-b": {
                "clip_id": "clip-b",
                "track_id": "track-video",
                "media_id": "media-b",
                "source_range": {"start": 0, "end": 10_000_000},
                "timeline_range": {"start": 5_000_000, "end": 15_000_000},
            }
        },
        "orderedClipIdsByTrackId": {"track-video": ["clip-b"]},
    }


def _multi_clip_context() -> dict:
    return {
        "timelineId": "timeline-test",
        "selectedClipId": "clip-b",
        "selectedTrackId": "track-video",
        "selectedRange": None,
        "playheadTimeUs": 10_000_000,
        "clipsById": {
            "clip-a": {
                "clip_id": "clip-a",
                "track_id": "track-video",
                "media_id": "media-a",
                "source_range": {"start": 0, "end": 5_000_000},
                "timeline_range": {"start": 0, "end": 5_000_000},
            },
            "clip-b": {
                "clip_id": "clip-b",
                "track_id": "track-video",
                "media_id": "media-b",
                "source_range": {"start": 0, "end": 10_000_000},
                "timeline_range": {"start": 5_000_000, "end": 15_000_000},
            },
            "clip-c": {
                "clip_id": "clip-c",
                "track_id": "track-video",
                "media_id": "media-c",
                "source_range": {"start": 0, "end": 5_000_000},
                "timeline_range": {"start": 15_000_000, "end": 20_000_000},
            },
        },
        "orderedClipIdsByTrackId": {"track-video": ["clip-a", "clip-b", "clip-c"]},
    }


def test_effect_only_plan_preserves_experimental_effects_without_actions():
    context = IntentCompilerContext.model_validate(_sample_context())
    plan = SemanticEditPlan(
        operations=[],
        experimentalEffectOperations=[
            ExperimentalEffectOperation(
                operation="addGrain",
                sourceText="make this clip feel vintage",
                target={"type": "selectedClip"},
                confidence=0.8,
                parameters={"amount": 0.4},
            )
        ],
        needsClarification=False,
        clarificationQuestion=None,
    )

    result = IntentCompiler().compile(
        plan,
        original_prompt="make this clip feel vintage",
        context=context,
    )

    assert result.actions == []
    assert result.experimentalEffectOperations[0].operation == "addGrain"
    assert IntentCompileWarning.unsupportedAction in result.warnings


def test_effect_parameters_are_clamped_to_capability_schema():
    capability = DEFAULT_EFFECT_CAPABILITIES[0]
    operation = ExperimentalEffectOperation(
        operation=capability.operation,
        sourceText="make it extremely grainy",
        target={"type": "selectedClip"},
        confidence=0.9,
        parameters={"amount": 99},
    )

    valid, warnings = IntentCompilerService._validate_effect_operations(
        [operation],
        [RelevantEffectCapability(capability=capability, score=1.0)],
    )

    assert warnings == []
    assert valid[0].parameters["amount"] == 1
    assert valid[0].parameterNotes["amount"] == "Adds visible film grain."


def test_missing_effect_parameters_are_inferred_with_value_notes():
    capability = next(item for item in DEFAULT_EFFECT_CAPABILITIES if item.operation == "setTemperature")
    operation = ExperimentalEffectOperation(
        operation=capability.operation,
        sourceText="make the clip cooler",
        target={"type": "selectedClip"},
        confidence=0.84,
        parameters={},
    )

    valid, warnings = IntentCompilerService._validate_effect_operations(
        [operation],
        [RelevantEffectCapability(capability=capability, score=1.0)],
    )

    assert warnings == []
    assert valid[0].parameters["value"] < 0
    assert valid[0].intention == "make it cooler"
    assert valid[0].parameterNotes["value"] == "Makes temperature cooler."


def test_missing_effect_parameters_use_operation_specific_values():
    capabilities = [
        capability
        for capability in DEFAULT_EFFECT_CAPABILITIES
        if capability.operation in {"setTemperature", "setContrast", "setSaturation"}
    ]
    operations = [
        ExperimentalEffectOperation(
            operation=capability.operation,
            sourceText="give it a warm cinematic look",
            target={"type": "selectedClip"},
            confidence=0.75,
            parameters={},
        )
        for capability in capabilities
    ]

    valid, warnings = IntentCompilerService._validate_effect_operations(
        operations,
        [RelevantEffectCapability(capability=capability, score=1.0) for capability in capabilities],
    )

    values = {operation.operation: operation.parameters["value"] for operation in valid}
    assert warnings == []
    assert values == {
        "setTemperature": 0.35,
        "setSaturation": 0.12,
        "setContrast": 0.32,
    }
    assert {operation.operation: operation.intention for operation in valid} == {
        "setTemperature": "make it warmer",
        "setSaturation": "boost color saturation",
        "setContrast": "add contrast",
    }


def test_trim_infers_spoken_seconds_from_source_text():
    context = IntentCompilerContext.model_validate(_sample_context())
    plan = SemanticEditPlan(
        operations=[
            SemanticEditOperation(
                type=IntentEditType.trimClip,
                sourceText="Trim the first part of the video by two seconds",
                target={"type": "selectedClip"},
                parameters={},
                confidence=0.9,
            )
        ],
        needsClarification=False,
        clarificationQuestion=None,
    )

    result = IntentCompiler().compile(
        plan,
        original_prompt="Trim the first part of the video by two seconds",
        context=context,
    )

    assert result.needsClarification is False
    assert result.warnings == []
    assert len(result.actions) == 1
    trim_payload = result.actions[0].payload["trimClip"]
    assert trim_payload["sourceRange"] == {"start": 2_000_000, "end": 10_000_000}
    assert trim_payload["timelineRange"] == {"start": 7_000_000, "end": 15_000_000}


def test_relevant_effect_capabilities_are_ranked_by_embedding_similarity():
    warm_capability = _test_capability("setTemperature", "warm cool temperature golden hour")
    grain_capability = _test_capability("addGrain", "grain analog noise texture")
    saturation_capability = _test_capability("setSaturation", "saturation vibrant color intensity")
    effect_request = SemanticEffectRequest(sourceText="make it feel warmer", intent="warmth")
    query = "make it feel warmer warmth"
    embedding_client = _FakeEmbeddingClient(
        {
            query: [1.0, 0.0],
            _capability_embedding_text(warm_capability): [0.95, 0.05],
            _capability_embedding_text(grain_capability): [0.0, 1.0],
            _capability_embedding_text(saturation_capability): [0.5, 0.5],
        }
    )
    service = IntentCompilerService(
        llm_compiler=object(),
        capabilities=[grain_capability, saturation_capability, warm_capability],
        embedding_client=embedding_client,
    )

    relevant = asyncio.run(service._relevant_capabilities(effect_request))

    assert [item.capability.operation for item in relevant] == [
        "setTemperature",
        "setSaturation",
        "addGrain",
    ]
    assert relevant[0].score > relevant[1].score > relevant[2].score


def test_effect_capability_embeddings_are_cached_per_service_instance():
    warm_capability = _test_capability("setTemperature", "warm cool temperature golden hour")
    grain_capability = _test_capability("addGrain", "grain analog noise texture")
    first_query = "make it feel warmer warmth"
    second_query = "add some film texture texture"
    capability_texts = [
        _capability_embedding_text(warm_capability),
        _capability_embedding_text(grain_capability),
    ]
    embedding_client = _FakeEmbeddingClient(
        {
            first_query: [1.0, 0.0],
            second_query: [0.0, 1.0],
            capability_texts[0]: [1.0, 0.0],
            capability_texts[1]: [0.0, 1.0],
        }
    )
    service = IntentCompilerService(
        llm_compiler=object(),
        capabilities=[warm_capability, grain_capability],
        embedding_client=embedding_client,
    )

    asyncio.run(
        service._relevant_capabilities(
            SemanticEffectRequest(sourceText="make it feel warmer", intent="warmth")
        )
    )
    asyncio.run(
        service._relevant_capabilities(
            SemanticEffectRequest(sourceText="add some film texture", intent="texture")
        )
    )

    assert embedding_client.requests.count(capability_texts) == 1


def test_compile_prompt_reports_embedding_unavailable_when_effect_retrieval_fails():
    class FakeLLMCompiler:
        async def make_semantic_plan(self, _prompt, _context):
            return SemanticEditPlan(
                effectRequests=[
                    SemanticEffectRequest(sourceText="make this clip feel dreamy", intent="dreamy")
                ]
            )

        async def plan_experimental_effects(self, **_kwargs):
            raise AssertionError("Effect planning should not run without retrieved capabilities")

    class FailingEmbeddingClient:
        async def aembed_documents(self, _texts):
            raise RuntimeError("embedding service unavailable")

    context = IntentCompilerContext.model_validate(_sample_context())
    service = IntentCompilerService(
        llm_compiler=FakeLLMCompiler(),
        embedding_client=FailingEmbeddingClient(),
    )

    result = asyncio.run(service.compile_prompt(prompt="make this clip feel dreamy", context=context))

    assert result.actions == []
    assert IntentCompileWarning.embeddingUnavailable in result.warnings
    assert IntentCompileWarning.noActionProduced in result.warnings


def test_compile_prompt_repairs_selected_clip_split_in_half_clarification():
    class FakeLLMCompiler:
        async def make_semantic_plan(self, _prompt, _context):
            return SemanticEditPlan(
                needsClarification=True,
                clarificationQuestion="Which clip should I split?",
            )

        async def plan_experimental_effects(self, **_kwargs):
            raise AssertionError("Effect planning should not run for a split operation")

    context = IntentCompilerContext.model_validate(_sample_context())
    service = IntentCompilerService(
        llm_compiler=FakeLLMCompiler(),
        embedding_client=object(),
    )

    result = asyncio.run(service.compile_prompt(prompt="Split this clip in half.", context=context))

    assert result.needsClarification is False
    assert result.confidence == 0.9
    assert result.actions[0].payload["splitClip"] == {"clipId": "clip-b", "atTimeUs": 10_000_000}
    assert IntentCompileWarning.ambiguousTarget not in result.warnings


def test_compile_prompt_assumes_selected_clip_for_targetless_operation():
    class FakeLLMCompiler:
        async def make_semantic_plan(self, _prompt, _context):
            return SemanticEditPlan(
                operations=[
                    SemanticEditOperation(
                        type=IntentEditType.splitClip,
                        sourceText="Split this clip in half.",
                        target=None,
                        parameters={"position": {"type": "fractionOfClip", "value": 0.5}},
                        confidence=0.2,
                    )
                ],
                needsClarification=True,
                clarificationQuestion="Which clip should I split?",
            )

        async def plan_experimental_effects(self, **_kwargs):
            raise AssertionError("Effect planning should not run for a split operation")

    context = IntentCompilerContext.model_validate(_sample_context())
    service = IntentCompilerService(
        llm_compiler=FakeLLMCompiler(),
        embedding_client=object(),
    )

    result = asyncio.run(service.compile_prompt(prompt="Split this clip in half.", context=context))

    assert result.needsClarification is False
    assert result.confidence == 0.85
    assert result.actions[0].payload["splitClip"] == {"clipId": "clip-b", "atTimeUs": 10_000_000}
    assert IntentCompileWarning.ambiguousTarget not in result.warnings


def test_compile_prompt_prioritizes_selected_clip_for_ambiguous_first_duration():
    class FakeLLMCompiler:
        async def make_semantic_plan(self, _prompt, _context):
            return SemanticEditPlan(
                operations=[
                    SemanticEditOperation(
                        type=IntentEditType.trimClip,
                        sourceText="Trim the first two seconds of the clip.",
                        target={
                            "type": "ordinal",
                            "value": "first",
                            "track": {"type": "selectedTrack"},
                        },
                        parameters={"edge": "start", "amount": {"type": "duration", "value": 2, "unit": "second"}},
                        confidence=0.8,
                    )
                ],
                needsClarification=False,
            )

        async def plan_experimental_effects(self, **_kwargs):
            raise AssertionError("Effect planning should not run for a trim operation")

    context = IntentCompilerContext.model_validate(_multi_clip_context())
    service = IntentCompilerService(
        llm_compiler=FakeLLMCompiler(),
        embedding_client=object(),
    )

    result = asyncio.run(service.compile_prompt(prompt="Trim the first two seconds of the clip.", context=context))

    assert result.needsClarification is False
    assert result.actions[0].payload["trimClip"]["clipId"] == "clip-b"


def test_compile_prompt_preserves_explicit_first_clip_target():
    class FakeLLMCompiler:
        async def make_semantic_plan(self, _prompt, _context):
            return SemanticEditPlan(
                operations=[
                    SemanticEditOperation(
                        type=IntentEditType.trimClip,
                        sourceText="Trim the first clip by two seconds.",
                        target={
                            "type": "ordinal",
                            "value": "first",
                            "track": {"type": "selectedTrack"},
                        },
                        parameters={"edge": "start", "amount": {"type": "duration", "value": 2, "unit": "second"}},
                        confidence=0.8,
                    )
                ],
                needsClarification=False,
            )

        async def plan_experimental_effects(self, **_kwargs):
            raise AssertionError("Effect planning should not run for a trim operation")

    context = IntentCompilerContext.model_validate(_multi_clip_context())
    service = IntentCompilerService(
        llm_compiler=FakeLLMCompiler(),
        embedding_client=object(),
    )

    result = asyncio.run(service.compile_prompt(prompt="Trim the first clip by two seconds.", context=context))

    assert result.needsClarification is False
    assert result.actions[0].payload["trimClip"]["clipId"] == "clip-a"


def test_intent_llm_compiler_uses_function_calling_for_planner_schemas():
    class FakeStructuredLLM:
        def __init__(self, result):
            self.result = result

        async def ainvoke(self, _messages):
            return self.result

    class FakeLLM:
        def __init__(self):
            self.calls = []

        def with_structured_output(self, schema, **kwargs):
            self.calls.append((schema, kwargs))
            if schema is SemanticEditPlan:
                return FakeStructuredLLM(SemanticEditPlan())
            if schema is ExperimentalEffectPlan:
                return FakeStructuredLLM(ExperimentalEffectPlan())
            raise AssertionError(f"Unexpected schema: {schema}")

    fake_llm = FakeLLM()
    compiler = IntentLLMCompiler(llm=fake_llm)
    context = IntentCompilerContext.model_validate(_sample_context())

    asyncio.run(compiler.make_semantic_plan("make it vintage", context))
    asyncio.run(
        compiler.plan_experimental_effects(
            original_prompt="make it vintage",
            effect_request=SemanticEffectRequest(sourceText="make it vintage"),
            relevant_capabilities=[
                RelevantEffectCapability(capability=DEFAULT_EFFECT_CAPABILITIES[0], score=1.0)
            ],
            context=context,
        )
    )

    assert fake_llm.calls == [
        (SemanticEditPlan, {"method": "function_calling"}),
        (ExperimentalEffectPlan, {"method": "function_calling"}),
    ]


def test_text_intent_run_streams_final_result(monkeypatch):
    class FakeIntentCompilerService:
        async def compile_prompt(self, *, prompt, context, event_handler=None):
            assert prompt == "make this clip feel vintage"
            if event_handler:
                await event_handler({"type": "planner_started", "status": "Parsing prompt."})
            return IntentCompileResult(
                actions=[],
                confidence=0,
                source=CompileSource.llm,
                unresolvedText=prompt,
                warnings=[IntentCompileWarning.unsupportedAction],
                needsClarification=False,
                experimentalEffectOperations=[],
            )

    monkeypatch.setattr(main, "IntentCompilerService", FakeIntentCompilerService)
    original_lifespan = main.app.router.lifespan_context
    main.app.router.lifespan_context = _noop_lifespan
    try:
        with TestClient(main.app) as client:
            response = client.post(
                "/intent-runs",
                json={"prompt": "make this clip feel vintage", "context": _sample_context()},
            )
            assert response.status_code == 200
            run_id = response.json()["run_id"]
            with client.websocket_connect(f"/ws/intent-runs/{run_id}") as websocket:
                assert websocket.receive_json()["type"] == "run_started"
                assert websocket.receive_json()["type"] == "planner_started"
                final = websocket.receive_json()
                assert final["type"] == "intent_result"
                assert final["result"]["actions"] == []
    finally:
        main.app.router.lifespan_context = original_lifespan


class _FakeEmbeddingClient:
    def __init__(self, embeddings_by_text):
        self.embeddings_by_text = embeddings_by_text
        self.requests = []

    async def aembed_documents(self, texts):
        self.requests.append(texts)
        return [self.embeddings_by_text[text] for text in texts]


def _test_capability(operation: str, retrieval_text: str) -> EffectCapability:
    return EffectCapability(
        operation=operation,
        description=retrieval_text,
        parameters=[
            EffectCapabilityParameter(
                name="value",
                valueType="number",
                minimum=-1,
                maximum=1,
                description="Test value.",
            )
        ],
        retrievalText=retrieval_text,
        examples=[retrieval_text],
    )


def test_voice_intent_websocket_delegates_after_start(monkeypatch):
    async def fake_stream_voice_intent(websocket, *, context, transcription_model):
        assert context.timelineId == "timeline-test"
        assert transcription_model == DEFAULT_TRANSCRIBE_MODEL
        await websocket.send_json(
            {
                "type": "intent_result",
                "prompt": "make it vintage",
                "result": {
                    "actions": [],
                    "confidence": 0,
                    "source": "llm",
                    "unresolvedText": "make it vintage",
                    "warnings": [],
                    "needsClarification": False,
                    "experimentalEffectOperations": [],
                },
            }
        )

    monkeypatch.setattr(main, "stream_voice_intent", fake_stream_voice_intent)
    original_lifespan = main.app.router.lifespan_context
    main.app.router.lifespan_context = _noop_lifespan
    try:
        with TestClient(main.app) as client:
            with client.websocket_connect("/ws/intent/voice") as websocket:
                websocket.send_json({"type": "start", "context": _sample_context()})
                final = websocket.receive_json()
                assert final["type"] == "intent_result"
    finally:
        main.app.router.lifespan_context = original_lifespan

