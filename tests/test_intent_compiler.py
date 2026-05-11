import asyncio
from contextlib import asynccontextmanager

from fastapi.testclient import TestClient

import main
from app.services.intent_compiler.capabilities import DEFAULT_EFFECT_CAPABILITIES
from app.services.intent_compiler.compiler import IntentCompiler
from app.services.intent_compiler.llm import IntentCompilerService, IntentLLMCompiler
from app.services.intent_compiler.models import (
    CompileSource,
    ExperimentalEffectOperation,
    ExperimentalEffectPlan,
    IntentCompileResult,
    IntentCompileWarning,
    IntentCompilerContext,
    RelevantEffectCapability,
    SemanticEffectRequest,
    SemanticEditPlan,
)


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


def test_voice_intent_websocket_delegates_after_start(monkeypatch):
    async def fake_stream_voice_intent(websocket, *, context, transcription_model):
        assert context.timelineId == "timeline-test"
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

