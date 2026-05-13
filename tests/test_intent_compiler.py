import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

import main
from app.intent_compiler.capabilities import DEFAULT_EFFECT_CAPABILITIES
from app.intent_compiler.compiler import IntentCompiler, action_execution_tier
from app.intent_compiler.llm import (
    IntentCompilerService,
    IntentLLMCompiler,
    _capability_embedding_text,
    _editor_context,
)
from app.intent_compiler.models import (
    ClipTranscriptContext,
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
    TranscriptWord,
)
from app.intent_compiler.transcript_phrases import collect_phrase_matches
from app.intent_compiler.transcripts import hydrate_intent_transcript_context, prepare_intent_transcript_context
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


def test_color_filter_effect_plan_emits_update_clip_color_filter_action():
    context = IntentCompilerContext.model_validate(_sample_context())
    plan = SemanticEditPlan(
        operations=[],
        experimentalEffectOperations=[
            ExperimentalEffectOperation(
                operation="setTemperature",
                sourceText="make it warmer",
                target={"type": "selectedClip"},
                confidence=0.9,
                parameters={"value": 0.4},
            )
        ],
        needsClarification=False,
    )

    result = IntentCompiler().compile(
        plan,
        original_prompt="make it warmer",
        context=context,
    )

    assert len(result.actions) == 1
    action = result.actions[0]
    assert action.type.value == "UPDATE_EFFECT_PARAMS"
    assert action.payload == {
        "updateClipColorFilter": {
            "clipId": "clip-b",
            "adjustments": {"temperature": 0.4},
        }
    }
    assert result.needsClarification is False
    assert IntentCompileWarning.unsupportedAction not in result.warnings


def test_color_filter_effect_groups_multiple_operations_per_clip():
    context = IntentCompilerContext.model_validate(_sample_context())
    plan = SemanticEditPlan(
        operations=[],
        experimentalEffectOperations=[
            ExperimentalEffectOperation(
                operation="setTemperature",
                sourceText="warmer and more saturated",
                target={"type": "selectedClip"},
                confidence=0.9,
                parameters={"value": 0.3},
            ),
            ExperimentalEffectOperation(
                operation="setSaturation",
                sourceText="warmer and more saturated",
                target={"type": "selectedClip"},
                confidence=0.85,
                parameters={"value": 0.2},
            ),
        ],
        needsClarification=False,
    )

    result = IntentCompiler().compile(
        plan,
        original_prompt="warmer and more saturated",
        context=context,
    )

    assert len(result.actions) == 1
    assert result.actions[0].payload == {
        "updateClipColorFilter": {
            "clipId": "clip-b",
            "adjustments": {"temperature": 0.3, "saturation": 0.2},
        }
    }


def test_unsupported_effect_operations_still_emit_warning():
    context = IntentCompilerContext.model_validate(_sample_context())
    plan = SemanticEditPlan(
        operations=[],
        experimentalEffectOperations=[
            ExperimentalEffectOperation(
                operation="addGrain",
                sourceText="make it grainy",
                target={"type": "selectedClip"},
                confidence=0.7,
                parameters={"amount": 0.3},
            )
        ],
        needsClarification=False,
    )

    result = IntentCompiler().compile(
        plan,
        original_prompt="make it grainy",
        context=context,
    )

    assert result.actions == []
    assert IntentCompileWarning.unsupportedAction in result.warnings


def test_color_filter_effect_with_unresolved_target_clarifies():
    context_payload = {
        **_sample_context(),
        "selectedClipId": None,
        "playheadTimeUs": None,
    }
    context = IntentCompilerContext.model_validate(context_payload)
    plan = SemanticEditPlan(
        operations=[],
        experimentalEffectOperations=[
            ExperimentalEffectOperation(
                operation="setTemperature",
                sourceText="make it warmer",
                target=None,
                confidence=0.8,
                parameters={"value": 0.3},
            )
        ],
        needsClarification=False,
    )

    result = IntentCompiler().compile(
        plan,
        original_prompt="make it warmer",
        context=context,
    )

    assert result.actions == []
    assert IntentCompileWarning.missingSelectedClip in result.warnings


def test_mixed_structural_and_effect_plan_emits_both_actions():
    context = IntentCompilerContext.model_validate(_sample_context())
    plan = SemanticEditPlan(
        operations=[
            SemanticEditOperation(
                type=IntentEditType.splitClip,
                sourceText="split this clip in half",
                target={"type": "selectedClip"},
                parameters={"position": {"type": "fractionOfClip", "value": 0.5}},
                confidence=0.9,
            )
        ],
        experimentalEffectOperations=[
            ExperimentalEffectOperation(
                operation="setSaturation",
                sourceText="and make it pop",
                target={"type": "selectedClip"},
                confidence=0.85,
                parameters={"value": 0.5},
            )
        ],
        needsClarification=False,
    )

    result = IntentCompiler().compile(
        plan,
        original_prompt="split this clip in half and make it pop",
        context=context,
    )

    assert len(result.actions) == 2
    assert result.actions[0].type.value == "UPDATE_EFFECT_PARAMS"
    assert result.actions[1].type.value == "SPLIT_CLIP"
    tiers = [action_execution_tier(action.type) for action in result.actions]
    assert tiers == sorted(tiers)

    update_action = result.actions[0]
    assert update_action.payload == {
        "updateClipColorFilter": {
            "clipId": "clip-b",
            "adjustments": {"saturation": 0.5},
        }
    }


def test_dead_space_plan_emits_remove_clip_ranges_action():
    context = IntentCompilerContext.model_validate(
        {
            **_sample_context(),
            "transcriptContextsByClipId": {
                "clip-b": {
                    "clipId": "clip-b",
                    "pauseRanges": [
                        {
                            "startUs": 2_000_000,
                            "endUs": 3_000_000,
                            "durationUs": 1_000_000,
                            "beforeWord": "hello",
                            "afterWord": "world",
                        }
                    ],
                }
            },
        }
    )
    plan = SemanticEditPlan(
        operations=[
            SemanticEditOperation(
                type=IntentEditType.removeClipRanges,
                sourceText="cut out the dead space",
                target={"type": "selectedClip"},
                parameters={},
                confidence=0.9,
            )
        ],
    )

    result = IntentCompiler().compile(plan, original_prompt="cut out the dead space", context=context)

    assert result.needsClarification is False
    assert len(result.actions) == 1
    action = result.actions[0]
    assert action.type == "REMOVE_CLIP_RANGES"
    assert action.payload == {
        "removeClipRanges": {
            "clipId": "clip-b",
            "sourceRanges": [{"start": 2_000_000, "end": 3_000_000}],
        }
    }


def test_remove_clip_ranges_rejects_out_of_clip_range():
    context = IntentCompilerContext.model_validate(_sample_context())
    plan = SemanticEditPlan(
        operations=[
            SemanticEditOperation(
                type=IntentEditType.removeClipRanges,
                sourceText="remove this dead space",
                target={"type": "selectedClip"},
                parameters={"sourceRanges": [{"start": 9_000_000, "end": 12_000_000}]},
                confidence=0.9,
            )
        ],
    )

    result = IntentCompiler().compile(plan, original_prompt="remove this dead space", context=context)

    assert result.actions == []
    assert IntentCompileWarning.invalidRemoveRange in result.warnings


def test_prepare_intent_transcript_context_skips_hydration_without_signal():
    async def _exercise():
        context = IntentCompilerContext.model_validate(
            {
                **_sample_context(),
                "transcriptContextsByClipId": {
                    "clip-b": {"clipId": "clip-b", "transcriptId": "1"},
                },
            }
        )

        async def _hydrate_should_not_run(*_args, **_kwargs):
            raise AssertionError("hydrate_intent_transcript_context should not run")

        with patch(
            "app.intent_compiler.transcripts.hydrate_intent_transcript_context",
            side_effect=_hydrate_should_not_run,
        ):
            prepared, meta = await prepare_intent_transcript_context(
                prompt="split here at the playhead",
                context=context,
                db=AsyncMock(),
            )

        assert prepared is context
        assert meta["transcript_preflight"] == "skipped_no_signal"
        assert meta["skip_reason"] == "no_transcript_intent_signal"

    asyncio.run(_exercise())


def test_prepare_intent_transcript_context_skips_when_transcript_needed_but_no_refs():
    async def _exercise():
        context = IntentCompilerContext.model_validate(_sample_context())

        async def _hydrate_should_not_run(*_args, **_kwargs):
            raise AssertionError("hydrate_intent_transcript_context should not run")

        with patch(
            "app.intent_compiler.transcripts.hydrate_intent_transcript_context",
            side_effect=_hydrate_should_not_run,
        ):
            prepared, meta = await prepare_intent_transcript_context(
                prompt="remove the long pauses",
                context=context,
                db=AsyncMock(),
            )

        assert prepared is context
        assert meta["transcript_preflight"] == "skipped_empty_refs"
        assert meta["skip_reason"] == "transcript_intent_but_no_clip_refs"

    asyncio.run(_exercise())


def test_prepare_intent_transcript_context_attaches_phrase_matches_after_hydrate():
    async def _exercise():
        context = IntentCompilerContext.model_validate(
            {
                **_sample_context(),
                "transcriptContextsByClipId": {
                    "clip-b": {"clipId": "clip-b", "transcriptId": "1"},
                },
            }
        )

        words = [
            TranscriptWord(word="this", startUs=0, endUs=100_000),
            TranscriptWord(word="is", startUs=100_000, endUs=200_000),
            TranscriptWord(word="iris", startUs=200_000, endUs=400_000),
        ]

        async def fake_hydrate(ctx: IntentCompilerContext, _db):
            return ctx.model_copy(
                update={
                    "transcriptContextsByClipId": {
                        "clip-b": ClipTranscriptContext(
                            clipId="clip-b",
                            transcriptId=1,
                            words=words,
                            pauseRanges=[],
                        )
                    }
                }
            )

        with patch(
            "app.intent_compiler.transcripts.hydrate_intent_transcript_context",
            side_effect=fake_hydrate,
        ):
            prepared, meta = await prepare_intent_transcript_context(
                prompt='remove where I say "this is iris"',
                context=context,
                db=AsyncMock(),
            )

        assert meta["transcript_preflight"] == "hydrated"
        assert meta["phrase_match_total"] >= 1
        clip_ctx = prepared.transcriptContextsByClipId["clip-b"]
        assert len(clip_ctx.phraseMatches) >= 1
        assert "iris" in clip_ctx.phraseMatches[0].phrase.lower()

    asyncio.run(_exercise())


def test_collect_phrase_matches_finds_contiguous_words():
    words = [
        TranscriptWord(word="Hello,", startUs=0, endUs=50_000),
        TranscriptWord(word="this", startUs=60_000, endUs=120_000),
        TranscriptWord(word="is", startUs=120_000, endUs=180_000),
        TranscriptWord(word="Iris.", startUs=180_000, endUs=240_000),
    ]
    matches = collect_phrase_matches('cut out "this is iris"', words)
    assert len(matches) == 1
    assert matches[0].startUs < matches[0].endUs


def test_remove_clip_ranges_compiler_falls_back_to_phrase_matches():
    words = [
        TranscriptWord(word="hello", startUs=0, endUs=50_000),
        TranscriptWord(word="this", startUs=60_000, endUs=120_000),
        TranscriptWord(word="is", startUs=120_000, endUs=180_000),
        TranscriptWord(word="iris", startUs=180_000, endUs=240_000),
    ]
    matches = collect_phrase_matches('cut "this is iris"', words)
    context = IntentCompilerContext.model_validate(
        {
            **_sample_context(),
            "transcriptContextsByClipId": {
                "clip-b": ClipTranscriptContext(
                    clipId="clip-b",
                    words=words,
                    pauseRanges=[],
                    phraseMatches=matches,
                )
            },
        }
    )
    plan = SemanticEditPlan(
        operations=[
            SemanticEditOperation(
                type=IntentEditType.removeClipRanges,
                sourceText='cut "this is iris"',
                target={"type": "selectedClip"},
                parameters={},
                confidence=0.9,
            )
        ],
    )

    result = IntentCompiler().compile(plan, original_prompt='cut "this is iris"', context=context)

    assert result.needsClarification is False
    assert len(result.actions) == 1
    assert result.actions[0].type == "REMOVE_CLIP_RANGES"


def test_split_clip_defaults_to_playhead_when_position_omitted():
    context = IntentCompilerContext.model_validate(_sample_context())
    plan = SemanticEditPlan(
        operations=[
            SemanticEditOperation(
                type=IntentEditType.splitClip,
                sourceText="Split here.",
                target={"type": "selectedClip"},
                parameters={},
                confidence=0.9,
            )
        ],
    )

    result = IntentCompiler().compile(plan, original_prompt="Split here.", context=context)

    assert result.needsClarification is False
    assert len(result.actions) == 1
    assert result.actions[0].payload["splitClip"] == {"clipId": "clip-b", "atTimeUs": 10_000_000}


def test_split_clip_accepts_playhead_position_type_any_case():
    context = IntentCompilerContext.model_validate(_sample_context())
    plan = SemanticEditPlan(
        operations=[
            SemanticEditOperation(
                type=IntentEditType.splitClip,
                sourceText="Split here.",
                target={"type": "selectedClip"},
                parameters={"position": {"type": "Playhead"}},
                confidence=0.9,
            )
        ],
    )

    result = IntentCompiler().compile(plan, original_prompt="Split here.", context=context)

    assert result.actions[0].payload["splitClip"]["atTimeUs"] == 10_000_000


def test_split_clip_without_position_clarifies_when_playhead_outside_clip():
    context = IntentCompilerContext.model_validate({**_sample_context(), "playheadTimeUs": 25_000_000})
    plan = SemanticEditPlan(
        operations=[
            SemanticEditOperation(
                type=IntentEditType.splitClip,
                sourceText="Split here.",
                target={"type": "selectedClip"},
                parameters={},
                confidence=0.9,
            )
        ],
    )

    result = IntentCompiler().compile(plan, original_prompt="Split here.", context=context)

    assert result.needsClarification is True
    assert IntentCompileWarning.missingPlayhead in result.warnings


def test_editor_context_includes_compact_transcript_pause_ranges():
    context = IntentCompilerContext.model_validate(
        {
            **_sample_context(),
            "transcriptContextsByClipId": {
                "clip-b": {
                    "clipId": "clip-b",
                    "fullText": "hello world",
                    "words": [
                        {"word": "hello", "startUs": 1_000_000, "endUs": 1_300_000},
                        {"word": "world", "startUs": 2_000_000, "endUs": 2_300_000},
                    ],
                    "pauseRanges": [
                        {
                            "startUs": 1_300_000,
                            "endUs": 2_000_000,
                            "durationUs": 700_000,
                            "beforeWord": "hello",
                            "afterWord": "world",
                        }
                    ],
                }
            },
        }
    )

    editor_context = _editor_context(context)

    assert editor_context["transcriptContext"]["fullTextExcerpt"] == "hello world"
    assert editor_context["transcriptContext"]["pauseRanges"] == [
        {
            "start": 1_300_000,
            "end": 2_000_000,
            "duration": 700_000,
            "beforeWord": "hello",
            "afterWord": "world",
        }
    ]


def test_hydrate_intent_transcript_context_reads_sql_and_caches(monkeypatch):
    calls = {}

    async def fake_get_cached_transcript(cache_key):
        calls["cache_key"] = cache_key
        return None

    async def fake_get_transcript_payload(_db, transcript_id):
        calls["transcript_id"] = transcript_id
        return {
            "full_text": "hello world",
            "segments": [
                {
                    "text": "hello world",
                    "words": [
                        {"word": "hello", "start": 1.0, "end": 1.3},
                        {"word": "world", "start": 2.0, "end": 2.3},
                    ],
                }
            ],
        }

    async def fake_cache_transcript(session_id, clip_id, transcript_payload):
        calls["cache"] = (session_id, clip_id, transcript_payload["full_text"])
        return f"session:{session_id}:transcript:{clip_id}"

    monkeypatch.setattr("app.intent_compiler.transcripts.get_cached_transcript", fake_get_cached_transcript)
    monkeypatch.setattr("app.intent_compiler.transcripts.get_transcript_payload", fake_get_transcript_payload)
    monkeypatch.setattr("app.intent_compiler.transcripts.cache_transcript", fake_cache_transcript)
    context = IntentCompilerContext.model_validate(
        {
            **_sample_context(),
            "sessionId": "session-1",
            "transcriptContextsByClipId": {
                "clip-b": {"clipId": "clip-b", "transcriptId": "501", "cacheKey": "missing-cache"}
            },
        }
    )

    hydrated = asyncio.run(hydrate_intent_transcript_context(context, object()))

    transcript = hydrated.transcriptContextsByClipId["clip-b"]
    assert calls["cache_key"] == "missing-cache"
    assert calls["transcript_id"] == 501
    assert calls["cache"] == ("session-1", "clip-b", "hello world")
    assert transcript.cacheKey == "session:session-1:transcript:clip-b"
    assert transcript.words[0].startUs == 1_000_000
    assert transcript.pauseRanges[0].startUs == 1_300_000


def test_hydrate_intent_transcript_context_loads_sentence_upload_uuid(monkeypatch):
    uid = "aaaaaaaa-bbbb-4ccc-a123-456789abcdef"
    calls: dict[str, object] = {}

    async def fake_get_cached_transcript(cache_key):
        calls["cache_key"] = cache_key
        return None

    async def fake_get_transcript_payload(_db, transcript_id):
        calls["clip_lookup"] = transcript_id
        return None

    async def fake_get_sentence_upload(_db, transcript_id):
        calls["sentence_lookup"] = transcript_id
        return {
            "full_text": "Hi there",
            "segments": [
                {
                    "text": "Hi there",
                    "words": [
                        {"word": "Hi", "start": 0.0, "end": 0.2},
                        {"word": "there", "start": 0.25, "end": 0.5},
                    ],
                }
            ],
        }

    async def fake_cache_transcript(session_id, clip_id, transcript_payload):
        calls["cache"] = (session_id, clip_id, transcript_payload["full_text"])
        return f"session:{session_id}:transcript:{clip_id}"

    monkeypatch.setattr("app.intent_compiler.transcripts.get_cached_transcript", fake_get_cached_transcript)
    monkeypatch.setattr("app.intent_compiler.transcripts.get_transcript_payload", fake_get_transcript_payload)
    monkeypatch.setattr(
        "app.intent_compiler.transcripts.get_sentence_upload_transcript_payload",
        fake_get_sentence_upload,
    )
    monkeypatch.setattr("app.intent_compiler.transcripts.cache_transcript", fake_cache_transcript)

    context = IntentCompilerContext.model_validate(
        {
            **_sample_context(),
            "sessionId": "session-1",
            "transcriptContextsByClipId": {
                "clip-b": {"clipId": "clip-b", "transcriptId": uid, "cacheKey": "missing-cache"}
            },
        }
    )

    hydrated = asyncio.run(hydrate_intent_transcript_context(context, object()))

    transcript = hydrated.transcriptContextsByClipId["clip-b"]
    assert "clip_lookup" not in calls
    assert calls["sentence_lookup"] == uid
    assert calls["cache"] == ("session-1", "clip-b", "Hi there")
    assert transcript.cacheKey == "session:session-1:transcript:clip-b"
    assert transcript.words[0].word == "Hi"
    assert transcript.words[1].word == "there"


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


def test_trim_first_second_idiom_parses_as_one_second():
    context = IntentCompilerContext.model_validate(_sample_context())
    plan = SemanticEditPlan(
        operations=[
            SemanticEditOperation(
                type=IntentEditType.trimClip,
                sourceText="",
                target={"type": "selectedClip"},
                parameters={},
                confidence=0.9,
            )
        ],
    )

    result = IntentCompiler().compile(
        plan,
        original_prompt="Trim the first second of this clip.",
        context=context,
    )

    assert result.needsClarification is False
    assert len(result.actions) == 1
    assert result.actions[0].payload["trimClip"]["sourceRange"] == {"start": 1_000_000, "end": 10_000_000}


def test_trim_first_two_seconds_idiom():
    context = IntentCompilerContext.model_validate(_sample_context())
    plan = SemanticEditPlan(
        operations=[
            SemanticEditOperation(
                type=IntentEditType.trimClip,
                sourceText="trim start",
                target={"type": "selectedClip"},
                parameters={},
                confidence=0.9,
            )
        ],
    )

    result = IntentCompiler().compile(
        plan,
        original_prompt="Please trim the first two seconds of this clip.",
        context=context,
    )

    assert result.needsClarification is False
    assert result.actions[0].payload["trimClip"]["sourceRange"] == {"start": 2_000_000, "end": 10_000_000}


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

