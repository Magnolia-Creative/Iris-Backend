from app.intent_compiler.transcript_preflight import (
    needs_phrase_matching,
    needs_transcript_hydration,
    resolve_transcript_target_clip_ids,
)
from app.intent_compiler.models import IntentCompilerContext


def test_needs_transcript_hydration_detects_pause_and_phrase_signals():
    assert needs_transcript_hydration("remove the dead space in this clip") is True
    assert needs_transcript_hydration('cut out "hello world"') is True
    assert needs_transcript_hydration("remove where I say hello") is True
    assert needs_transcript_hydration("take out my ums and uhs") is True
    assert needs_transcript_hydration("split here at the playhead") is False


def test_needs_phrase_matching_is_narrower_than_hydration():
    assert needs_phrase_matching("remove long pauses") is False
    assert needs_phrase_matching('remove "bad take"') is True


def test_resolve_transcript_targets_prefers_selected_with_ref():
    ctx = IntentCompilerContext.model_validate(
        {
            "timelineId": "t1",
            "selectedClipId": "c2",
            "selectedTrackId": "tr1",
            "playheadTimeUs": 20_000_000,
            "clipsById": {
                "c1": {
                    "clip_id": "c1",
                    "track_id": "tr1",
                    "media_id": "m1",
                    "source_range": {"start": 0, "end": 5_000_000},
                    "timeline_range": {"start": 0, "end": 5_000_000},
                },
                "c2": {
                    "clip_id": "c2",
                    "track_id": "tr1",
                    "media_id": "m2",
                    "source_range": {"start": 0, "end": 5_000_000},
                    "timeline_range": {"start": 5_000_000, "end": 10_000_000},
                },
            },
            "orderedClipIdsByTrackId": {"tr1": ["c1", "c2"]},
            "transcriptContextsByClipId": {
                "c1": {"clipId": "c1", "transcriptId": "1"},
                "c2": {"clipId": "c2", "transcriptId": "2"},
            },
        }
    )
    targets = resolve_transcript_target_clip_ids(ctx, "remove pauses")
    assert targets == {"c2"}


def test_resolve_transcript_targets_all_clips_marker():
    ctx = IntentCompilerContext.model_validate(
        {
            "timelineId": "t1",
            "selectedClipId": "c1",
            "clipsById": {
                "c1": {
                    "clip_id": "c1",
                    "track_id": "tr1",
                    "media_id": "m1",
                    "source_range": {"start": 0, "end": 5_000_000},
                    "timeline_range": {"start": 0, "end": 5_000_000},
                },
                "c2": {
                    "clip_id": "c2",
                    "track_id": "tr1",
                    "media_id": "m2",
                    "source_range": {"start": 0, "end": 5_000_000},
                    "timeline_range": {"start": 5_000_000, "end": 10_000_000},
                },
            },
            "orderedClipIdsByTrackId": {"tr1": ["c1", "c2"]},
            "transcriptContextsByClipId": {
                "c1": {"clipId": "c1", "transcriptId": "1"},
                "c2": {"clipId": "c2", "transcriptId": "2"},
            },
        }
    )
    targets = resolve_transcript_target_clip_ids(ctx, "remove pauses from all clips")
    assert targets == {"c1", "c2"}
