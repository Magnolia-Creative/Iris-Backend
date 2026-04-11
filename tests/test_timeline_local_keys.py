from app.graph.nodes.timeline_validator import _validate_timeline
from app.services.session_state_builder import build_initial_state_from_session_payload


def test_build_initial_state_includes_clip_local_keys():
    state = build_initial_state_from_session_payload(
        session_id="62",
        session_payload={
            "project_id": 11,
            "videos": [
                {
                    "clip_id": 106,
                    "local_key": "clip-local-106",
                    "transcript_id": 501,
                    "file_name": "clip.mov",
                    "mime_type": "video/quicktime",
                    "extension": ".mov",
                    "file_size_bytes": 1234,
                    "transcript_segments": [],
                    "transcript_full_text": "hello world",
                    "clip_meta": {"duration_seconds": 80.0},
                    "video_report": {},
                }
            ],
        },
        user_prompt="Split the clip in half.",
    )

    assert state["clips"][0]["clip_id"] == "106"
    assert state["clips"][0]["local_key"] == "clip-local-106"


def test_validate_timeline_attaches_local_keys_from_clip_state():
    result = _validate_timeline(
        {
            "session_id": "62",
            "project_id": 11,
            "user_prompt": "Split the clip in half.",
            "clips": [
                {
                    "clip_id": "106",
                    "local_key": "clip-local-106",
                    "metadata": {"duration_seconds": 80.0},
                }
            ],
            "timeline": [
                {
                    "clip_id": "106",
                    "in_sec": 0.0,
                    "out_sec": 40.0,
                    "rationale": "First half",
                },
                {
                    "clip_id": "106",
                    "in_sec": 40.0,
                    "out_sec": 80.0,
                    "rationale": "Second half",
                },
            ],
            "waiting_for_user": False,
            "iteration_count": 0,
            "notes": [],
            "errors": [],
        }
    )

    assert result.is_valid is True
    assert [entry.local_key for entry in result.normalized_timeline] == [
        "clip-local-106",
        "clip-local-106",
    ]
