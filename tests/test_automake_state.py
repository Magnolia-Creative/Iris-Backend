from app.agent.automake.state import SESSION_GRAPH_STATE_VERSION, normalize_session_graph_state


def test_normalize_session_graph_state_adds_version_and_defaults():
    state = normalize_session_graph_state({"user_prompt": "make a cut"}, session_id=7)

    assert state["graph_state_version"] == SESSION_GRAPH_STATE_VERSION
    assert state["session_id"] == "7"
    assert state["waiting_for_user"] is False
    assert state["iteration_count"] == 0
    assert state["notes"] == []
    assert state["errors"] == []
    assert state["clips"] == []


def test_normalize_session_graph_state_preserves_existing_version():
    state = normalize_session_graph_state(
        {"graph_state_version": 99, "session_id": "old", "clips": [{"clip_id": "a"}]},
        session_id=8,
    )

    assert state["graph_state_version"] == 99
    assert state["session_id"] == "8"
    assert state["clips"] == [{"clip_id": "a"}]
