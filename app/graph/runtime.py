import logging
from typing import Any, Awaitable, Callable

from langchain_openai import ChatOpenAI

from app.config import settings
from app.graph.build import build_session_graph
from app.graph.state import SessionGraphState
from app.services.session_graph_state_store import persist_session_graph_state


logger = logging.getLogger(__name__)

SessionEventHandler = Callable[[dict[str, Any]], Awaitable[None]]

_session_store: dict[str, SessionGraphState] = {}


def _trace(message: str) -> None:
    print(f"[TRACE][runtime] {message}", flush=True)


def _default_llm() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        temperature=0,
    )


def set_session_state(session_id: str, state: SessionGraphState) -> None:
    _session_store[session_id] = state


async def run_session_until_pause(
    *,
    state: SessionGraphState,
    db: Any,
    event_handler: SessionEventHandler,
    llm: Any | None = None,
) -> SessionGraphState:
    graph = build_session_graph()
    session_id = state["session_id"]
    logger.info("[runtime] Starting graph run session=%s", session_id)
    _trace(
        f"run_start session={session_id} iteration={state.get('iteration_count', 0)} "
        f"prompt={state.get('user_prompt', '')!r}"
    )

    latest_state = state
    config = {
        "configurable": {
            "db": db,
            "event_handler": event_handler,
            "llm": llm or _default_llm(),
        }
    }
    async for graph_state in graph.astream(state, config=config, stream_mode="values"):
        latest_state = graph_state
        _trace(
            "graph_state_update "
            f"next_action={latest_state.get('next_action')} "
            f"waiting_for_user={latest_state.get('waiting_for_user')} "
            f"timeline_entries={len(latest_state.get('timeline', []))}"
        )

    _session_store[session_id] = latest_state
    await persist_session_graph_state(db=db, session_id=int(session_id), state=latest_state)

    if latest_state.get("waiting_for_user"):
        logger.info("[runtime] Session paused waiting_for_user session=%s", session_id)
        _trace(f"pause_for_user session={session_id}")
        await event_handler(
            {
                "type": "timeline_update",
                "session_id": session_id,
                "timeline": latest_state.get("timeline", []),
                "timeline_notes": latest_state.get("timeline_notes", []),
            }
        )
        await event_handler(
            {
                "type": "waiting_for_user",
                "session_id": session_id,
                "project_id": latest_state.get("project_id"),
            }
        )
    else:
        logger.info("[runtime] Session completed session=%s", session_id)
        _trace(f"session_complete session={session_id}")
        await event_handler(
            {
                "type": "session_complete",
                "session_id": session_id,
                "project_id": latest_state.get("project_id"),
                "timeline": latest_state.get("timeline", []),
            }
        )

    return latest_state


async def resume_session_from_reprompt(
    *,
    session_id: str,
    prompt: str,
    db: Any,
    event_handler: SessionEventHandler,
    llm: Any | None = None,
) -> SessionGraphState:
    state = _session_store.get(session_id)
    if state is None:
        raise KeyError(f"Unknown session_id: {session_id}")

    resumed_state: SessionGraphState = {
        **state,
        "user_prompt": prompt,
        "waiting_for_user": False,
        "iteration_count": int(state.get("iteration_count", 0)) + 1,
        "notes": list(state.get("notes", [])) + ["Resumed from user re-prompt."],
    }
    logger.info("[runtime] Resuming session=%s iteration=%s", session_id, resumed_state["iteration_count"])
    _trace(
        f"resume session={session_id} iteration={resumed_state['iteration_count']} "
        f"new_prompt={prompt!r}"
    )
    await event_handler(
        {
            "type": "session_resumed",
            "session_id": session_id,
            "iteration_count": resumed_state["iteration_count"],
        }
    )
    return await run_session_until_pause(
        state=resumed_state,
        db=db,
        event_handler=event_handler,
        llm=llm,
    )


async def approve_session_timeline(
    *,
    session_id: str,
    db: Any,
    event_handler: SessionEventHandler,
) -> SessionGraphState:
    state = _session_store.get(session_id)
    if state is None:
        raise KeyError(f"Unknown session_id: {session_id}")

    approved_state: SessionGraphState = {
        **state,
        "waiting_for_user": False,
        "next_action": "finish",
        "notes": list(state.get("notes", [])) + ["User approved proposed timeline."],
    }
    _session_store[session_id] = approved_state
    await persist_session_graph_state(db=db, session_id=int(session_id), state=approved_state)
    logger.info("[runtime] Timeline approved session=%s", session_id)
    _trace(f"approved session={session_id}")
    await event_handler(
        {
            "type": "session_complete",
            "session_id": session_id,
            "project_id": approved_state.get("project_id"),
            "timeline": approved_state.get("timeline", []),
        }
    )
    return approved_state


def get_session_state(session_id: str) -> SessionGraphState | None:
    return _session_store.get(session_id)
