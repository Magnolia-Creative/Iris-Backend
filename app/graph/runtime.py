import logging
from typing import Any, Awaitable, Callable

from langchain_openai import ChatOpenAI

from app.config import settings
from app.graph.build import build_session_graph
from app.graph.state import SessionGraphState


logger = logging.getLogger(__name__)

SessionEventHandler = Callable[[dict[str, Any]], Awaitable[None]]

_session_store: dict[str, SessionGraphState] = {}


def _default_llm() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        temperature=0,
    )


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

    _session_store[session_id] = latest_state

    if latest_state.get("waiting_for_user"):
        logger.info("[runtime] Session paused waiting_for_user session=%s", session_id)
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


def get_session_state(session_id: str) -> SessionGraphState | None:
    return _session_store.get(session_id)
