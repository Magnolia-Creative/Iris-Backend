import logging
from time import perf_counter
from typing import Any, Awaitable, Callable

from langchain_openai import ChatOpenAI

from app.config import settings
from app.agent.automake.build import build_session_graph
from app.agent.automake.state import SessionGraphState
from app.services.session_debug_store import initialize_session_debug, update_session_debug_state
from app.services.session_graph_state_store import persist_session_graph_state


logger = logging.getLogger(__name__)

SessionEventHandler = Callable[[dict[str, Any]], Awaitable[None]]

_session_store: dict[str, SessionGraphState] = {}


def _trace(message: str) -> None:
    print(f"[TRACE][runtime] {message}", flush=True)


def _elapsed_ms(started_at: float) -> int:
    return round((perf_counter() - started_at) * 1000)


def _default_llm() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        temperature=0,
    )


def set_session_state(session_id: str, state: SessionGraphState) -> None:
    _session_store[session_id] = state


def _is_finish_prompt(prompt: str) -> bool:
    return " ".join(prompt.lower().strip().split()) == "finish"


def _reset_llm_artifacts_for_reprompt(state: SessionGraphState) -> SessionGraphState:
    # Preserve prior planning artifacts so a re-prompt can modify only part of the current
    # draft instead of forcing a full restart. Nodes can still overwrite these fields.
    refreshed_state: SessionGraphState = {
        **state,
        "next_action": None,
    }
    return refreshed_state


async def run_session_until_pause(
    *,
    state: SessionGraphState,
    db: Any,
    event_handler: SessionEventHandler,
    llm: Any | None = None,
) -> SessionGraphState:
    started_at = perf_counter()
    graph = build_session_graph()
    session_id = state["session_id"]
    initialize_session_debug(session_id, state)
    logger.info("[runtime] Starting graph run session=%s", session_id)
    _trace(
        f"run_start session={session_id} iteration={state.get('iteration_count', 0)} "
        f"prompt={state.get('user_prompt', '')!r}"
    )

    latest_state = state
    last_status_fingerprint: tuple[str, str] | None = None
    config = {
        "configurable": {
            "db": db,
            "event_handler": event_handler,
            "llm": llm or _default_llm(),
        }
    }
    async for graph_state in graph.astream(state, config=config, stream_mode="values"):
        latest_state = graph_state
        update_session_debug_state(session_id, latest_state)
        status_message = str(latest_state.get("status_message") or "").strip()
        status_details = latest_state.get("status_details")
        if status_message:
            node_name = (
                str(status_details.get("node"))
                if isinstance(status_details, dict) and status_details.get("node")
                else "unknown"
            )
            fingerprint = (node_name, status_message)
            if fingerprint != last_status_fingerprint:
                logger.info(
                    "[runtime] Status update session=%s node=%s message=%r details=%s",
                    session_id,
                    node_name,
                    status_message,
                    status_details if isinstance(status_details, dict) else {},
                )
                await event_handler(
                    {
                        "type": "status_update",
                        "session_id": session_id,
                        "node": node_name,
                        "status_message": status_message,
                        "status_details": status_details if isinstance(status_details, dict) else {},
                    }
                )
                last_status_fingerprint = fingerprint
        await event_handler(
            {
                "type": "state_snapshot",
                "session_id": session_id,
                "state": latest_state,
            }
        )
        _trace(
            "graph_state_update "
            f"next_action={latest_state.get('next_action')} "
            f"waiting_for_user={latest_state.get('waiting_for_user')} "
            f"timeline_entries={len(latest_state.get('timeline', []))}"
        )

    _session_store[session_id] = latest_state
    update_session_debug_state(session_id, latest_state)
    persist_started_at = perf_counter()
    logger.info("[runtime] Persisting graph state session=%s", session_id)
    await persist_session_graph_state(db=db, session_id=int(session_id), state=latest_state)
    logger.info(
        "[runtime] Persisted graph state session=%s elapsed_ms=%s",
        session_id,
        _elapsed_ms(persist_started_at),
    )

    if latest_state.get("waiting_for_user"):
        logger.info(
            "[runtime] Session paused waiting_for_user session=%s elapsed_ms=%s",
            session_id,
            _elapsed_ms(started_at),
        )
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
        logger.info(
            "[runtime] Session completed session=%s elapsed_ms=%s",
            session_id,
            _elapsed_ms(started_at),
        )
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
    if _is_finish_prompt(prompt):
        return await approve_session_timeline(
            session_id=session_id,
            db=db,
            event_handler=event_handler,
        )

    state = _session_store.get(session_id)
    if state is None:
        raise KeyError(f"Unknown session_id: {session_id}")

    resumed_state: SessionGraphState = {
        **_reset_llm_artifacts_for_reprompt(state),
        "user_prompt": prompt,
        "waiting_for_user": False,
        "force_reconsider": True,
        "iteration_count": int(state.get("iteration_count", 0)) + 1,
        "notes": list(state.get("notes", []))
        + [
            "Resumed from user re-prompt.",
            "Preserve unaffected prior timeline/content unless the user asks for a full overhaul.",
        ],
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
        "force_reconsider": False,
        "notes": list(state.get("notes", [])) + ["User approved proposed timeline."],
    }
    _session_store[session_id] = approved_state
    update_session_debug_state(session_id, approved_state)
    await persist_session_graph_state(db=db, session_id=int(session_id), state=approved_state)
    logger.info("[runtime] Timeline approved session=%s", session_id)
    _trace(f"approved session={session_id}")
    await event_handler(
        {
            "type": "state_snapshot",
            "session_id": session_id,
            "state": approved_state,
        }
    )
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
