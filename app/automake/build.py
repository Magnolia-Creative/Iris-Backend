from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from app.automake.nodes.clip_cleanup import clip_cleanup_node
from app.automake.nodes.decision import decision_agent_node
from app.automake.nodes.hydrate_transcripts import hydrate_transcripts_node
from app.automake.nodes.timeline_planner import timeline_planner_node
from app.automake.nodes.timeline_validator import timeline_validator_node
from app.automake.state import SessionGraphState


def _route_from_decision(state: SessionGraphState) -> str:
    next_action = state.get("next_action")
    if next_action == "hydrate_transcripts":
        return "hydrate_transcripts"
    if next_action == "clip_cleanup":
        return "clip_cleanup"
    if next_action == "timeline_planner":
        return "timeline_planner"
    return "finish"


def _route_from_validator(state: SessionGraphState) -> str:
    return "pause" if state.get("waiting_for_user") else "decision_agent"


@lru_cache(maxsize=1)
def build_session_graph():
    graph = StateGraph(SessionGraphState)
    graph.add_node("decision_agent", decision_agent_node)
    graph.add_node("hydrate_transcripts", hydrate_transcripts_node)
    graph.add_node("clip_cleanup", clip_cleanup_node)
    graph.add_node("timeline_planner", timeline_planner_node)
    graph.add_node("timeline_validator", timeline_validator_node)

    graph.add_edge(START, "decision_agent")
    graph.add_conditional_edges(
        "decision_agent",
        _route_from_decision,
        {
            "hydrate_transcripts": "hydrate_transcripts",
            "clip_cleanup": "clip_cleanup",
            "timeline_planner": "timeline_planner",
            "finish": END,
        },
    )
    graph.add_edge("hydrate_transcripts", "decision_agent")
    graph.add_edge("clip_cleanup", "timeline_planner")
    graph.add_edge("timeline_planner", "timeline_validator")
    graph.add_conditional_edges(
        "timeline_validator",
        _route_from_validator,
        {
            "pause": END,
            "decision_agent": "decision_agent",
        },
    )
    return graph.compile()
