from langgraph.graph import StateGraph, START, END

from agents.state import AgentState
from agents.nodes import (
    planner_node, researcher_node, verifier_node, synthesizer_node, route_after_verifier,
)


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("planner", planner_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("verifier", verifier_node)
    builder.add_node("synthesizer", synthesizer_node)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "researcher")
    builder.add_edge("researcher", "verifier")

    # The loop: after verification, either go back to the researcher or move on to the report.
    builder.add_conditional_edges(
        "verifier",
        route_after_verifier,
        {"researcher": "researcher", "synthesizer": "synthesizer"},
    )
    builder.add_edge("synthesizer", END)

    return builder.compile()