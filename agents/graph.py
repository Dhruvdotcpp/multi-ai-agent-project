from langgraph.graph import StateGraph, START, END

from agents.state import AgentState
from agents.nodes import planner_node, researcher_node


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("planner", planner_node)
    builder.add_node("researcher", researcher_node)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "researcher")
    builder.add_edge("researcher", END)

    return builder.compile()