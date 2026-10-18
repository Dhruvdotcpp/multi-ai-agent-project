from langgraph.graph import StateGraph, START, END

from agents.state import AgentState, Subtask, AgentResult, create_initial_state


def fake_planner(state: AgentState) -> dict:
    subtasks = [
        Subtask(id=1, description="Find benefits of multi-agent systems", assigned_agent="researcher"),
        Subtask(id=2, description="Find drawbacks of multi-agent systems", assigned_agent="researcher"),
    ]
    return {"subtasks": subtasks, "log": ["planner: created 2 subtasks"]}


def fake_researcher(state: AgentState) -> dict:
    new_results = []
    for st in state["subtasks"]:
        new_results.append(
            AgentResult(
                subtask_id=st.id,
                agent="researcher",
                content=f"(fake notes for: {st.description})",
                sources=["https://example.com"],
            )
        )
    return {"results": new_results, "log": [f"researcher: finished {len(new_results)} subtasks"]}


builder = StateGraph(AgentState)
builder.add_node("planner", fake_planner)
builder.add_node("researcher", fake_researcher)
builder.add_edge(START, "planner")
builder.add_edge("planner", "researcher")
builder.add_edge("researcher", END)
graph = builder.compile()

final_state = graph.invoke(create_initial_state("Evaluate multi-agent AI systems"))

print("TASK:", final_state["task"])

print("\nSUBTASKS:")
for s in final_state["subtasks"]:
    print(f"  [{s.id}] {s.description} -> {s.assigned_agent}")

print("\nRESULTS:")
for r in final_state["results"]:
    print(f"  subtask {r.subtask_id} ({r.agent}): {r.content}")

print("\nLOG:")
for line in final_state["log"]:
    print("  -", line)