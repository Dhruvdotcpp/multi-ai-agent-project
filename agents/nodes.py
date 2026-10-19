from agents.state import AgentState, AgentResult
from agents.planner_agent import plan
from agents.researcher_agent import research
from agents.state import AgentState, AgentResult, Subtask


def planner_node(state: AgentState) -> dict:
    try:
        subtasks, rationale = plan(state["task"])
    except Exception as e:
        subtasks = [Subtask(id=1, description=state["task"], assigned_agent="researcher")]
        rationale = f"Planner failed ({type(e).__name__}); falling back to a single subtask."

    return {
        "subtasks": subtasks,
        "log": [
            f"planner: created {len(subtasks)} subtasks. Rationale: {rationale}",
            *[f"planner: subtask {s.id} -> {s.description}" for s in subtasks],
        ],
    }

def researcher_node(state: AgentState) -> dict:
    updated_subtasks = []
    new_results = []
    log = []

    for st in state["subtasks"]:
        if st.assigned_agent != "researcher" or st.status != "pending":
            updated_subtasks.append(st)
            continue

        log.append(f"researcher: started subtask {st.id}")
        try:
            out = research(st.description)
            new_results.append(
                AgentResult(
                    subtask_id=st.id,
                    agent="researcher",
                    content=out["notes"],
                    sources=[s["url"] for s in out["sources"].values()],
                    source_details=out["sources"],
                    trace=out["trace"],
                    stats=out["stats"],
                )
            )
            updated_subtasks.append(st.model_copy(update={"status": "done"}))
            log.append(
                f"researcher: finished subtask {st.id} "
                f"({out['stats']['searches']} searches, {out['stats']['total_tokens']} tokens, "
                f"{out['stats']['seconds']}s)"
            )
        except Exception as e:
            updated_subtasks.append(st.model_copy(update={"status": "failed"}))
            log.append(f"researcher: subtask {st.id} FAILED ({type(e).__name__}: {e})")

    return {"subtasks": updated_subtasks, "results": new_results, "log": log}