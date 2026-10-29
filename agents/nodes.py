from collections import Counter

from agents.state import AgentState, AgentResult, Subtask, VerificationResult
from agents.planner_agent import plan
from agents.researcher_agent import research
from agents.verifier_agent import verify_result
from agents.synthesizer_agent import synthesize


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

        attempt = st.attempts + 1
        what = "started" if attempt == 1 else f"re-researching (attempt {attempt})"
        log.append(f"researcher: {what} subtask {st.id}")
        try:
            out = research(st.description, feedback=st.feedback)
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
            updated_subtasks.append(st.model_copy(update={"status": "done", "attempts": attempt}))
            log.append(
                f"researcher: finished subtask {st.id} "
                f"({out['stats']['searches']} searches, {out['stats']['total_tokens']} tokens, "
                f"{out['stats']['seconds']}s)"
            )
        except Exception as e:
            updated_subtasks.append(st.model_copy(update={"status": "failed", "attempts": attempt}))
            log.append(f"researcher: subtask {st.id} FAILED ({type(e).__name__}: {e})")

    return {"subtasks": updated_subtasks, "results": new_results, "log": log}


def verifier_node(state: AgentState) -> dict:
    latest = {}
    for r in state["results"]:
        latest[r.subtask_id] = r

    n_results = Counter(r.subtask_id for r in state["results"])
    n_verified = Counter(v.subtask_id for v in state["verifications"])

    updated_subtasks, new_verifications, log = [], [], []
    sent_back = 0

    for st in state["subtasks"]:
        needs_check = st.status == "done" and st.id in latest and n_results[st.id] > n_verified[st.id]
        if not needs_check:
            updated_subtasks.append(st)
            continue

        try:
            v = verify_result(latest[st.id])
        except Exception as e:
            v = VerificationResult(subtask_id=st.id, approved=False, error=f"{type(e).__name__}: {e}")
            new_verifications.append(v)
            updated_subtasks.append(st)
            log.append(f"verifier: subtask {st.id} could NOT be verified ({v.error})")
            continue

        new_verifications.append(v)
        counts = dict(Counter(c.verdict for c in v.claims))
        summary = f"{len(v.claims)} claims, support rate {v.support_rate:.0%}, {counts}"

        if v.approved:
            updated_subtasks.append(st)
            log.append(f"verifier: subtask {st.id} APPROVED ({summary})")
        elif state["revision_count"] < state["max_revisions"]:
            updated_subtasks.append(st.model_copy(update={"status": "pending", "feedback": v.feedback}))
            sent_back += 1
            log.append(f"verifier: subtask {st.id} REJECTED, sent back for revision ({summary})")
        else:
            updated_subtasks.append(st)
            log.append(f"verifier: subtask {st.id} still FLAGGED, revision budget used up, kept with warnings ({summary})")

    revision_count = state["revision_count"]
    if sent_back:
        revision_count += 1
        log.append(f"verifier: revision round {revision_count} of {state['max_revisions']}")

    return {
        "subtasks": updated_subtasks,
        "verifications": new_verifications,
        "revision_count": revision_count,
        "log": log,
    }


def synthesizer_node(state: AgentState) -> dict:
    try:
        report, info = synthesize(state)
        return {"final_report": report, "log": [f"synthesizer: wrote the report ({info})"]}
    except Exception as e:
        return {"final_report": None, "log": [f"synthesizer: FAILED ({type(e).__name__}: {e})"]}


def route_after_verifier(state: AgentState) -> str:
    """Decides where to go next: back to the Researcher if anything was sent back, else write the report."""
    if any(s.status == "pending" for s in state["subtasks"]):
        return "researcher"
    return "synthesizer"