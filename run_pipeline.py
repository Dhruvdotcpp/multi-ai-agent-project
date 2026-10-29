import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from agents import llm
from agents.graph import build_graph
from agents.state import create_initial_state


def main():
    task = " ".join(sys.argv[1:]) or (
        "Evaluate whether multi-agent AI systems are worth using compared to a single LLM"
    )

    graph = build_graph()
    state = create_initial_state(task)
    llm.reset_usage()

    printed = 0
    final = state

    # stream_mode="values" gives us the FULL state after every node finishes.
    for snapshot in graph.stream(state, stream_mode="values"):
        for line in snapshot["log"][printed:]:
            print(f"[log] {line}")
        printed = len(snapshot["log"])
        final = snapshot

    print("\n" + "=" * 60)
    print("TASK:", final["task"])
    print(f"Revision rounds used: {final['revision_count']} of {final['max_revisions']}")

    print("\nSUBTASK STATUS:")
    for s in final["subtasks"]:
        print(f"  [{s.id}] {s.status.upper():7} (attempts: {s.attempts}) {s.description[:100]}")

    # Every attempt is kept in the state, so we can show how support changed from attempt to attempt.
    by_subtask = {}
    for v in final["verifications"]:
        by_subtask.setdefault(v.subtask_id, []).append(v)

    print("\nVERIFICATION (support rate per attempt):")
    for sid, vs in by_subtask.items():
        rates = " -> ".join("n/a" if v.error else f"{v.support_rate:.0%}" for v in vs)
        last = vs[-1]
        counts = dict(Counter(c.verdict for c in last.claims))
        print(f"  [{sid}] {rates}   (latest attempt: {counts}, gaps noted: {len(last.gaps)})")

    usage = llm.get_usage()
    print("\n" + "=" * 60)
    print(f"TOTAL (fresh LLM calls, all agents): {usage['calls']} calls, {usage['tokens']} tokens")

    report = final.get("final_report")
    if report:
        Path("reports").mkdir(exist_ok=True)
        path = Path("reports") / f"report_{datetime.now():%Y%m%d_%H%M%S}.md"
        path.write_text(report, encoding="utf-8")
        print(f"Report saved to: {path}")
        print("\n" + "=" * 60 + "\n")
        print(report)
    else:
        print("No report was produced (see the synthesizer line in the log above).")


if __name__ == "__main__":
    main()