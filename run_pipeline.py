import sys

from agents.graph import build_graph
from agents.state import create_initial_state


def main():
    task = " ".join(sys.argv[1:]) or (
        "Evaluate whether multi-agent AI systems are worth using compared to a single LLM"
    )

    graph = build_graph()
    state = create_initial_state(task)

    printed = 0
    final = state

    for snapshot in graph.stream(state, stream_mode="values"):
        for line in snapshot["log"][printed:]:
            print(f"[log] {line}")
        printed = len(snapshot["log"])
        final = snapshot

    print("\n" + "=" * 60)
    print("TASK:", final["task"])

    print("\nSUBTASK STATUS:")
    for s in final["subtasks"]:
        print(f"  [{s.id}] {s.status.upper():7} {s.description}")

    total_tokens = 0
    total_seconds = 0.0
    for r in final["results"]:
        total_tokens += r.stats.get("total_tokens", 0)
        total_seconds += r.stats.get("seconds", 0)
        print(f"\n--- RESULT for subtask {r.subtask_id} ({len(r.source_details)} sources) ---")
        print(r.content[:500] + ("..." if len(r.content) > 500 else ""))

    print("\n" + "=" * 60)
    print(f"TOTAL: {len(final['results'])} results, {total_tokens} tokens, {total_seconds:.1f}s of research")


if __name__ == "__main__":
    main()