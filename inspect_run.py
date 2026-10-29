import sys

from agents.graph import build_graph
from agents.state import create_initial_state

SHOW_FLAGGED = 3     
SHOW_SUPPORTED = 3   


def main():
    task = " ".join(sys.argv[1:]) or (
        "Evaluate whether multi-agent AI systems are worth using compared to a single LLM"
    )

    # Same task + same day = every LLM call and search comes from the cache, so this costs 0 tokens.
    final = build_graph().invoke(create_initial_state(task))

    latest_v = {v.subtask_id: v for v in final["verifications"]}
    latest_r = {r.subtask_id: r for r in final["results"]}

    for subtask_id, v in latest_v.items():
        r = latest_r[subtask_id]
        flagged = [c for c in v.claims if c.verdict != "supported"]
        supported = [c for c in v.claims if c.verdict == "supported"]
        print("\n" + "=" * 70)
        print(f"SUBTASK {subtask_id}: support rate {v.support_rate:.0%} "
              f"({len(supported)} supported, {len(flagged)} flagged)")

        print("\n  FLAGGED CLAIMS (audit: is the claim really missing from the source?)")
        for c in flagged[:SHOW_FLAGGED]:
            print("  " + "-" * 66)
            print(f"  CLAIM   : {c.claim}")
            print(f"  VERDICT : {c.verdict}  (judge's reason: {c.reason})")
            print(f"  CITED   : {c.cited_sources}")
            for n in c.cited_sources[:2]:
                src = r.source_details.get(n) or r.source_details.get(str(n))
                if src:
                    print(f"    SOURCE {n} TEXT: {src['content']}")

        print("\n  SUPPORTED CLAIMS (audit: does the quote really back the claim?)")
        for c in supported[:SHOW_SUPPORTED]:
            print("  " + "-" * 66)
            print(f"  CLAIM    : {c.claim}")
            print(f"  EVIDENCE : {c.evidence}")
            print(f"  CITED    : {c.cited_sources}")


if __name__ == "__main__":
    main()