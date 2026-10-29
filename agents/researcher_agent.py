import os
import time
from datetime import date

from dotenv import load_dotenv
from tavily import TavilyClient

from agents.cache import cache_get, cache_set
from agents.llm import chat

from agents.textutil import clean_text

load_dotenv()

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

MAX_STEPS = 3             # max model calls per research run
MAX_RESULTS = 4           # search results per query
SOURCE_CHAR_LIMIT = 800   # characters of each source the model sees (saves tokens)

SYSTEM_PROMPT = """You are a research agent with access to a web search tool.

Process:
1. Restate the research question in one sentence.
2. Call search_web 2 or 3 times IN A SINGLE STEP, each time with a different angle on the question.
   Use plain natural-language queries. Do NOT use quotation marks, OR/AND operators or other search syntax.
3. Search again ONLY if the results are clearly off-topic or miss something essential.
4. Then write concise factual notes.

Rules:
- Use ONLY information found in the search results.
- Cite every claim with [Source N], using the source numbers shown in the results.
- If something could not be found, say so in a separate line that starts with "Gap:" instead of guessing.
- Never state a number, percentage, date or named result unless it appears in a source.
  Fewer well-supported claims are better than many weak ones.
- Every sentence must be traceable to a specific source. Do not add background knowledge from memory.
- For questions about recent, latest or current events, include the current year in your
  queries and prefer the newest sources. If the results look older than expected, say so."""

# The "menu" of tools the model is allowed to use (OpenAI-compatible format).
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "Search the web for current information. Returns numbered sources with title, URL and content.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A focused, plain-language search query."}
            },
            "required": ["query"],
        },
    },
}


def run_search(query: str) -> list[dict]:
    """Performs the web search via Tavily. Results are cached on disk and trimmed to save tokens."""
    key = {"q": query, "n": MAX_RESULTS}
    hit = cache_get("search", key)
    if hit is not None:
        return hit

    raw = tavily_client.search(query=query, max_results=MAX_RESULTS)["results"]
    results = [
        {"title": r["title"], "url": r["url"], "content": r["content"][:SOURCE_CHAR_LIMIT]}
        for r in raw
    ]
    cache_set("search", key, results)
    return results


def register_sources(results: list[dict], sources: dict) -> str:
    """
    Gives every new URL a permanent number (so 'Source 3' means the same page
    across multiple searches), stores the text in `sources`,
    and returns a text block to show the model.
    """
    url_to_id = {s["url"]: sid for sid, s in sources.items()}
    text = ""
    for r in results:
        if r["url"] not in url_to_id:
            sid = len(sources) + 1
            sources[sid] = {"title": clean_text(r["title"]), "url": r["url"], "content": clean_text(r["content"])}
            url_to_id[r["url"]] = sid
        sid = url_to_id[r["url"]]
        s = sources[sid]
        text += f"Source {sid}: {s['title']}\nURL: {s['url']}\nContent: {s['content']}\n\n"
    return text


def research(topic: str, feedback: str = "", avoid_queries: list = None) -> dict:
    """
    Tool-calling research agent. The model decides what to search and when to stop.
    Returns notes plus the sources, a decision trace, and run statistics.
    """
    start = time.time()
    sources: dict = {}
    trace: list[str] = []
    total_tokens = 0
    searches = 0
    llm_calls = 0
    notes = ""
    queries: list[str] = []
    avoid = {q.strip().lower() for q in (avoid_queries or [])}

    user_message = f"Research this: {topic}"
    if feedback:
        user_message += (
            "\n\nA fact-checker reviewed your previous notes on this question and found problems:\n"
            f"{feedback}\n\n"
        )
        if avoid_queries:
            user_message += (
                "Queries already tried (do NOT repeat them, use different angles):\n"
                + "\n".join(f"- {q}" for q in avoid_queries) + "\n\n"
            )
        user_message += (
            "Rewrite your COMPLETE notes so they answer the whole research question again. "
            "Drop every flagged claim unless a NEW search finds a source that states it, and do not "
            "restate a flagged claim in different words. You may also search for other evidence the "
            "question needs. Every number, percentage and date must appear in a source."
        )

    messages = [
        {"role": "system", "content": f"Today's date is {date.today():%B %d, %Y}.\n\n{SYSTEM_PROMPT}"},
        {"role": "user", "content": user_message},
    ]

    for step in range(1, MAX_STEPS + 1):
        is_last = step == MAX_STEPS
        if is_last:
            messages.append({
                "role": "user",
                "content": "Your search budget is used up. Write your final notes now, "
                           "using only the sources above.",
            })

        reply = chat(messages, tools=None if is_last else [SEARCH_TOOL])
        llm_calls += 1
        total_tokens += reply.tokens

        if not reply.tool_calls:
            notes = clean_text(reply.content or "")
            msg = f"step {step}: model wrote final notes [{reply.provider}]"
            trace.append(msg)
            print(msg)
            break

        # Keep the model's own tool request in the conversation, then answer each request.
        messages.append(reply.as_assistant_message())

        for call in reply.tool_calls:
            query = str(call.args.get("query", "")).strip() or topic

            if query.lower() in avoid:
                msg = f"step {step}: blocked a repeated query -> '{query}'"
                trace.append(msg)
                print(msg)
                messages.append({
                    "role": "tool", "tool_call_id": call.id,
                    "content": "BLOCKED: this query was already tried in an earlier attempt and returned "
                               "the same sources. Use a substantially different query.",
                })
                continue

            queries.append(query)
            t0 = time.time()
            results = run_search(query)
            result_text = register_sources(results, sources)
            searches += 1

            msg = f"step {step}: model chose to search -> '{query}' ({len(results)} results, {time.time() - t0:.1f}s)"
            trace.append(msg)
            print(msg)

            messages.append({"role": "tool", "tool_call_id": call.id, "content": result_text})

    return {
        "topic": topic,
        "notes": notes,
        "sources": sources,
        "trace": trace,
        "stats": {
            "searches": searches,
            "queries": queries,
            "llm_calls": llm_calls,
            "total_tokens": total_tokens,
            "seconds": round(time.time() - start, 1),
        },
    }

if __name__ == "__main__":
    result = research("What are the main benefits of using multi-agent AI systems?")
    print("\n--- RESEARCH NOTES ---\n")
    print(result["notes"])
    print("\n--- STATS ---")
    print(result["stats"])
    print(f"Sources collected: {len(result['sources'])}")