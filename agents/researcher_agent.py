import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from tavily import TavilyClient
from datetime import date
from agents.llm import generate_with_retry

load_dotenv()

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

MODEL = "gemini-3.6-flash"
MAX_STEPS = 5  

SYSTEM_PROMPT = """You are a research agent with access to a web search tool.

Process:
1. Restate the research question in one sentence, then decide what you need to find out.
2. Use the search_web tool. Prefer 2-4 focused queries over one broad query.
   If results are weak or off-topic, search again with a better query.
3. When you have enough evidence, write concise factual notes.

Rules:
- Use ONLY information found in the search results.
- Cite every claim with [Source N], using the source numbers shown in the results.
- If something could not be found, say so instead of guessing.
- For questions about recent, latest or current events, include the current year in your
  search queries and prefer the newest sources. If the results look older than expected, say so.
"""

search_tool = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="search_web",
        description="Search the web for current information. Returns numbered sources with title, URL and content.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A focused search query."}
            },
            "required": ["query"],
        },
    )
])


def run_search(query: str, max_results: int = 5) -> list[dict]:
    response = tavily_client.search(query=query, max_results=max_results)
    return response["results"]


def register_sources(results: list[dict], sources: dict) -> str:
    url_to_id = {s["url"]: sid for sid, s in sources.items()}
    text = ""
    for r in results:
        if r["url"] not in url_to_id:
            sid = len(sources) + 1
            sources[sid] = {"title": r["title"], "url": r["url"], "content": r["content"]}
            url_to_id[r["url"]] = sid
        sid = url_to_id[r["url"]]
        text += f"Source {sid}: {r['title']}\nURL: {r['url']}\nContent: {r['content']}\n\n"
    return text


def research(topic: str) -> dict:
    start = time.time()
    sources: dict = {}
    trace: list[str] = []
    total_tokens = 0
    searches = 0
    notes = ""
    system_prompt = f"Today's date is {date.today():%B %d, %Y}.\n\n{SYSTEM_PROMPT}"

    tool_config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        tools=[search_tool],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    final_config = types.GenerateContentConfig(system_instruction=system_prompt)

    contents = [types.Content(role="user", parts=[types.Part(text=f"Research this: {topic}")])]

    for step in range(1, MAX_STEPS + 1):
        is_last = step == MAX_STEPS
        response = generate_with_retry(
            gemini_client,
            model=MODEL,
            contents=contents,
            config=final_config if is_last else tool_config,
        )

        if response.usage_metadata:
            total_tokens += response.usage_metadata.total_token_count or 0

        calls = response.function_calls

        if not calls:
            notes = response.text
            msg = f"step {step}: model wrote final notes"
            trace.append(msg)
            print(msg)
            break

        contents.append(response.candidates[0].content)

        reply_parts = []
        for call in calls:
            query = call.args["query"]
            t0 = time.time()
            results = run_search(query)
            result_text = register_sources(results, sources)
            searches += 1

            msg = f"step {step}: model chose to search -> '{query}' ({len(results)} results, {time.time() - t0:.1f}s)"
            trace.append(msg)
            print(msg)

            reply_parts.append(
                types.Part.from_function_response(name=call.name, response={"result": result_text})
            )

        contents.append(types.Content(role="user", parts=reply_parts))

    return {
        "topic": topic,
        "notes": notes,
        "sources": sources,
        "trace": trace,
        "stats": {
            "searches": searches,
            "total_tokens": total_tokens,
            "seconds": round(time.time() - start, 1),
        },
    }


if __name__ == "__main__":
    result = research("latest AI model releases this month")
    print("\n--- RESEARCH NOTES ---\n")
    print(result["notes"])
    print("\n--- STATS ---")
    print(result["stats"])
    print(f"Sources collected: {len(result['sources'])}")