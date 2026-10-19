import os
from datetime import date
from agents.llm import generate_with_retry
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

from agents.state import Subtask

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = "gemini-3.6-flash"
MAX_SUBTASKS = 4


class Plan(BaseModel):
    """The exact shape we require the LLM's answer to have."""
    rationale: str       
    subtasks: list[str]  


def plan(task: str) -> tuple[list[Subtask], str]:
    """
    Breaks a broad task into focused research subtasks.
    Returns (list of Subtask objects, rationale text).
    """
    prompt = f"""Today's date is {date.today():%B %d, %Y}.

    You are the planning agent in a multi-agent research system.
    Break the task below into 1 to {MAX_SUBTASKS} research subtasks.

    Rules:
    - Each subtask must be a specific, self-contained research question. It will be given
    to a researcher who sees NOTHING else, so include any needed context in the question.
    - Subtasks should cover different angles and must not overlap.
    - Do NOT create a subtask for writing the final report. A different agent does that.
    - If the task is already narrow, 1 or 2 subtasks is correct. Do not pad.
    - In "rationale", explain in one or two sentences why you chose this breakdown.

    TASK: {task}"""

    response = generate_with_retry(
        client,
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Plan,
        ),
    )

    parsed = Plan.model_validate_json(response.text)

    questions = [q.strip() for q in parsed.subtasks if q.strip()][:MAX_SUBTASKS]

    if not questions:
        questions = [task]

    subtasks = [
        Subtask(id=i, description=q, assigned_agent="researcher")
        for i, q in enumerate(questions, start=1)
    ]
    return subtasks, parsed.rationale


if __name__ == "__main__":
    subtasks, rationale = plan("Evaluate whether multi-agent AI systems are better than a single LLM")
    print("Rationale:", rationale)
    for s in subtasks:
        print(f"[{s.id}] {s.description}")