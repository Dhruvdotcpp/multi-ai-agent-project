from datetime import date

from pydantic import BaseModel

from agents.llm import generate_json
from agents.state import Subtask

MAX_SUBTASKS = 3


class Plan(BaseModel):
    """The exact shape we require the LLM's answer to have."""
    rationale: str       # one or two sentences: why this breakdown
    subtasks: list[str]  # each item is one research question


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
- Ask questions that web articles and papers can plausibly answer. Do not demand statistical
  significance or exact numeric metrics unless they are essential to the task.
- Put your breakdown in "subtasks" (a list of strings) and a one or two sentence
  explanation in "rationale".

TASK: {task}"""

    parsed = generate_json([{"role": "user", "content": prompt}], Plan)

    # Instruct in the prompt, enforce in code.
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