import operator
from typing import Annotated, Literal, Optional, TypedDict

from pydantic import BaseModel, Field


class Subtask(BaseModel):
    id: int
    description: str
    assigned_agent: Literal["researcher", "analyst"]
    status: Literal["pending", "in_progress", "done", "failed"] = "pending"


class AgentResult(BaseModel):
    subtask_id: int
    agent: str
    content: str
    sources: list[str] = Field(default_factory=list)


class VerificationResult(BaseModel):
    subtask_id: int
    approved: bool
    issues: list[str] = Field(default_factory=list)
    feedback: str = ""


class AgentState(TypedDict):
    task: str
    subtasks: list[Subtask]
    results: Annotated[list[AgentResult], operator.add]
    verifications: Annotated[list[VerificationResult], operator.add]
    revision_count: int
    max_revisions: int
    final_report: Optional[str]
    log: Annotated[list[str], operator.add]


def create_initial_state(task: str, max_revisions: int = 2) -> AgentState:
    return {
        "task": task,
        "subtasks": [],
        "results": [],
        "verifications": [],
        "revision_count": 0,
        "max_revisions": max_revisions,
        "final_report": None,
        "log": [f"Task received: {task}"],
    }