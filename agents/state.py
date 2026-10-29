import operator
from typing import Annotated, Literal, Optional, TypedDict

from pydantic import BaseModel, Field


class Subtask(BaseModel):
    id: int
    description: str
    assigned_agent: Literal["researcher", "analyst"]
    status: Literal["pending", "in_progress", "done", "failed"] = "pending"
    attempts: int = 0     
    feedback: str = ""     


class AgentResult(BaseModel):
    subtask_id: int
    agent: str
    content: str
    sources: list[str] = Field(default_factory=list)
    source_details: dict[int, dict] = Field(default_factory=dict)
    trace: list[str] = Field(default_factory=list)
    stats: dict = Field(default_factory=dict)


class ClaimCheck(BaseModel):
    claim: str
    cited_sources: list[int] = Field(default_factory=list)
    verdict: Literal["supported", "unsupported", "contradicted", "uncited"]
    evidence: str = ""
    evidence_found: bool = False   
    reason: str = ""


class VerificationResult(BaseModel):
    subtask_id: int
    approved: bool
    claims: list[ClaimCheck] = Field(default_factory=list)
    support_rate: float = 0.0
    invalid_citations: list[int] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    feedback: str = ""
    error: str = ""               


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