import json
import queue
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agents import llm
from agents.graph import build_graph
from agents.state import create_initial_state

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"
MAX_QUEUED = 10          # refuse new jobs when this many are already waiting
KEEP_IN_MEMORY = 100     # finished jobs loaded back from disk at start-up


# ---------- Data shapes ----------

class ResearchRequest(BaseModel):
    question: str = Field(min_length=10, max_length=500, description="The research question or task.")
    max_revisions: int = Field(default=2, ge=0, le=3, description="How many revision rounds the Verifier may request.")


class Job(BaseModel):
    id: str
    status: Literal["queued", "running", "done", "failed"]
    question: str
    max_revisions: int
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    log: list[str] = Field(default_factory=list)
    report: Optional[str] = None
    error: Optional[str] = None
    usage: Optional[dict] = None


class JobCreated(BaseModel):
    id: str
    status: str
    queue_position: int


class JobSummary(BaseModel):
    id: str
    status: str
    question: str
    created_at: str


# ---------- Job store and queue ----------

JOBS: dict = {}
LOCK = threading.Lock()
QUEUE: "queue.Queue[str]" = queue.Queue()
_worker_started = False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _save(job: Job) -> None:
    try:
        RUNS_DIR.mkdir(exist_ok=True)
        (RUNS_DIR / f"{job.id}.json").write_text(job.model_dump_json(indent=2), encoding="utf-8")
    except Exception as e:  # saving is a convenience; never let it break a job
        print(f"[api] could not save run {job.id}: {e}")


def _load_saved() -> None:
    if not RUNS_DIR.exists():
        return
    files = sorted(RUNS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)[-KEEP_IN_MEMORY:]
    for f in files:
        try:
            job = Job.model_validate_json(f.read_text(encoding="utf-8"))
            if job.status in ("done", "failed"):
                JOBS[job.id] = job
        except Exception as e:
            print(f"[api] skipped unreadable run file {f.name}: {e}")


def run_job(job_id: str) -> None:
    """Runs one research job to completion. Called by the single worker thread."""
    with LOCK:
        job = JOBS[job_id]
        job.status = "running"
        job.started_at = _now()
        question, max_revisions = job.question, job.max_revisions

    llm.reset_usage()   # safe: only one job runs at a time, so the counter belongs to this job
    try:
        graph = build_graph()
        state = create_initial_state(question, max_revisions=max_revisions)
        printed, final = 0, state

        # Each time a node finishes we receive the full state, so new log lines can be published live.
        for snapshot in graph.stream(state, stream_mode="values"):
            new_lines = snapshot["log"][printed:]
            printed = len(snapshot["log"])
            final = snapshot
            if new_lines:
                with LOCK:
                    job.log.extend(new_lines)

        report = final.get("final_report")
        with LOCK:
            job.report = report
            job.usage = llm.get_usage()
            if report:
                job.status = "done"
            else:
                job.status = "failed"
                job.error = "No report was produced. See the log for the reason."
    except Exception as e:
        with LOCK:
            job.status = "failed"
            job.error = f"{type(e).__name__}: {e}"
            job.usage = llm.get_usage()
    finally:
        with LOCK:
            job.finished_at = _now()
            snapshot_job = job.model_copy(deep=True)
        _save(snapshot_job)


def _worker() -> None:
    # One worker on purpose: free-tier LLM limits are shared, so jobs run one after another.
    while True:
        job_id = QUEUE.get()
        try:
            run_job(job_id)
        except Exception as e:
            print(f"[api] worker error on {job_id}: {e}")
        finally:
            QUEUE.task_done()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _worker_started
    _load_saved()
    if not _worker_started:
        threading.Thread(target=_worker, daemon=True, name="research-worker").start()
        _worker_started = True
    yield


# ---------- The API ----------

app = FastAPI(
    title="Multi-Agent Research Assistant API",
    description="Submit a research question, follow its progress, and fetch a cited, verified report.",
    version="1.0.0",
    lifespan=lifespan,
)

# Open CORS is for local development with a separate frontend. Restrict it before any real deployment.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    """Service status plus the state of each model in the LLM pool."""
    now = time.time()
    with LOCK:
        running = any(j.status == "running" for j in JOBS.values())
    return {
        "status": "ok",
        "queued_jobs": QUEUE.qsize(),
        "job_running": running,
        "models": [
            {"model": ep.label, "dead": ep.dead, "blocked_for_s": max(0, round(ep.blocked_until - now))}
            for ep in llm.POOL
        ],
    }


@app.post("/research", response_model=JobCreated, status_code=202)
def create_research(req: ResearchRequest):
    """Queue a research job. Returns immediately with a job id."""
    if QUEUE.qsize() >= MAX_QUEUED:
        raise HTTPException(status_code=429, detail="Too many jobs are waiting. Try again later.")
    job = Job(id=uuid.uuid4().hex[:12], status="queued", question=req.question.strip(),
              max_revisions=req.max_revisions, created_at=_now())
    with LOCK:
        JOBS[job.id] = job
        position = QUEUE.qsize() + 1
    QUEUE.put(job.id)
    return JobCreated(id=job.id, status=job.status, queue_position=position)


@app.get("/research", response_model=list[JobSummary])
def list_research(limit: int = 20):
    """Most recent jobs first."""
    with LOCK:
        jobs = sorted(JOBS.values(), key=lambda j: j.created_at, reverse=True)[: max(1, min(limit, 100))]
        return [JobSummary(id=j.id, status=j.status, question=j.question, created_at=j.created_at) for j in jobs]


@app.get("/research/{job_id}", response_model=Job)
def get_research(job_id: str):
    """Status, log and (when finished) the report of one job."""
    with LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job id.")
        return job.model_copy(deep=True)


@app.get("/research/{job_id}/events")
def research_events(job_id: str):
    """Server-Sent Events: each new log line as it appears, then a final 'end' event."""
    with LOCK:
        if job_id not in JOBS:
            raise HTTPException(status_code=404, detail="Unknown job id.")

    def stream():
        sent = 0
        while True:
            with LOCK:
                job = JOBS[job_id]
                new_lines, status = job.log[sent:], job.status
            for line in new_lines:
                yield f"data: {json.dumps({'log': line})}\n\n"
            sent += len(new_lines)
            if status in ("done", "failed") and not new_lines:
                yield f"event: end\ndata: {json.dumps({'status': status})}\n\n"
                return
            time.sleep(0.5)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})