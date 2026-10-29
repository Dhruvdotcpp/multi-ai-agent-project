import re
import unicodedata
from difflib import SequenceMatcher
from typing import Literal

from pydantic import BaseModel

from agents.llm import generate_json
from agents.state import AgentResult, ClaimCheck, VerificationResult
from agents.textutil import clean_text

MAX_CLAIMS = 8             # claims checked per result (keeps token cost bounded)
APPROVE_THRESHOLD = 0.75   # share of claims that must be supported for approval
FUZZY_THRESHOLD = 0.85     # how much of a quoted piece of evidence must match the source
MIN_BLOCK = 8              # ignore accidental matches shorter than this many characters


# ---------- What we ask the LLM judge to return ----------

class JudgedClaim(BaseModel):
    claim: str
    cited_sources: list[int]
    verdict: Literal["supported", "unsupported", "contradicted"]
    evidence: str
    reason: str


class Judgement(BaseModel):
    claims: list[JudgedClaim]
    gaps: list[str] = []


JUDGE_PROMPT = """You are a strict fact-checking agent. Below are RESEARCH NOTES and the numbered SOURCES they cite.

Your job:
1. Extract up to {max_claims} of the most important factual claims made in the notes.
   State each claim as the notes state it. Do not generalize it or drop its qualifiers.
   Skip headings, restated questions, and generic filler.
2. For each claim, list the source numbers the notes cite for it ("cited_sources").
3. Judge each claim ONLY against the text of the sources it cites:
   - "supported": a cited source clearly states this.
   - "contradicted": a cited source states the opposite.
   - "unsupported": the cited sources do not say this, even if it might be true elsewhere.
4. A claim must not be broader than its evidence. If the source says "small multi-agent systems"
   or "sequential pipelines", a claim about "multi-agent systems" or "latency in general" is
   "unsupported". Check qualifiers and words like "all", "always", "can", "may" and "significantly".
5. For "supported" or "contradicted" claims, copy the exact sentence or phrase from the source
   into "evidence" (verbatim, under 40 words). For "unsupported", use an empty string.
6. "reason" is one short sentence.
7. Statements saying that information was NOT found or NOT provided (for example "the sources give
   no latency numbers") are not claims about the world. Do NOT put them in "claims".
   List each one, shortened, in "gaps".

Be strict about numbers, dates and names."""


# ---------- Deterministic checks (plain code, no LLM) ----------

def _norm(s: str) -> str:
    """Lowercase, fold look-alike characters (math letters, ligatures) to plain ones, keep letters/digits."""
    s = unicodedata.normalize("NFKC", s).lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def _get_source(sources: dict, sid: int):
    return sources.get(sid) or sources.get(str(sid))


def _coverage(evidence_norm: str, haystack_norm: str) -> float:
    """Share of the evidence text that appears in the source, allowing small differences."""
    sm = SequenceMatcher(None, evidence_norm, haystack_norm, autojunk=False)
    matched = sum(b.size for b in sm.get_matching_blocks() if b.size >= MIN_BLOCK)
    return matched / max(len(evidence_norm), 1)


def quote_in_sources(evidence: str, source_ids: list, sources: dict) -> bool:
    """True if the quoted evidence really appears in one of the cited sources."""
    parts = [_norm(p) for p in re.split(r"\.\.\.|\u2026", evidence)]  # allow "..." inside a quote
    parts = [p for p in parts if len(p) >= 10]
    if not parts:
        return False
    for sid in source_ids:
        src = _get_source(sources, sid)
        if not src:
            continue
        haystack = _norm(src["title"] + " " + src["content"])
        if all(p in haystack for p in parts):                              # exact match
            return True
        if all(_coverage(p, haystack) >= FUZZY_THRESHOLD for p in parts):  # near match (odd symbols)
            return True
    return False


def extract_citations(notes: str) -> set:
    """Finds every source number cited in the notes, e.g. [Source 2, Source 4] -> {2, 4}."""
    ids = set()
    for group in re.findall(r"\[([^\]]*Source[^\]]*)\]", notes, flags=re.I):
        ids.update(int(n) for n in re.findall(r"\d+", group))
    return ids


def _format_sources(sources: dict) -> str:
    return "\n\n".join(
        f"Source {sid}: {s['title']}\n{s['content']}"
        for sid, s in sorted(sources.items(), key=lambda kv: int(kv[0]))
    )


# ---------- The verifier ----------

def verify_result(result: AgentResult) -> VerificationResult:
    notes = clean_text(result.content)
    sources = result.source_details

    # Check 1 (code): does the text cite sources that don't exist?
    invalid = sorted(n for n in extract_citations(notes) if _get_source(sources, n) is None)

    # Check 2 (LLM): extract claims and judge each against its cited sources.
    prompt = (
        JUDGE_PROMPT.replace("{max_claims}", str(MAX_CLAIMS))
        + f"\n\nRESEARCH NOTES:\n{notes}\n\nSOURCES:\n{_format_sources(sources)}"
    )
    judgement = generate_json([{"role": "user", "content": prompt}], Judgement)

    # Check 3 (code): don't trust the judge. Confirm every quoted piece of evidence really exists.
    checks = []
    for jc in judgement.claims[:MAX_CLAIMS]:
        verdict, reason, evidence_found = jc.verdict, jc.reason, False

        if not jc.cited_sources:
            verdict = "uncited"
        elif verdict in ("supported", "contradicted"):
            evidence_found = quote_in_sources(jc.evidence, jc.cited_sources, sources)
            if not evidence_found:
                verdict = "unsupported"
                reason = (reason + " [Downgraded: quoted evidence not found in the cited source.]").strip()

        checks.append(ClaimCheck(
            claim=jc.claim, cited_sources=jc.cited_sources, verdict=verdict,
            evidence=jc.evidence, evidence_found=evidence_found, reason=reason,
        ))

    gaps = [g.strip() for g in judgement.gaps if g.strip()][:5]

    total = len(checks)
    n_supported = sum(c.verdict == "supported" for c in checks)
    n_contradicted = sum(c.verdict == "contradicted" for c in checks)
    support_rate = n_supported / total if total else 1.0

    issues = []
    if invalid:
        issues.append(f"Cites sources that do not exist: {invalid}")
    for c in checks:
        if c.verdict != "supported":
            issues.append(f"[{c.verdict}] {c.claim} ({c.reason})")
    if not total:
        issues.append("No checkable factual claims were extracted.")

    approved = n_contradicted == 0 and support_rate >= APPROVE_THRESHOLD and not invalid
    feedback = "" if approved else "Problems found in your notes:\n" + "\n".join(f"- {i}" for i in issues)

    return VerificationResult(
        subtask_id=result.subtask_id, approved=approved, claims=checks,
        support_rate=support_rate, invalid_citations=invalid, gaps=gaps,
        issues=issues, feedback=feedback,
    )