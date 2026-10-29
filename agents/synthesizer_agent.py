import re
from collections import Counter, defaultdict
from datetime import date
from urllib.parse import urlparse

from agents.llm import chat
from agents.state import AgentState
from agents.textutil import clean_text

MAX_CLAIMS_FOR_REPORT = 30   # keeps the writing prompt small

WRITER_PROMPT = """You are the report-writing agent of a research system. Write a report that answers
the TASK using ONLY the VERIFIED CLAIMS provided. Each claim was fact-checked against a source and
carries reference numbers.

Rules:
- Use only information contained in the verified claims. Do not add facts, numbers, names or dates
  from your own knowledge.
- After every sentence that states a fact, cite the reference numbers of the claims you used,
  like [2] or [1, 3]. Use only reference numbers that appear in the claims.
- Merge claims that say the same thing. Do not repeat yourself.
- Keep the qualifiers of the claims (for example "small", "sequential", "can", "may").
  Words like "all", "always" and "consistently" may only appear if the claim itself uses them.
- Each reference is tagged with a type and a website, like "3 (web: example.com)". A statement that rests
  only on (web) references, such as blogs and vendor pages, must be attributed, for example
  "According to example.com [3], ...", and never stated as settled fact.
- If the claims do not answer part of the task, say what is not covered instead of guessing.
- Write as a normal research report. Never use the words "claims", "verified" or "provided"; say
  "the sources reviewed" instead.
- Format (markdown): a "## Summary" section (3 to 5 sentences giving the overall answer), then a
  "## Findings" section with a short "###" heading per theme (no numbers in headings) and
  2 to 5 sentences under each.
- Do NOT write a sources section or a limitations section. Those are added automatically."""


ACADEMIC_SITES = (
    "arxiv.org", "openreview.net", "aclanthology.org", "nature.com", "science.org", "acm.org",
    "ieee.org", "springer.com", "sciencedirect.com", "pnas.org", "semanticscholar.org",
    "proceedings.mlr.press", "neurips.cc", "papers.nips.cc", "pubmed.ncbi.nlm.nih.gov",
)


def _host(url: str) -> str:
    host = urlparse(url).netloc.lower().split(":")[0]
    return host[4:] if host.startswith("www.") else host


def source_type(url: str) -> str:
    """A rough, rule-based label: 'academic' (papers, universities), 'official' (.gov) or 'web' (everything else)."""
    host = _host(url)
    if host.endswith(".edu") or ".edu." in host or ".ac." in host:
        return "academic"
    if any(host == site or host.endswith("." + site) for site in ACADEMIC_SITES):
        return "academic"
    if host.endswith(".gov") or ".gov." in host:
        return "official"
    return "web"


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def collect_verified(state: AgentState) -> dict:
    """
    Pools the SUPPORTED claims from every attempt of every subtask and pairs each with its sources.
    `results` and `verifications` only ever grow, so the k-th result of a subtask belongs to
    its k-th verification.
    """
    results, verifs = defaultdict(list), defaultdict(list)
    for r in state["results"]:
        results[r.subtask_id].append(r)
    for v in state["verifications"]:
        verifs[v.subtask_id].append(v)

    pooled, seen, summaries = [], set(), []

    for st in state["subtasks"]:
        pairs = list(zip(results[st.id], verifs[st.id]))
        attempts, best = [], None      # best = (score, index of the best attempt)

        for k, (r, v) in enumerate(pairs, start=1):
            n_supported = sum(c.verdict == "supported" for c in v.claims)
            attempts.append({
                "attempt": k, "support_rate": v.support_rate,
                "supported": n_supported, "total": len(v.claims), "error": v.error,
            })
            if v.error:
                continue

            score = (n_supported, v.support_rate)
            if best is None or score > best[0]:
                best = (score, k - 1)

            for c in v.claims:
                if c.verdict != "supported":
                    continue
                key = (st.id, _key(c.claim))
                if key in seen:
                    continue
                seen.add(key)
                sources = []
                for n in c.cited_sources:
                    s = r.source_details.get(n) or r.source_details.get(str(n))
                    if s:
                        sources.append({"title": s["title"], "url": s["url"]})
                pooled.append({
                    "subtask": st.id, "attempt": k, "claim": c.claim,
                    "evidence": c.evidence, "sources": sources,
                })

        gaps, excluded = [], []
        if best is not None:
            v_best = pairs[best[1]][1]
            gaps = list(v_best.gaps)
            excluded = [c for c in v_best.claims if c.verdict != "supported"]

        summaries.append({"subtask": st, "attempts": attempts, "gaps": gaps, "excluded": excluded})

    return {"pooled": pooled, "summaries": summaries}


def build_references(pooled: list) -> list:
    """Numbers every distinct source URL once ([1], [2], ...) and attaches ref numbers to each claim."""
    ref_ids, refs = {}, []
    for item in pooled:
        item["refs"] = []
        for s in item["sources"]:
            if s["url"] not in ref_ids:
                refs.append({**s, "type": source_type(s["url"]), "host": _host(s["url"])})
                ref_ids[s["url"]] = len(refs)
            n = ref_ids[s["url"]]
            if n not in item["refs"]:
                item["refs"].append(n)
    return refs


def _numbers(text: str) -> set:
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def audit_report(body: str, pooled: list, n_refs: int) -> list:
    """Checks the written report in plain code: valid citations, and no numbers the claims don't contain."""
    problems = []

    cited = set()
    for group in re.findall(r"\[([\d,\s]+)\]", body):
        cited.update(int(n) for n in re.findall(r"\d+", group))
    bad = sorted(n for n in cited if n < 1 or n > n_refs)
    if bad:
        problems.append(f"cites reference numbers that do not exist: {bad}")

    text = "\n".join(line for line in body.splitlines() if not line.lstrip().startswith("#"))
    text = re.sub(r"(?m)^\s*\d+[.)]\s+", "", text)      # list markers like "1."
    text = re.sub(r"\[[\d,\s]+\]", " ", text)           # citations
    allowed = _numbers(" ".join(p["claim"] for p in pooled))
    extra = sorted(_numbers(text) - allowed)
    if extra:
        problems.append(f"contains numbers that appear in no verified claim: {extra}")

    return problems


def _assemble(state: AgentState, body: str, data: dict, pooled: list, refs: list, problems: list) -> str:
    used = Counter(p["subtask"] for p in pooled)

    out = [
        "# Research report",
        "",
        f"**Task:** {state['task']}",
        "",
        f"_Generated {date.today():%B %d, %Y}. Every statement below comes from a claim that was checked "
        "against the text of its source. \"Verified\" means the source says it, not that the source is correct._",
        "",
        body.strip(),
        "",
        "---",
        "",
        "## Verification summary",
        "",
        "| Subtask | Status | Support rate by attempt | Verified claims used |",
        "|---|---|---|---|",
    ]

    for s in data["summaries"]:
        st = s["subtask"]
        if s["attempts"]:
            rates = " → ".join("n/a" if a["error"] else f"{a['support_rate']:.0%}" for a in s["attempts"])
        else:
            rates = "no result"
        out.append(f"| {st.id}. {st.description[:70]}... | {st.status} | {rates} | {used[st.id]} |")

    types = Counter(r["type"] for r in refs)
    if refs:
        out += ["", "Sources used by type: " + ", ".join(f"{t} {n}" for t, n in sorted(types.items())) + "."]

    gap_lines = []
    for s in data["summaries"]:
        st = s["subtask"]
        if not s["attempts"]:
            if st.status == "failed":
                gap_lines.append(f"- (subtask {st.id}) Research failed, so this part of the task is not covered.")
        elif all(a["error"] for a in s["attempts"]):
            gap_lines.append(
                f"- (subtask {st.id}) The research could not be verified (verification failed), "
                "so none of it is used. This part of the task is NOT covered."
            )
        elif used[st.id] == 0:
            gap_lines.append(f"- (subtask {st.id}) No statement from this research could be verified, so this part of the task is NOT covered.")
        for g in s["gaps"]:
            gap_lines.append(f"- (subtask {st.id}) {g}")
    if gap_lines:
        out += ["", "## Coverage gaps reported by the researcher", ""] + gap_lines

    excl_lines = []
    for s in data["summaries"]:
        for c in s["excluded"][:5]:
            excl_lines.append(f"- (subtask {s['subtask'].id}) [{c.verdict}] {c.claim}")
    if excl_lines:
        out += ["", "## Claims left out because they could not be verified", ""] + excl_lines

    out += ["", "## Sources", ""]
    out += [f"[{i}] ({r['type']}) {r['title']} - {r['url']}" for i, r in enumerate(refs, start=1)] or ["(none)"]

    out += ["", "## Report self-check", ""]
    if problems:
        out += [f"- WARNING: {p}" for p in problems]
    else:
        out += ["All citations point to listed sources, and every number in the findings appears in a verified claim."]

    return "\n".join(out) + "\n"


def synthesize(state: AgentState) -> tuple:
    """Writes the final report from verified claims only. Returns (markdown report, short info string)."""
    data = collect_verified(state)
    pooled = data["pooled"][:MAX_CLAIMS_FOR_REPORT]
    refs = build_references(pooled)

    if pooled:
        listing = "\n".join(
            f"- {p['claim']} [refs: "
            + ", ".join(f"{n} ({refs[n - 1]['type']}: {refs[n - 1]['host']})" for n in p["refs"])
            + "]"
            for p in pooled
        )
        messages = [
            {"role": "system", "content": WRITER_PROMPT},
            {"role": "user", "content": f"TASK: {state['task']}\n\nVERIFIED CLAIMS:\n{listing}"},
        ]
        body = clean_text(chat(messages).content or "").strip()
        problems = audit_report(body, pooled, len(refs))
    else:
        body = "No claim could be verified against its sources, so no findings are reported."
        problems = []

    report = _assemble(state, body, data, pooled, refs, problems)
    info = f"{len(pooled)} verified claims, {len(refs)} sources"
    if problems:
        info += f", {len(problems)} self-check warning(s)"
    return report, info