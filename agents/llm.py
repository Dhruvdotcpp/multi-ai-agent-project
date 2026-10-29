import json
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from typing import Optional

from dotenv import load_dotenv
from openai import (
    OpenAI,
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)
from pydantic import BaseModel, ValidationError

from agents.cache import cache_get, cache_set

load_dotenv()


# ---------- Data shapes ----------

@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class Reply:
    content: Optional[str]
    tool_calls: list = field(default_factory=list)
    tokens: int = 0
    provider: str = ""

    def as_assistant_message(self) -> dict:
        """The message to append to the conversation so the model remembers its own tool request."""
        msg = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": json.dumps(c.args)},
                }
                for c in self.tool_calls
            ]
        return msg


class AllProvidersExhausted(RuntimeError):
    pass

# Run-wide usage counter (fresh API calls only; cache hits cost nothing).
USAGE = {"calls": 0, "tokens": 0}


def reset_usage() -> None:
    USAGE["calls"] = 0
    USAGE["tokens"] = 0


def get_usage() -> dict:
    return dict(USAGE)

# ---------- The provider pool ----------

@dataclass
class Endpoint:
    provider: str
    base_url: str
    api_key: str
    model: str
    blocked_until: float = 0.0   # skip this endpoint until this time (unix seconds)
    dead: bool = False           # True if the model doesn't exist or the key is rejected
    _client: Optional[OpenAI] = None

    @property
    def label(self) -> str:
        return f"{self.provider}/{self.model}"

    def client(self) -> OpenAI:
        if self._client is None:
            # max_retries=0: WE handle retries, so the SDK doesn't silently burn extra requests.
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key,
                                  max_retries=0, timeout=60)
        return self._client


def _build_pool() -> list:
    pool = []
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        models = os.getenv("GROQ_MODELS", "openai/gpt-oss-120b,openai/gpt-oss-20b")
        for m in models.split(","):
            if m.strip():
                pool.append(Endpoint("groq", "https://api.groq.com/openai/v1", groq_key, m.strip()))
    or_key = os.getenv("OPENROUTER_API_KEY")
    if or_key:
        models = os.getenv("OPENROUTER_MODELS", "meta-llama/llama-3.3-70b-instruct:free")
        for m in models.split(","):
            if m.strip():
                pool.append(Endpoint("openrouter", "https://openrouter.ai/api/v1", or_key, m.strip()))
    if not pool:
        raise RuntimeError("No LLM provider configured. Add GROQ_API_KEY to your .env file.")
    return pool


POOL = _build_pool()


# ---------- Reading rate-limit messages ----------

def _parse_wait(msg: str) -> Optional[float]:
    """Turns 'Please try again in 3m43.2s' into seconds. Returns None if no hint is present."""
    m = re.search(r"try again in\s+([0-9hms.]+)", msg)
    if not m:
        return None
    total, found = 0.0, False
    for num, unit in re.findall(r"([0-9.]+)(ms|h|m|s)", m.group(1)):
        found = True
        total += float(num) * {"h": 3600, "m": 60, "s": 1, "ms": 0.001}[unit]
    return total if found else None


def _is_daily_limit(msg: str) -> bool:
    low = msg.lower()
    return any(t in low for t in ("per day", "(tpd)", "(rpd)", "per-day", "perday"))


# ---------- One call to one endpoint ----------

def _call_endpoint(ep: Endpoint, messages, tools, json_mode, temperature) -> Reply:
    kwargs = dict(model=ep.model, messages=messages, temperature=temperature)
    if tools:
        kwargs["tools"] = tools
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if "gpt-oss" in ep.model:
        # Reasoning tokens count against free limits, so keep reasoning short.
        kwargs["extra_body"] = {"reasoning_effort": "low"}

    resp = ep.client().chat.completions.create(**kwargs)
    msg = resp.choices[0].message

    calls = []
    for tc in (msg.tool_calls or []):
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        calls.append(ToolCall(id=tc.id, name=tc.function.name, args=args))

    tokens = resp.usage.total_tokens if resp.usage else 0
    return Reply(content=msg.content, tool_calls=calls, tokens=tokens, provider=ep.label)


# ---------- Try the pool in order ----------

MAX_TOTAL_WAIT = 240   # seconds ONE call may spend waiting out per-minute limits before giving up


def _call_pool(messages, tools, json_mode, temperature) -> Reply:
    last_error = None
    waited = 0.0
    skip = set()                  # endpoints to avoid for THIS call only
    fails = defaultdict(int)      # temporary failures per endpoint, for THIS call only

    while True:
        now = time.time()
        alive = [ep for ep in POOL if not ep.dead and ep.label not in skip]
        ready = [ep for ep in alive if now >= ep.blocked_until]

        if not ready:
            if not alive:
                break
            # Everything usable is blocked right now. If the shortest block is brief, wait it out.
            pause = min(ep.blocked_until for ep in alive) - now
            if pause > 90 or waited + pause > MAX_TOTAL_WAIT:
                break
            pause += 1
            print(f"[llm] all models are at a per-minute limit, waiting {pause:.0f}s")
            time.sleep(pause)
            waited += pause
            continue

        ep = ready[0]
        try:
            return _call_endpoint(ep, messages, tools, json_mode, temperature)

        except RateLimitError as e:
            msg = str(e)
            wait = _parse_wait(msg)
            last_error = e
            if _is_daily_limit(msg) or (wait is not None and wait > 90):
                ep.blocked_until = time.time() + (wait or 3600)
                print(f"[llm] {ep.label}: daily/long limit reached, skipping it for now.")
            else:
                ep.blocked_until = time.time() + (wait or 5)
                print(f"[llm] {ep.label}: per-minute limit, it will be free again in about {wait or 5:.0f}s")

        except (NotFoundError, AuthenticationError, PermissionDeniedError) as e:
            ep.dead = True
            last_error = e
            print(f"[llm] {ep.label}: unusable ({type(e).__name__}). Check the model name / API key. Skipping it.")

        except BadRequestError as e:
            last_error = e
            fails[ep.label] += 1
            if "tool_use_failed" in str(e) and fails[ep.label] < 3:
                print(f"[llm] {ep.label}: malformed tool call, retrying (attempt {fails[ep.label]}/3)")
                continue
            raise

        except (APIConnectionError, InternalServerError) as e:
            last_error = e
            fails[ep.label] += 1
            if fails[ep.label] >= 3:
                skip.add(ep.label)
            else:
                pause = 2 ** fails[ep.label]
                print(f"[llm] {ep.label}: temporary error ({type(e).__name__}), retrying in {pause}s")
                time.sleep(pause)

        except APIStatusError as e:
            last_error = e
            if e.status_code == 413:
                skip.add(ep.label)
                print(f"[llm] {ep.label}: request too large for this model's limits, trying the next one.")
            else:
                raise

    raise AllProvidersExhausted(
        f"Every provider/model in the pool is exhausted or unavailable. Last error: {last_error}"
    )

# ---------- Public functions the agents use ----------

def chat(messages: list, tools: Optional[list] = None, json_mode: bool = False,
         temperature: float = 0.2) -> Reply:
    """Send a conversation, get a Reply. Identical requests are answered from the disk cache."""
    key = {"messages": messages, "tools": tools, "json": json_mode}

    hit = cache_get("llm", key)
    if hit is not None:
        return Reply(
            content=hit["content"],
            tool_calls=[ToolCall(**t) for t in hit["tool_calls"]],
            tokens=0,
            provider="cache",
        )

    reply = _call_pool(messages, tools, json_mode, temperature)
    USAGE["calls"] += 1
    USAGE["tokens"] += reply.tokens
    cache_set("llm", key, {
        "content": reply.content,
        "tool_calls": [asdict(t) for t in reply.tool_calls],
    })
    return reply


def generate_json(messages: list, model_cls: type[BaseModel], attempts: int = 2):
    """Gets a reply as JSON and validates it against a Pydantic class."""
    schema = json.dumps(model_cls.model_json_schema())
    msgs = [{
        "role": "system",
        "content": "Respond with ONLY a JSON object matching this JSON schema. "
                   f"No markdown, no commentary.\n{schema}",
    }] + messages

    last_error = None
    for _ in range(attempts):
        reply = chat(msgs, json_mode=True)
        try:
            return model_cls.model_validate_json(reply.content or "")
        except ValidationError as e:
            last_error = e
            msgs = msgs + [
                {"role": "assistant", "content": reply.content or ""},
                {"role": "user", "content": f"That JSON was invalid: {e}. Return corrected JSON only."},
            ]
    raise last_error