import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# .cache/ lives in the project root, no matter where you run Python from.
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"


def _enabled() -> bool:
    return os.getenv("LLM_CACHE", "1") == "1"


def _path(namespace: str, key_obj) -> Path:
    raw = json.dumps(key_obj, sort_keys=True, default=str)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
    return CACHE_DIR / namespace / f"{digest}.json"


def cache_get(namespace: str, key_obj):
    """Returns the saved value for this key, or None if there isn't one."""
    if not _enabled():
        return None
    path = _path(namespace, key_obj)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def cache_set(namespace: str, key_obj, value) -> None:
    if not _enabled():
        return
    path = _path(namespace, key_obj)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")