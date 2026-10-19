import random
import time

from google.genai import errors

RETRYABLE_CODES = {429, 500, 503, 504}


def generate_with_retry(client, max_attempts: int = 5, **kwargs):
    """
    Calls client.models.generate_content(**kwargs), retrying on temporary errors.
    Waits longer after each failure: ~2s, 4s, 8s, 16s (plus a little random jitter).
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return client.models.generate_content(**kwargs)
        except (errors.ServerError, errors.ClientError) as e:
            code = getattr(e, "code", None)

            if code not in RETRYABLE_CODES or attempt == max_attempts:
                raise

            wait = min(2 ** attempt, 30) + random.uniform(0, 1)
            print(f"[retry] Gemini returned {code}. Attempt {attempt}/{max_attempts}, retrying in {wait:.1f}s...")
            time.sleep(wait)