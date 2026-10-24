"""
The ONLY place embeddings happen. Swap providers by editing this file.

Voyage is used because Anthropic has no embeddings endpoint — Claude generates
the answer, Voyage turns text into vectors. Two providers, one job each.
"""

import os
import time

import numpy as np
import voyageai
import voyageai.error

# voyage-4-lite: 1024 dims, ~$0.02/1M tokens, and the voyage-4 generation ships
# 200M free tokens per account — this project will not come close to that.
MODEL = os.environ.get("VOYAGE_MODEL", "voyage-4-lite")

# Voyage's free tier WITHOUT a payment method on file is capped at 3 requests/min
# and 10K tokens/min. That is a deliberate choice here: with no card attached the
# account cannot be billed at all, so we throttle to fit inside the free limits
# rather than buy our way out. Raise these via env if a card is ever added.
MAX_RPM = int(os.environ.get("VOYAGE_MAX_RPM", "3"))
MAX_TPM = int(os.environ.get("VOYAGE_MAX_TPM", "10000"))
BATCH_TOKENS = int(os.environ.get("VOYAGE_BATCH_TOKENS", "3000"))
BATCH_ITEMS = 128  # hard API ceiling on inputs per request

_client = None
_last_request = 0.0


def _client_once() -> voyageai.Client:
    global _client
    if _client is None:
        _client = voyageai.Client()  # reads VOYAGE_API_KEY
    return _client


def _est_tokens(text: str) -> int:
    """Rough token estimate. ~4 chars/token is close enough for budgeting."""
    return max(1, len(text) // 4)


def _batches(texts: list[str]) -> list[list[str]]:
    """Split into requests that respect BOTH the item cap and the token budget."""
    out, cur, cur_tokens = [], [], 0
    for t in texts:
        n = _est_tokens(t)
        if cur and (cur_tokens + n > BATCH_TOKENS or len(cur) >= BATCH_ITEMS):
            out.append(cur)
            cur, cur_tokens = [], 0
        cur.append(t)
        cur_tokens += n
    if cur:
        out.append(cur)
    return out


def _throttle(tokens: int) -> None:
    """
    Wait long enough to satisfy whichever limit binds.

    Requests/min needs 60/MAX_RPM seconds of spacing. Tokens/min needs this
    request's tokens to be worth `tokens * 60 / MAX_TPM` seconds. Taking the max
    keeps us legal on both without tracking a rolling window.
    """
    global _last_request
    required = max(60.0 / MAX_RPM, tokens * 60.0 / MAX_TPM)
    wait = required - (time.monotonic() - _last_request)
    if wait > 0:
        time.sleep(wait)
    _last_request = time.monotonic()


def embed_texts(texts: list[str], input_type: str, verbose: bool = False) -> np.ndarray:
    """
    Embed a list of texts.

    input_type matters and is not cosmetic: Voyage embeds a corpus chunk and a
    search query differently. Use "document" when indexing, "query" when asking.
    Getting this backwards quietly degrades retrieval.

    Returns an (n, dims) float32 array, L2-normalized so cosine similarity is
    just a dot product.
    """
    if input_type not in ("document", "query"):
        raise ValueError('input_type must be "document" or "query"')

    groups = _batches(texts)
    vectors: list[list[float]] = []

    for i, group in enumerate(groups, 1):
        tokens = sum(_est_tokens(t) for t in group)
        _throttle(tokens)

        for attempt in range(6):
            try:
                result = _client_once().embed(group, model=MODEL, input_type=input_type)
                vectors.extend(result.embeddings)
                break
            except voyageai.error.RateLimitError:
                backoff = 20.0 * (attempt + 1)
                if verbose:
                    print(f"    rate limited; waiting {backoff:.0f}s", flush=True)
                time.sleep(backoff)
                _last_request = time.monotonic()
        else:
            raise RuntimeError(
                "Voyage kept rate limiting. Lower VOYAGE_BATCH_TOKENS, or add a "
                "payment method to raise the limits (free tokens still apply)."
            )

        if verbose and len(groups) > 1:
            print(f"    embedded {i}/{len(groups)} batches "
                  f"({len(vectors)}/{len(texts)} chunks)", flush=True)

    arr = np.asarray(vectors, dtype=np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)
    return arr
