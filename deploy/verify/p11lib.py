#!/usr/bin/env python3
"""Shared helpers for the P11 rack experiment kit (T1-T4 harnesses).

Design constraints honored here:

  * stdlib only -- the rack venv is uv-managed and is NOT guaranteed to carry
    requests/numpy. urllib + json + argparse only.
  * The rack's network is flaky: every HTTP call gets an explicit timeout and a
    BOUNDED retry budget, and every failure is classified as "network" or
    "server" so a report can never blame the model for a dropped socket.
  * Long generations stream, so a stalled connection is detected by an idle
    (per-recv) socket timeout rather than by a wall-clock read timeout that
    would also fire on a legitimately slow 200k prefill.
  * Prompts are built ONCE, cached to disk as raw token-ID arrays, and sent as
    token IDs (``prompt`` accepts ``list[int]``; verified in
    vllm/entrypoints/openai/completion/protocol.py:49-55). Configs B and C
    therefore see byte-identical inputs, and every harness records the SHA-256
    of what it sent so the analysis can PROVE it.

Verified facts this module depends on (all re-checked against
this fork):

  * ``CompletionRequest.return_token_ids`` (protocol.py:159) makes the server
    return ``choices[].token_ids`` (non-streaming: full array; streaming: the
    per-chunk delta). This is how T1 gets TOKEN IDS without a local tokenizer.
  * ``UsageInfo.prompt_tokens_details.cached_tokens``
    (vllm/entrypoints/openai/engine/protocol.py:105-119) EXISTS, but is only
    populated when the server was started with ``--enable-prompt-tokens-details``
    (cli_args.py:132, default False) AND, in streaming mode, only on the final
    usage chunk, which itself requires ``stream_options.include_usage=true``.
    Harnesses fall back to a /metrics delta when the field is absent.
  * Prometheus counters are ``vllm:prefix_cache_queries`` and
    ``vllm:prefix_cache_hits`` (vllm/v1/metrics/loggers.py:584-601), plus
    ``vllm:prompt_tokens_cached`` (:695). prometheus_client appends ``_total``
    to counters on the wire, so lookups here accept both spellings.
  * ``POST /reset_prefix_cache`` exists but only when the server runs with
    ``VLLM_SERVER_DEV_MODE=1`` (api_server.py:231). Probed, never assumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

# --------------------------------------------------------------------------
# Defaults. Everything is overridable by flag or environment variable.
# --------------------------------------------------------------------------

DEFAULT_BASE_URL = os.environ.get("P11_BASE_URL", "http://127.0.0.1:8000")
DEFAULT_SEED = int(os.environ.get("P11_SEED", "20260804"))
DEFAULT_CONNECT_TIMEOUT = float(os.environ.get("P11_CONNECT_TIMEOUT", "10"))
DEFAULT_READ_TIMEOUT = float(os.environ.get("P11_READ_TIMEOUT", "120"))
DEFAULT_STALL_TIMEOUT = float(os.environ.get("P11_STALL_TIMEOUT", "120"))
DEFAULT_RETRIES = int(os.environ.get("P11_RETRIES", "3"))
DEFAULT_GEN_RETRIES = int(os.environ.get("P11_GEN_RETRIES", "2"))
DEFAULT_BACKOFF = float(os.environ.get("P11_BACKOFF", "2.0"))
DEFAULT_METRICS_SETTLE = float(os.environ.get("P11_METRICS_SETTLE", "3.0"))

# The scheduler block size at dcp=4 under PROFILE-CACHE. Hits are aligned to it.
DEFAULT_SCHED_BLOCK = int(os.environ.get("P11_SCHED_BLOCK", "1024"))

RETRYABLE_STATUS = (500, 502, 503, 504, 429)


# --------------------------------------------------------------------------
# Errors -- the "network vs server" split every report has to preserve.
# --------------------------------------------------------------------------


class P11Error(Exception):
    """Base class. ``kind`` is always one of network|server|protocol."""

    kind = "protocol"

    def as_dict(self) -> dict:
        return {"error_kind": self.kind, "error": str(self)}


class P11NetworkError(P11Error):
    """Socket-level failure: refused, timed out, reset, DNS, or a stalled stream.

    The server may be perfectly healthy; nothing about the model is implied.
    """

    kind = "network"


class P11StallError(P11NetworkError):
    """A stream produced no bytes for longer than the idle timeout."""


class P11ServerError(P11Error):
    """The server answered with a non-2xx status. The model/engine is implicated."""

    kind = "server"

    def __init__(self, status: int, body: str, url: str):
        super().__init__(f"HTTP {status} from {url}: {body[:500]}")
        self.status = status
        self.body = body
        self.url = url

    def as_dict(self) -> dict:
        d = super().as_dict()
        d["status"] = self.status
        return d


class P11ProtocolError(P11Error):
    """The response parsed as HTTP but not as the shape we require."""


# --------------------------------------------------------------------------
# Deterministic RNG. Explicitly NOT random.Random: this is xorshift64* written
# out longhand so the same seed yields the same prompt bytes on any CPython,
# any platform, forever. B and C must see identical prompts even if they are
# built weeks apart on a reinstalled rack.
# --------------------------------------------------------------------------

_MASK64 = (1 << 64) - 1


class Rng:
    def __init__(self, seed: int):
        s = (int(seed) ^ 0x9E3779B97F4A7C15) & _MASK64
        self._s = s if s else 0x2545F4914F6CDD1D

    def next_u64(self) -> int:
        x = self._s
        x ^= x >> 12
        x &= _MASK64
        x ^= (x << 25) & _MASK64
        x ^= x >> 27
        self._s = x
        return (x * 0x2545F4914F6CDD1D) & _MASK64

    def below(self, n: int) -> int:
        if n <= 0:
            raise ValueError("below(n) needs n > 0")
        return self.next_u64() % n

    def choice(self, seq):
        return seq[self.below(len(seq))]


def seed_of(*parts) -> int:
    """Stable integer seed derived from arbitrary parts (str/int/float)."""
    payload = "|".join(repr(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


# A closed vocabulary keeps the filler text stable and tokenizer-friendly.
_WORDS = (
    "system module packet buffer index vector matrix cluster kernel thread "
    "signal channel border latency window segment anchor cursor pointer offset "
    "region tunnel bridge gateway harbor lantern meadow orchard granite basalt "
    "quartz copper silver cobalt indigo crimson amber olive maroon violet "
    "north south east west river valley canyon plateau summit glacier "
    "morning evening winter summer autumn harvest lantern beacon compass ledger "
    "archive dossier bulletin protocol schedule inventory manifest register "
    "trace sample metric budget quota policy routine fragment sequence lattice "
    "prism filament conduit reservoir aqueduct terrace pavilion corridor alcove "
    "rotation cadence tempo interval gradient contour texture pattern motif "
    "wander gather refine measure record report resolve confirm observe adjust "
    "quiet steady narrow broad distant nearby ancient modern hollow solid "
    "beneath beyond across within toward against between through around "
    "the of and to in for with on at by from as into over under"
).split()


def filler_text(n_words: int, seed: int) -> str:
    """Deterministic prose-shaped filler. Sentence breaks keep the tokenizer
    from collapsing everything into one long merge run."""
    rng = Rng(seed)
    out = []
    since_break = 0
    for _ in range(n_words):
        out.append(rng.choice(_WORDS))
        since_break += 1
        if since_break >= 8 + rng.below(9):
            out[-1] = out[-1] + "."
            since_break = 0
    return " ".join(out)


def sha256_ids(ids) -> str:
    h = hashlib.sha256()
    for t in ids:
        h.update(int(t).to_bytes(4, "big"))
    return h.hexdigest()


# --------------------------------------------------------------------------
# HTTP client
# --------------------------------------------------------------------------


class Client:
    """Minimal OpenAI-compatible client with bounded retries and stall detection.

    Every method raises a P11Error subclass on failure; nothing returns a
    sentinel that a caller could mistake for a measurement.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        read_timeout: float = DEFAULT_READ_TIMEOUT,
        stall_timeout: float = DEFAULT_STALL_TIMEOUT,
        retries: int = DEFAULT_RETRIES,
        gen_retries: int = DEFAULT_GEN_RETRIES,
        backoff: float = DEFAULT_BACKOFF,
        verbose: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.stall_timeout = stall_timeout
        self.retries = max(1, retries)
        self.gen_retries = max(1, gen_retries)
        self.backoff = backoff
        self.verbose = verbose
        self._model = None
        self._cache: dict = {}
        self._opener = urllib.request.build_opener()

    # -- low level ---------------------------------------------------------

    def _url(self, path: str) -> str:
        return self.base_url + path

    def _open(self, path: str, payload=None, timeout=None, method=None):
        url = self._url(path)
        data = None
        headers = {"Accept": "application/json", "Accept-Encoding": "identity"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            return self._opener.open(req, timeout=timeout or self.read_timeout)
        except urllib.error.HTTPError as exc:  # server answered, badly
            try:
                body = exc.read().decode("utf-8", "replace")
            except Exception:  # noqa: BLE001 - body is best-effort diagnostics
                body = "<unreadable body>"
            raise P11ServerError(exc.code, body, url) from exc
        except urllib.error.URLError as exc:
            raise P11NetworkError(f"{url}: {exc.reason}") from exc
        except (socket.timeout, TimeoutError) as exc:
            raise P11NetworkError(f"{url}: socket timeout") from exc
        except (ConnectionError, ssl.SSLError, OSError) as exc:
            raise P11NetworkError(f"{url}: {exc}") from exc

    def _retrying(self, fn, budget: int, what: str):
        """Run ``fn`` up to ``budget`` times. Returns (result, attempts, errors)."""
        errors = []
        for attempt in range(1, budget + 1):
            try:
                return fn(), attempt, errors
            except P11ServerError as exc:
                errors.append(exc.as_dict())
                if exc.status not in RETRYABLE_STATUS or attempt == budget:
                    raise
            except P11NetworkError as exc:
                errors.append(exc.as_dict())
                if attempt == budget:
                    raise
            delay = self.backoff * (2 ** (attempt - 1))
            self.log(f"retry {attempt}/{budget - 1} for {what} after {delay:.1f}s")
            time.sleep(delay)
        raise P11ProtocolError("unreachable retry exit")  # pragma: no cover

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"[p11] {msg}", file=sys.stderr, flush=True)

    # -- plain JSON --------------------------------------------------------

    def get_json(self, path: str, timeout=None):
        def once():
            with self._open(path, timeout=timeout or self.connect_timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))

        result, _, _ = self._retrying(once, self.retries, f"GET {path}")
        return result

    def post_json(self, path: str, payload, timeout=None, budget=None):
        def once():
            with self._open(path, payload, timeout=timeout or self.read_timeout) as r:
                return json.loads(r.read().decode("utf-8"))

        result, attempts, _ = self._retrying(
            once, budget or self.retries, f"POST {path}"
        )
        return result, attempts

    def get_text(self, path: str, timeout=None) -> str:
        def once():
            with self._open(path, timeout=timeout or self.connect_timeout) as resp:
                return resp.read().decode("utf-8", "replace")

        result, _, _ = self._retrying(once, self.retries, f"GET {path}")
        return result

    # -- server introspection ---------------------------------------------

    def model(self) -> str:
        """Model name from GET /v1/models. NEVER hardcoded (rack rule)."""
        if self._model is None:
            data = self.get_json("/v1/models")
            entries = data.get("data") or []
            if not entries:
                raise P11ProtocolError("/v1/models returned no models")
            self._model = entries[0]["id"]
            self.log(f"model={self._model}")
        return self._model

    def health(self) -> bool:
        try:
            self.get_text("/health")
            return True
        except P11Error:
            return False

    def tokenize(self, text: str, add_special_tokens: bool = True) -> list:
        payload = {
            "model": self.model(),
            "prompt": text,
            "add_special_tokens": add_special_tokens,
        }
        data, _ = self.post_json("/tokenize", payload, timeout=self.read_timeout)
        toks = data.get("tokens")
        if not isinstance(toks, list):
            raise P11ProtocolError(f"/tokenize returned no token list: {data!r}")
        return toks

    def tokenize_chat(self, messages, thinking: bool = False) -> list:
        """Token IDs for a chat request as the server would build them,
        including the generation prompt. Used to drive /v1/completions with a
        chat-templated, thinking-DISABLED prompt while keeping byte-exact
        control of the token array (which /v1/chat/completions does not give)."""
        payload = {
            "model": self.model(),
            "messages": messages,
            "add_generation_prompt": True,
            "chat_template_kwargs": {"thinking": thinking},
        }
        data, _ = self.post_json("/tokenize", payload, timeout=self.read_timeout)
        toks = data.get("tokens")
        if not isinstance(toks, list):
            raise P11ProtocolError(f"/tokenize(chat) returned no tokens: {data!r}")
        return toks

    def chat_wrapper(self, thinking: bool = False):
        """(prefix_ids, suffix_ids) of the chat template around a user message.

        Derived, not hardcoded: a distinctive marker is templated, then located
        inside the result. A template change therefore cannot silently shift the
        prompt geometry -- it fails loudly here instead.
        """
        key = ("wrapper", thinking)
        if key in self._cache:
            return self._cache[key]
        marker = "斑馬騎士的密函"
        mids = self.tokenize(marker, add_special_tokens=False)
        full = self.tokenize_chat([{"role": "user", "content": marker}],
                                  thinking=thinking)
        n = len(mids)
        idx = next((i for i in range(len(full) - n + 1)
                    if full[i:i + n] == mids), -1)
        if idx < 0:
            raise P11ProtocolError(
                "chat template marker not found verbatim in the templated "
                "prompt; the template re-tokenizes the message boundary")
        out = (full[:idx], full[idx + n:])
        self._cache[key] = out
        return out

    def reset_prefix_cache(self):
        """Returns True/False on success/failure, or None if the endpoint is
        absent (server not started with VLLM_SERVER_DEV_MODE=1)."""
        try:
            data, _ = self.post_json("/reset_prefix_cache", {}, budget=1)
        except P11ServerError as exc:
            if exc.status in (404, 405):
                return None
            raise
        return bool(data.get("success"))

    # -- metrics -----------------------------------------------------------

    def metrics(self) -> dict:
        return parse_prometheus(self.get_text("/metrics"))

    def metric(self, snapshot: dict, name: str) -> float:
        """Sum a counter across label sets, tolerating the ``_total`` suffix
        prometheus_client adds on the wire."""
        total = 0.0
        found = False
        for key in (name, name + "_total"):
            if key in snapshot:
                total += snapshot[key]
                found = True
        return total if found else float("nan")

    def settle_metrics(self, before: dict, watch: str, timeout: float) -> dict:
        """Poll /metrics until ``watch`` moves or ``timeout`` elapses.

        Engine-core stats reach the Prometheus registry asynchronously, so a
        scrape taken the instant a request returns can miss it. Bounded so a
        genuinely-zero delta (a real miss) costs at most ``timeout`` seconds.
        """
        deadline = time.monotonic() + timeout
        base = self.metric(before, watch)
        last = before
        while time.monotonic() < deadline:
            time.sleep(0.2)
            try:
                last = self.metrics()
            except P11Error:
                continue
            now = self.metric(last, watch)
            if now != now or base != base:  # NaN: metric absent, stop polling
                break
            if now > base:
                break
        return last

    # -- generation --------------------------------------------------------

    def complete(
        self,
        prompt_ids,
        max_tokens: int,
        temperature: float = 0.0,
        top_k: int = 1,
        seed: int = DEFAULT_SEED,
        logprobs=None,
        stream: bool = True,
        extra: dict | None = None,
    ) -> dict:
        """One /v1/completions call. Returns a measurement record.

        The record ALWAYS carries: ok, error_kind (None on success), attempts,
        token_ids, text, ttft_s, total_s, decode_tok_s, usage, cached_tokens.
        A caller must never have to guess whether a blank cell was a crash, a
        dropped socket, or a real empty completion.
        """
        payload = {
            "model": self.model(),
            "prompt": list(prompt_ids),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_k": top_k,
            # OpenAI seed is int64; seed_of() yields u64 -- mask into range.
            "seed": int(seed) & 0x7FFFFFFFFFFFFFFF,
            "return_token_ids": True,
            "stream": stream,
        }
        if logprobs is not None:
            payload["logprobs"] = logprobs
        if stream:
            # prompt_tokens_details rides ONLY on the final usage chunk, and
            # that chunk only exists when include_usage is set.
            payload["stream_options"] = {"include_usage": True}
        if extra:
            payload.update(extra)

        started = time.monotonic()
        try:
            if stream:
                rec, attempts = self._complete_stream(payload)
            else:
                raw, attempts = self.post_json(
                    "/v1/completions", payload, timeout=self.read_timeout,
                    budget=self.gen_retries,
                )
                rec = _record_from_blocking(raw)
        except P11Error as exc:
            rec = {
                "ok": False,
                "token_ids": [],
                "text": "",
                "token_logprobs": [],
                "ttft_s": None,
                "total_s": time.monotonic() - started,
                "decode_tok_s": None,
                "usage": None,
                "cached_tokens": None,
                "attempts": self.gen_retries,
            }
            rec.update(exc.as_dict())
            return rec

        rec["ok"] = True
        rec["error_kind"] = None
        rec["error"] = None
        rec["attempts"] = attempts
        rec["retried"] = attempts > 1
        rec.setdefault("total_s", time.monotonic() - started)
        rec["prompt_sha256"] = sha256_ids(prompt_ids)
        rec["prompt_len"] = len(prompt_ids)
        return rec

    def _complete_stream(self, payload):
        """SSE read loop. The socket timeout is the IDLE timeout: it fires only
        when the server sends nothing for stall_timeout seconds, which is what
        distinguishes a dead connection from a slow 200k prefill."""

        def once():
            t0 = time.monotonic()
            first_token_at = None
            last_token_at = None
            token_ids: list[int] = []
            token_logprobs: list = []
            chunks_text: list[str] = []
            usage = None
            finish_reason = None
            with self._open(
                "/v1/completions", payload, timeout=self.stall_timeout
            ) as resp:
                while True:
                    try:
                        raw = resp.readline()
                    except (socket.timeout, TimeoutError) as exc:
                        raise P11StallError(
                            f"stream idle > {self.stall_timeout}s after "
                            f"{len(token_ids)} tokens"
                        ) from exc
                    except (ConnectionError, OSError) as exc:
                        raise P11NetworkError(f"stream broke: {exc}") from exc
                    if not raw:
                        break
                    line = raw.decode("utf-8", "replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    body = line[5:].strip()
                    if body == "[DONE]":
                        break
                    try:
                        obj = json.loads(body)
                    except json.JSONDecodeError as exc:
                        raise P11ProtocolError(f"bad SSE chunk: {body[:200]}") from exc
                    if obj.get("usage"):
                        usage = obj["usage"]
                    for choice in obj.get("choices") or []:
                        now = time.monotonic()
                        ids = choice.get("token_ids") or []
                        if ids:
                            if first_token_at is None:
                                first_token_at = now
                            last_token_at = now
                            token_ids.extend(ids)
                        text = choice.get("text") or ""
                        if text:
                            chunks_text.append(text)
                        lp = choice.get("logprobs") or {}
                        token_logprobs.extend(lp.get("token_logprobs") or [])
                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]
            if not token_ids:
                raise P11ProtocolError(
                    "stream returned no token_ids -- is return_token_ids "
                    "supported by this build?"
                )
            decode_tok_s = None
            if last_token_at and first_token_at and last_token_at > first_token_at:
                decode_tok_s = (len(token_ids) - 1) / (last_token_at - first_token_at)
            return {
                "token_ids": token_ids,
                "text": "".join(chunks_text),
                "token_logprobs": token_logprobs,
                "ttft_s": (first_token_at - t0) if first_token_at else None,
                "total_s": time.monotonic() - t0,
                "decode_tok_s": decode_tok_s,
                "finish_reason": finish_reason,
                "usage": usage,
                "cached_tokens": _cached_tokens(usage),
            }

        return self._retrying(once, self.gen_retries, "/v1/completions (stream)")[:2]


def _record_from_blocking(raw: dict) -> dict:
    choice = (raw.get("choices") or [{}])[0]
    usage = raw.get("usage")
    lp = choice.get("logprobs") or {}
    return {
        "token_ids": list(choice.get("token_ids") or []),
        "text": choice.get("text") or "",
        "token_logprobs": list(lp.get("token_logprobs") or []),
        "ttft_s": None,
        "decode_tok_s": None,
        "finish_reason": choice.get("finish_reason"),
        "usage": usage,
        "cached_tokens": _cached_tokens(usage),
    }


def _cached_tokens(usage):
    if not usage:
        return None
    details = usage.get("prompt_tokens_details")
    if not details:
        return None
    return details.get("cached_tokens")


# --------------------------------------------------------------------------
# Prometheus text parsing
# --------------------------------------------------------------------------


def parse_prometheus(text: str) -> dict:
    """name -> summed value across label sets. Labels are dropped on purpose:
    every harness runs against a single engine, and summing is the honest
    aggregation for a counter that only ever has one live label set."""
    out: dict = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            left, value = line.rsplit(" ", 1)
            val = float(value)
        except ValueError:
            continue
        name = left.split("{", 1)[0].strip()
        out[name] = out.get(name, 0.0) + val
    return out


PREFIX_CACHE_METRICS = (
    "vllm:prefix_cache_queries",
    "vllm:prefix_cache_hits",
    "vllm:prompt_tokens_cached",
    "vllm:prompt_tokens",
    "vllm:num_preemptions",
)


def metrics_delta(client: Client, before: dict, after: dict) -> dict:
    delta = {}
    for name in PREFIX_CACHE_METRICS:
        b = client.metric(before, name)
        a = client.metric(after, name)
        delta[name] = None if (a != a or b != b) else a - b
    return delta


# --------------------------------------------------------------------------
# Prompt cache -- the mechanism that makes "B and C saw the same bytes" a
# checkable claim rather than an assumption.
# --------------------------------------------------------------------------


class PromptCache:
    def __init__(self, path: str | None):
        self.path = path
        self.entries: dict = {}
        self.dirty = False
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                self.entries = json.load(fh).get("entries", {})

    def get(self, key: str):
        entry = self.entries.get(key)
        return entry["ids"] if entry else None

    def put(self, key: str, ids) -> None:
        self.entries[key] = {
            "ids": list(ids),
            "n": len(ids),
            "sha256": sha256_ids(ids),
        }
        self.dirty = True

    def digest(self, key: str):
        entry = self.entries.get(key)
        return entry["sha256"] if entry else None

    def save(self) -> None:
        if not self.path or not self.dirty:
            return
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"entries": self.entries}, fh)
        os.replace(tmp, self.path)
        self.dirty = False


def build_ids(
    client: Client,
    n_tokens: int,
    seed: int,
    cache: PromptCache | None = None,
    key: str | None = None,
    add_special_tokens: bool = True,
) -> list:
    """Exactly ``n_tokens`` deterministic token IDs.

    Text is generated from the fixed word list, tokenized server-side (so the
    tokenizer is by construction the model's own), then the ID array is sliced
    to the exact length. Slicing IDs -- rather than trying to hit a token count
    by adjusting text -- is what makes the length exact instead of approximate.
    """
    if cache and key:
        hit = cache.get(key)
        if hit is not None and len(hit) == n_tokens:
            return hit
    words = max(16, int(n_tokens * 1.05))
    ids: list = []
    for attempt in range(6):
        text = filler_text(words, seed)
        ids = client.tokenize(text, add_special_tokens=add_special_tokens)
        if len(ids) >= n_tokens:
            break
        words = int(words * 1.6) + 64
        client.log(f"filler short ({len(ids)}<{n_tokens}), retry {attempt + 1} @ {words} words")
    if len(ids) < n_tokens:
        raise P11ProtocolError(
            f"could not build {n_tokens} tokens (best {len(ids)}); "
            "widen the word list or raise the multiplier"
        )
    ids = ids[:n_tokens]
    if cache and key:
        cache.put(key, ids)
    return ids


def encode_fragment(client: Client, text: str) -> list:
    """Token IDs for a short fragment, WITHOUT special tokens, for splicing
    into an already-built ID array (needles, user turns)."""
    return client.tokenize(text, add_special_tokens=False)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def fmt(value, spec="{:.3f}", dash="-"):
    if value is None:
        return dash
    if isinstance(value, float) and value != value:
        return "nan"
    try:
        return spec.format(value)
    except (ValueError, TypeError):
        return str(value)


def print_table(headers, rows, title=None, stream=sys.stdout) -> None:
    """Fixed-width table. Rows are printed RAW -- no harness aggregates away a
    per-cell value, because an aggregate hides exactly the single corrupted
    seam these tests exist to find."""
    cells = [[str(c) for c in row] for row in rows]
    widths = [len(str(h)) for h in headers]
    for row in cells:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(cell))
    if title:
        print(f"\n== {title} ==", file=stream)
    line = "  ".join(str(h).ljust(w) for h, w in zip(headers, widths))
    print(line, file=stream)
    print("  ".join("-" * w for w in widths), file=stream)
    for row in cells:
        print("  ".join(c.ljust(w) for c, w in zip(row, widths)), file=stream)


def write_json(path: str | None, obj) -> None:
    if not path:
        return
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=False)
    os.replace(tmp, path)
    print(f"\n[p11] wrote {path}", file=sys.stderr)


def run_meta(args, extra=None) -> dict:
    meta = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "monotonic_epoch": time.monotonic(),
        "host": socket.gethostname(),
        "argv": sys.argv,
        "base_url": getattr(args, "base_url", None),
        "config": getattr(args, "config", None),
        "seed": getattr(args, "seed", None),
    }
    if extra:
        meta.update(extra)
    return meta


def server_context(client: Client) -> dict:
    """Everything about the live server that a result file must carry so a
    number can never be read out of its boot context."""
    ctx = {"model": None, "dev_mode_reset_available": None, "metrics": None}
    try:
        ctx["model"] = client.model()
    except P11Error as exc:
        ctx["model_error"] = exc.as_dict()
    try:
        snap = client.metrics()
        ctx["metrics"] = {
            name: client.metric(snap, name) for name in PREFIX_CACHE_METRICS
        }
        ctx["prefix_cache_metrics_present"] = (
            ctx["metrics"]["vllm:prefix_cache_queries"]
            == ctx["metrics"]["vllm:prefix_cache_queries"]
        )
    except P11Error as exc:
        ctx["metrics_error"] = exc.as_dict()
    return ctx


# --------------------------------------------------------------------------
# argparse wiring shared by all four harnesses
# --------------------------------------------------------------------------


def add_common_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("common (P11 rack kit)")
    g.add_argument("--base-url", default=DEFAULT_BASE_URL,
                   help=f"OpenAI-compatible server (default {DEFAULT_BASE_URL})")
    g.add_argument("--config", default="B", choices=["A", "B", "C"],
                   help="which boot config the server is currently running")
    g.add_argument("--log", default=None,
                   help="path to THIS BOOT's serve-<config>-<n>.log "
                        "(recorded in the output; tailed by T4)")
    g.add_argument("--out", default=None, help="machine-readable JSON output path")
    g.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help="prompt seed; must match between B and C runs")
    g.add_argument("--prompt-cache", default=None,
                   help="JSON file of prebuilt token-ID prompts. Point B and C "
                        "at the SAME file to guarantee byte-identical inputs.")
    g.add_argument("--connect-timeout", type=float, default=DEFAULT_CONNECT_TIMEOUT)
    g.add_argument("--read-timeout", type=float, default=DEFAULT_READ_TIMEOUT)
    g.add_argument("--stall-timeout", type=float, default=DEFAULT_STALL_TIMEOUT,
                   help="idle seconds on a stream before declaring a stall")
    g.add_argument("--retries", type=int, default=DEFAULT_RETRIES,
                   help="retry budget for idempotent calls (models/tokenize/metrics)")
    g.add_argument("--gen-retries", type=int, default=DEFAULT_GEN_RETRIES,
                   help="retry budget for generations. >1 can turn a COLD cell "
                        "warm; every retried cell is flagged in the output.")
    g.add_argument("--backoff", type=float, default=DEFAULT_BACKOFF)
    g.add_argument("--metrics-settle", type=float, default=DEFAULT_METRICS_SETTLE,
                   help="bounded wait for /metrics to catch up after a request")
    g.add_argument("--sched-block", type=int, default=DEFAULT_SCHED_BLOCK,
                   help="scheduler_block_size; hit lengths are aligned to it")
    g.add_argument("-v", "--verbose", action="store_true")


def client_from_args(args) -> Client:
    return Client(
        base_url=args.base_url,
        connect_timeout=args.connect_timeout,
        read_timeout=args.read_timeout,
        stall_timeout=args.stall_timeout,
        retries=args.retries,
        gen_retries=args.gen_retries,
        backoff=args.backoff,
        verbose=args.verbose,
    )


def expected_hit(prev_total_tokens: int, sched_block: int) -> int:
    """The design's I2 invariant: hit == floor(prev_ctx / block) * block."""
    return (prev_total_tokens // sched_block) * sched_block
