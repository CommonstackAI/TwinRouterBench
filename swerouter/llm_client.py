"""OpenAI-compatible chat completion client used by the agent loop.

Unlike CommonRouterBench's :class:`main.router_llm.OpenAICompatRouterClassifier`
(which is a one-shot single-digit classifier), this client supports:

* Multi-turn ``messages`` histories with ``role=tool`` responses.
* OpenAI-style ``tools`` array for function/tool calling.
* Provider-aware ``cache_control`` block injection so Anthropic-family models
  exercise the ephemeral prompt cache in the same 5-minute TTL window that the
  harness tracks on its side.

This module does NOT own retry policy for semantic errors (bad tool JSON from
the model, etc.): those surface as Python exceptions and abort the instance
(fail fast). Only transport-level errors (timeouts, 5xx) are retried.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import requests

from swerouter.usage import UsageBuckets, normalize_usage

_RETRYABLE_HTTP_STATUSES: frozenset[int] = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class TransientResponseError(ValueError):
    """Base class for HTTP-response-level failures that are worth retrying.

    Any parse-time failure on an HTTP 200 body that plausibly reflects
    upstream flakiness (truncated / non-JSON body, missing required fields
    in the OpenAI-compat envelope, missing usage block) is raised as a
    subclass here so the retry loop in :meth:`LLMClient.chat` treats it the
    same way as a retryable HTTP status. Extending :class:`ValueError`
    preserves backwards compatibility with callers that used to see
    ``ValueError`` after the hard failure path.
    """


class ProviderMissingUsageError(TransientResponseError):
    """Raised when a provider returns HTTP 200 without a ``usage`` object.

    OpenRouter occasionally forwards upstream responses from some models
    (observed with ``minimax/minimax-m2.7``) that omit ``usage`` in a subset
    of 200 responses. Since the harness cannot price the call without that
    object, we retry like any other transient response-level failure.
    """


class MalformedResponseError(TransientResponseError):
    """Raised when an HTTP 200 body is not valid JSON or is missing the
    required OpenAI-compat shape (``choices[0].message``). Typically the
    symptom of a vendor returning an HTML error page or truncated body.
    """


def _chat_completions_url(base_url: str) -> str:
    root = base_url.strip().rstrip("/")
    if root.endswith("/chat/completions"):
        return root
    return f"{root}/chat/completions"


def _inject_cache_control_for_anthropic(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert the last user / system content into an Anthropic-style ``list``
    with an ``ephemeral`` ``cache_control`` block so the prompt cache is used.

    Rules
    -----
    * System message (if present at index 0) is wrapped.
    * The last preceding assistant / tool / user turn is wrapped so Anthropic
      will reuse the cache up to that point. (OpenRouter will pass the
      ``cache_control`` field through to Anthropic; other providers ignore it.)
    """

    result: list[dict[str, Any]] = []
    last_cacheable_idx = -1
    for i, m in enumerate(messages):
        if m.get("role") in ("system", "user", "tool"):
            last_cacheable_idx = i

    for i, m in enumerate(messages):
        new = dict(m)
        if i == 0 and m.get("role") == "system" and isinstance(m.get("content"), str):
            new["content"] = [
                {"type": "text", "text": m["content"], "cache_control": {"type": "ephemeral"}}
            ]
        elif i == last_cacheable_idx and i != 0 and isinstance(m.get("content"), str):
            new["content"] = [
                {"type": "text", "text": m["content"], "cache_control": {"type": "ephemeral"}}
            ]
        result.append(new)
    return result


@dataclass(frozen=True)
class ChatCallResult:
    """Return value of :meth:`LLMClient.chat`.

    ``assistant_message`` is the full OpenAI-style message dict (has ``role``,
    ``content`` and optionally ``tool_calls``). ``raw_usage`` is the provider's
    raw ``usage`` payload exactly as returned; ``normalized_usage`` is the
    4-bucket version used for pricing. ``latency_ms`` measures only the POST.
    """

    assistant_message: Mapping[str, Any]
    raw_usage: Mapping[str, Any]
    normalized_usage: UsageBuckets
    latency_ms: float
    model_id: str
    provider: str


class LLMClient:
    """Thin OpenAI-compatible client with provider-aware cache hints.

    One instance is safe to share across threads because each ``chat`` call
    makes its own ``requests.post`` (the ``requests`` library is thread-safe
    for concurrent requests).
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout_s: int = 180,
        max_attempts: int = 4,
        retry_backoff_s: float = 1.0,
    ) -> None:
        if not base_url:
            raise ValueError("base_url is required")
        if not api_key:
            raise ValueError("api_key is required")
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self._url = _chat_completions_url(base_url)
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._max_attempts = max_attempts
        self._retry_backoff_s = retry_backoff_s

    def chat(
        self,
        *,
        model_id: str,
        provider: str,
        messages: Iterable[Mapping[str, Any]],
        tools: Iterable[Mapping[str, Any]] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        cache_control_style: str | None = None,
    ) -> ChatCallResult:
        """POST to the OpenAI-compatible ``/chat/completions`` endpoint.

        ``provider`` drives usage-payload normalization (see
        :mod:`swerouter.usage`). ``cache_control_style`` is an independent
        switch that injects Anthropic-style ephemeral ``cache_control`` blocks;
        when accessing Anthropic models via OpenRouter the returned usage is
        still OpenAI-compat, so ``provider="openai_compat"`` is the right
        choice for parsing while ``cache_control_style="anthropic"`` is the
        right choice for the prompt shape.
        """

        msgs_list: list[dict[str, Any]] = [dict(m) for m in messages]
        if not msgs_list:
            raise ValueError("messages must be non-empty")

        if cache_control_style == "anthropic":
            msgs_list = _inject_cache_control_for_anthropic(msgs_list)
        elif cache_control_style not in (None, ""):
            raise ValueError(
                f"Unsupported cache_control_style: {cache_control_style!r}"
            )

        payload: dict[str, Any] = {
            "model": model_id,
            "messages": msgs_list,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        tool_list = list(tools) if tools is not None else None
        if tool_list:
            payload["tools"] = tool_list

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        last_error: BaseException | None = None
        for attempt in range(1, self._max_attempts + 1):
            t0 = time.monotonic()
            try:
                resp = requests.post(
                    self._url, headers=headers, json=payload, timeout=self._timeout_s
                )
            except (requests.Timeout, requests.ConnectionError) as ex:
                last_error = ex
                if attempt >= self._max_attempts:
                    raise RuntimeError(
                        f"{type(ex).__name__} after {self._max_attempts} attempt(s) to {self._url}: {ex}"
                    ) from ex
                time.sleep(self._retry_backoff_s * (2 ** (attempt - 1)))
                continue

            latency_ms = (time.monotonic() - t0) * 1000.0

            if resp.status_code == 200:
                try:
                    return self._parse_response(
                        resp=resp,
                        model_id=model_id,
                        provider=provider,
                        latency_ms=latency_ms,
                    )
                except TransientResponseError as ex:
                    last_error = ex
                    if attempt >= self._max_attempts:
                        raise
                    time.sleep(self._retry_backoff_s * (2 ** (attempt - 1)))
                    continue

            body = (resp.text or "")[:1200]
            err = RuntimeError(f"HTTP {resp.status_code} from {self._url}: {body}")
            last_error = err
            if (
                resp.status_code not in _RETRYABLE_HTTP_STATUSES
                or attempt >= self._max_attempts
            ):
                raise err
            time.sleep(self._retry_backoff_s * (2 ** (attempt - 1)))

        raise RuntimeError("internal retry loop exited unexpectedly") from last_error

    @staticmethod
    def _parse_response(
        *,
        resp: requests.Response,
        model_id: str,
        provider: str,
        latency_ms: float,
    ) -> ChatCallResult:
        try:
            data = resp.json()
        except json.JSONDecodeError as e:
            body = (resp.text or "")[:400]
            raise MalformedResponseError(
                f"response is not JSON (body prefix): {body!r}"
            ) from e

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise MalformedResponseError(
                f"unexpected response shape (choices missing): keys={list(data.keys())}"
            )
        first = choices[0]
        if not isinstance(first, Mapping):
            raise MalformedResponseError("choices[0] is not an object")
        msg = first.get("message")
        if not isinstance(msg, Mapping):
            raise MalformedResponseError("choices[0].message is missing or not an object")

        raw_usage = data.get("usage")
        if not isinstance(raw_usage, Mapping):
            raise ProviderMissingUsageError(
                f"response for model_id={model_id!r} missing usage object (required for pricing)"
            )
        normalized = normalize_usage(provider, raw_usage)

        return ChatCallResult(
            assistant_message=dict(msg),
            raw_usage=dict(raw_usage),
            normalized_usage=normalized,
            latency_ms=latency_ms,
            model_id=model_id,
            provider=provider,
        )
