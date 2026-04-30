"""OpenAI-compatible HTTP client: classify routing tier as a single digit 0–3 (public tier ids)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Literal

import requests

from main.tiers import TIER_HIGH, TIER_LOW, TIER_MID, TIER_MID_HIGH

_RETRYABLE_HTTP_STATUSES: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504})

DEFAULT_ROUTER_SYSTEM_INSTRUCTION = """You are a machine-facing routing labeler, not a chat assistant.

The user message contains ONE case (a trajectory excerpt). Your job: pick which capability tier is appropriate for the NEXT model call in that trajectory.

Tier ids (pick exactly one digit):
- 0 = low
- 1 = mid
- 2 = mid_high
- 3 = high

You MUST NOT explain, think out loud, plan, apologize, or add any prose. No "THOUGHT:", no XML/HTML tags, no markdown fences, no bullet lists, no quotes around the digit.

Your assistant message is consumed by a strict parser: it must be ONE line only, and that line must be a single ASCII digit 0–3. If you output a second line, or any non-digit character on the line, the run fails.

Do not describe the case. Do not justify. Output the digit only."""


# Appended after the case text on every request (not cached with system); reminds models right before they answer.
ROUTER_USER_OUTPUT_SUFFIX = """
---
[OUTPUT_FORMAT — read before you reply]
Your next assistant message must satisfy ALL of:
1) Exactly one line (no line breaks inside the message body).
2) That line is exactly one character.
3) That character is one of: 0, 1, 2, 3 (tier id for the next model call).

Invalid (parser will reject): multiple lines; leading/trailing words; "THOUGHT:"; "```"; "Answer: 2"; spaces like " 2 "; anything except a lone digit on one line.

Valid examples (the entire assistant message): 0
Valid examples: 3
"""


def chat_completions_url(base_url: str) -> str:
    root = base_url.strip().rstrip("/")
    if root.endswith("/chat/completions"):
        return root
    return f"{root}/chat/completions"


def build_system_content(system_text: str, *, use_cache_block: bool) -> str | list[dict[str, Any]]:
    """Plain string, or Anthropic-style one-block list with ephemeral cache_control."""
    if use_cache_block:
        return [
            {
                "type": "text",
                "text": system_text,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    return system_text


def parse_tier_response_to_id(content: str) -> int:
    """
    After strip(), the assistant message must be exactly one character in 0–3.

    Leading/trailing whitespace and newlines are stripped first (some APIs append a trailing newline).
    After strip, the body must be a single line: exactly one character in 0–3.
    Does not scan inside longer strings or strip XML-style thinking blocks.
    """
    if not isinstance(content, str):
        raise ValueError("model content must be a string")
    s = content.strip()
    if "\n" in s or "\r" in s:
        raise ValueError("model content must not contain embedded line breaks (only one line after strip)")
    if len(s) != 1:
        raise ValueError(f"expected exactly one character after strip, got length {len(s)!r}: {s!r}")
    if s not in "0123":
        raise ValueError(f"expected digit 0-3, got {s!r}")
    return int(s)


def _parse_chat_completions_response(resp: requests.Response) -> tuple[str, dict[str, Any] | None]:
    try:
        data = resp.json()
    except json.JSONDecodeError as e:
        body = (resp.text or "")[:400]
        raise ValueError(f"response is not JSON (body prefix): {body!r}") from e
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError(f"unexpected response shape (choices): {list(data.keys())}")
    msg = choices[0].get("message")
    if not isinstance(msg, dict):
        raise ValueError("unexpected response shape (choices[0].message)")
    c = msg.get("content") or ""
    if not isinstance(c, str):
        raise ValueError("choices[0].message.content is not a string")
    usage = data.get("usage")
    usage_out: dict[str, Any] | None = usage if isinstance(usage, dict) else None
    return c, usage_out


def post_chat_completions(
    *,
    url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    timeout_s: int,
    max_attempts: int,
    retry_backoff_s: float,
) -> tuple[str, dict[str, Any] | None]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    last_error: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
        except (requests.Timeout, requests.ConnectionError) as ex:
            last_error = ex
            if attempt >= max_attempts:
                raise RuntimeError(
                    f"{type(ex).__name__} after {max_attempts} attempt(s): {ex}"
                ) from ex
            delay = retry_backoff_s * (2 ** (attempt - 1))
            time.sleep(delay)
            continue

        if resp.status_code == 200:
            return _parse_chat_completions_response(resp)

        body = (resp.text or "")[:800]
        err = RuntimeError(f"HTTP {resp.status_code} from {url}: {body}")
        last_error = err
        if resp.status_code not in _RETRYABLE_HTTP_STATUSES or attempt >= max_attempts:
            raise err
        delay = retry_backoff_s * (2 ** (attempt - 1))
        time.sleep(delay)

    raise RuntimeError("internal retry loop exited unexpectedly") from last_error


@dataclass(frozen=True)
class TierPredictionResult:
    tier_id: int
    raw_content: str
    usage: dict[str, Any] | None


SystemPromptCacheMode = Literal["auto", "on", "off"]


class OpenAICompatRouterClassifier:
    """
    Call ``POST .../chat/completions`` with system (optional cached block for Claude) + user string.

    Tier ids match ``target_tier_id`` in the public question bank: 0 low, 1 mid, 2 mid_high, 3 high.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_s: int = 180,
        max_attempts: int = 4,
        retry_backoff_s: float = 1.0,
        system_prompt_cache: SystemPromptCacheMode = "auto",
        system_instruction: str | None = None,
        max_completion_tokens: int = 4,
    ) -> None:
        self._base_url = base_url
        self._api_key = api_key
        self._model = model
        self._timeout_s = timeout_s
        self._max_attempts = max_attempts
        self._retry_backoff_s = retry_backoff_s
        self._system_prompt_cache = system_prompt_cache
        self._system_instruction = system_instruction
        self._max_completion_tokens = max_completion_tokens

    @property
    def model(self) -> str:
        return self._model

    def _effective_system_text(self) -> str:
        return self._system_instruction if self._system_instruction is not None else DEFAULT_ROUTER_SYSTEM_INSTRUCTION

    def _use_system_cache_block(self) -> bool:
        if self._system_prompt_cache == "off":
            return False
        if self._system_prompt_cache == "on":
            return True
        return "claude" in self._model.lower()

    def predict_tier_id(self, prompt: str) -> TierPredictionResult:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be str")
        system_text = self._effective_system_text()
        system_content = build_system_content(system_text, use_cache_block=self._use_system_cache_block())
        user_content = f"{prompt.rstrip()}{ROUTER_USER_OUTPUT_SUFFIX}"
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        url = chat_completions_url(self._base_url)
        raw_text, usage = post_chat_completions(
            url=url,
            api_key=self._api_key,
            model=self._model,
            messages=messages,
            max_tokens=self._max_completion_tokens,
            timeout_s=self._timeout_s,
            max_attempts=self._max_attempts,
            retry_backoff_s=self._retry_backoff_s,
        )
        tier_id = parse_tier_response_to_id(raw_text)
        stripped = raw_text.strip()
        return TierPredictionResult(tier_id=tier_id, raw_content=stripped, usage=usage)


def tier_id_to_public_label(tier_id: int) -> str:
    """Map 0..3 to English tier label (for debugging or logging)."""
    m = {0: TIER_LOW, 1: TIER_MID, 2: TIER_MID_HIGH, 3: TIER_HIGH}
    if tier_id not in m:
        raise ValueError(f"tier_id must be 0..3, got {tier_id!r}")
    return m[tier_id]
