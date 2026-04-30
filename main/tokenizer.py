"""Per-tier token counting and prompt-token splitting for cost calculation.

Each public tier uses its vendor's native tokenizer where available:

* high        — Anthropic native tokenizer (bundled JSON, from claude-opus-4-6 family)
* mid_high    — ``cl100k_base`` (Gemini has no offline tokenizer; this is the fallback)
* mid         — MiniMax native tokenizer (HuggingFace: ``MiniMaxAI/MiniMax-Text-01``)
* low         — DeepSeek native tokenizer (HuggingFace: ``deepseek-ai/DeepSeek-V3``)

The ``tokenizers`` (HuggingFace) package is required for native tokenizer support.
If it is not installed, all tiers fall back to ``tiktoken cl100k_base``.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import tiktoken

from main.tiers import TIER_HIGH, TIER_LOW, TIER_MID, TIER_MID_HIGH

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tokenizer abstraction — unifies tiktoken and HuggingFace tokenizers
# ---------------------------------------------------------------------------

class _TokenEncoder(Protocol):
    def count(self, text: str) -> int: ...


class _TiktokenEncoder:
    def __init__(self, encoding_name: str) -> None:
        self._enc = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        return len(self._enc.encode(text))


class _HuggingFaceEncoder:
    def __init__(self, tokenizer: Any) -> None:
        self._tok = tokenizer

    def count(self, text: str) -> int:
        return len(self._tok.encode(text).ids)


# ---------------------------------------------------------------------------
# Tier → tokenizer singleton loading
# ---------------------------------------------------------------------------

_FALLBACK_ENCODING = "cl100k_base"

_ANTHROPIC_TOKENIZER_PATH = Path(__file__).parent / "tokenizer_data" / "anthropic_tokenizer.json"

_HF_TOKENIZER_IDENTIFIERS: dict[str, str] = {
    TIER_MID: "MiniMaxAI/MiniMax-Text-01",
    TIER_LOW: "deepseek-ai/DeepSeek-V3",
}


@lru_cache(maxsize=8)
def _load_tier_encoder(tier: str) -> _TokenEncoder:
    """Load the best available tokenizer for *tier*."""
    try:
        from tokenizers import Tokenizer as HFTokenizer
    except ImportError:
        logger.warning(
            "tokenizers package not installed; falling back to cl100k_base for all tiers"
        )
        return _TiktokenEncoder(_FALLBACK_ENCODING)

    if tier == TIER_HIGH:
        if _ANTHROPIC_TOKENIZER_PATH.exists():
            tok = HFTokenizer.from_str(_ANTHROPIC_TOKENIZER_PATH.read_text(encoding="utf-8"))
            return _HuggingFaceEncoder(tok)
        logger.warning("Anthropic tokenizer JSON not found; falling back to cl100k_base for high tier")
        return _TiktokenEncoder(_FALLBACK_ENCODING)

    hf_id = _HF_TOKENIZER_IDENTIFIERS.get(tier)
    if hf_id is not None:
        try:
            tok = HFTokenizer.from_pretrained(hf_id)
            return _HuggingFaceEncoder(tok)
        except Exception:
            logger.warning(
                "Failed to load HuggingFace tokenizer %s for tier %s; falling back to cl100k_base",
                hf_id, tier,
            )
            return _TiktokenEncoder(_FALLBACK_ENCODING)

    return _TiktokenEncoder(_FALLBACK_ENCODING)


# ---------------------------------------------------------------------------
# Message-level token counting
# ---------------------------------------------------------------------------

def _message_text(msg: dict[str, Any]) -> str:
    """Extract all billable text from a single chat message.

    Handles both ``content: str`` and ``content: list[{type, text, ...}]``
    formats.  Also serialises ``tool_calls`` (function name + arguments) which
    count toward output tokens when the role is ``assistant``.
    """
    parts: list[str] = []

    content = msg.get("content")
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if text:
                    parts.append(str(text))
            elif isinstance(block, str):
                parts.append(block)

    tool_calls = msg.get("tool_calls")
    if tool_calls:
        for tc in tool_calls:
            fn = tc.get("function", tc) if isinstance(tc, dict) else tc
            if isinstance(fn, dict):
                parts.append(fn.get("name", ""))
                args = fn.get("arguments", "")
                if isinstance(args, dict):
                    parts.append(json.dumps(args, ensure_ascii=False))
                elif args:
                    parts.append(str(args))

    return "\n".join(parts)


def count_messages_tokens(messages: list[dict[str, Any]], tier: str) -> int:
    """Count total tokens in *messages* using the tokenizer bound to *tier*."""
    encoder = _load_tier_encoder(tier)
    total = 0
    for msg in messages:
        total += encoder.count(_message_text(msg))
        total += 4  # ~4 tokens overhead per message for role / separators
    total += 2  # priming tokens
    return total


def count_text_tokens(text: str, tier: str) -> int:
    """Count tokens for a raw text string."""
    encoder = _load_tier_encoder(tier)
    return encoder.count(text)


# ---------------------------------------------------------------------------
# Semantic prefix check
# ---------------------------------------------------------------------------

_SEMANTIC_KEYS = ("role", "content", "tool_calls", "tool_call_id", "name")

_CONTENT_BLOCK_IGNORE_KEYS = frozenset({"cache_control"})


def _normalise_content(content: Any) -> str:
    """Normalise content to a plain text string for semantic comparison.

    Handles both ``str`` content and ``list[dict]`` (structured content blocks)
    as emitted by OpenClaw / Anthropic-style APIs.  OpenClaw may serialise the
    same message as a plain string in one turn and as
    ``[{"type": "text", "text": "..."}]`` in another; normalising to text makes
    the comparison format-agnostic.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if text:
                    parts.append(str(text))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content) if content is not None else ""


def _semantic_fingerprint(msg: dict[str, Any]) -> str:
    """Deterministic string capturing only the semantically meaningful fields."""
    parts: dict[str, Any] = {}
    for k in _SEMANTIC_KEYS:
        if k in msg:
            val = msg[k]
            if k == "content":
                val = _normalise_content(val)
            parts[k] = val
    return json.dumps(parts, sort_keys=True, ensure_ascii=False)


def is_semantic_prefix(msgs_short: list[dict], msgs_long: list[dict]) -> bool:
    """Return True if *msgs_short* is a semantic prefix of *msgs_long*.

    Compares only ``role``, ``content``, ``tool_calls``, ``tool_call_id``, and
    ``name`` — ignoring metadata like ``cache_control`` that OpenClaw may
    serialise inconsistently across turns.
    """
    if len(msgs_short) > len(msgs_long):
        return False
    for a, b in zip(msgs_short, msgs_long):
        if _semantic_fingerprint(a) != _semantic_fingerprint(b):
            return False
    return True


# ---------------------------------------------------------------------------
# Per-step prompt token splitting: input / cache_read / cache_write
# ---------------------------------------------------------------------------

def split_prompt_tokens_for_step(
    *,
    prev_tier: str | None,
    curr_tier: str,
    msgs_prev: list[dict[str, Any]] | None,
    msgs_curr: list[dict[str, Any]],
    cache_expired: bool = False,
) -> tuple[int, int, int]:
    """Return ``(input_tokens, cache_read_tokens, cache_write_tokens)``.

    Rules
    -----
    * First step (``prev_tier is None``) or single-turn → all cache_write
      (cold start: the full prompt seeds the cache).
    * Tier switch (``curr_tier != prev_tier``) → cold start, all cache_write.
    * Cache expired (TTL exceeded) → cold start, all cache_write.
    * Same tier, semantic prefix match, cache valid → cache_read for prefix,
      cache_write for delta.
    * Same tier, prefix mismatch → fallback to all cache_write.
    """
    total = count_messages_tokens(msgs_curr, curr_tier)

    if prev_tier is None or msgs_prev is None:
        return (0, 0, total)

    if curr_tier != prev_tier:
        return (0, 0, total)

    if cache_expired:
        return (0, 0, total)

    if not is_semantic_prefix(msgs_prev, msgs_curr):
        return (0, 0, total)

    prefix_tokens = count_messages_tokens(msgs_prev, curr_tier)
    delta_tokens = max(total - prefix_tokens, 0)
    return (0, prefix_tokens, delta_tokens)


# ---------------------------------------------------------------------------
# Output-token estimation from message deltas
# ---------------------------------------------------------------------------

def estimate_output_tokens_from_delta(
    msgs_curr: list[dict[str, Any]],
    msgs_next: list[dict[str, Any]],
    tier: str,
) -> int:
    """Estimate output tokens for the current step from the next step's delta.

    Only ``role=assistant`` messages in the delta count (including their
    ``tool_calls`` JSON).  ``role=tool`` / ``role=user`` messages in the delta
    are environment or user input, not model output.
    """
    encoder = _load_tier_encoder(tier)
    n_curr = len(msgs_curr)
    delta = msgs_next[n_curr:]

    tokens = 0
    for msg in delta:
        if msg.get("role") == "assistant":
            tokens += encoder.count(_message_text(msg))
            tokens += 4  # per-message overhead
    return tokens
