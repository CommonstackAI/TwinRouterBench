"""Wall-clock prompt cache model used by the harness and by the baseline
re-simulator in ``swerouter.leaderboard.score``.

The cache here is a *harness-side hint* for routers: it tracks, per model, the
last outgoing ``messages`` prefix and when it was sent. Whether the provider
truly cached the prompt is decided by the provider itself and reflected in the
``usage`` payload we receive (see :mod:`swerouter.usage`).

Semantic-prefix matching uses
:func:`main.tokenizer.is_semantic_prefix` from CommonRouterBench so that the
two benchmarks agree on what constitutes "same conversation prefix". This
avoids spurious misses due to formatting drift (e.g. string vs
``[{"type": "text", ...}]`` content blocks).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from main.tokenizer import is_semantic_prefix

from swerouter.router import CacheStateSnapshot, ModelCacheView


@dataclass(frozen=True)
class TTLPolicy:
    """Wall-clock TTL policy.

    The official SWERouterBench leaderboard uses the single policy loaded from
    ``data/ttl_policy.json`` (``WALLCLOCK_5MIN``, 300s). Custom policies may be
    used for research, but are not accepted for official submissions.
    """

    policy_name: str
    wallclock_ttl_sec: int

    @classmethod
    def load(cls, path: str | Path) -> "TTLPolicy":
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"ttl policy file not found: {p}")
        with p.open("r", encoding="utf-8") as fh:
            doc = json.load(fh)
        if not isinstance(doc, dict):
            raise ValueError(f"ttl policy root must be object, got {type(doc).__name__}")
        name = doc.get("policy_name")
        ttl = doc.get("wallclock_ttl_sec")
        if not isinstance(name, str) or not name:
            raise ValueError("ttl policy missing policy_name")
        if not isinstance(ttl, int) or isinstance(ttl, bool) or ttl <= 0:
            raise ValueError(f"ttl policy wallclock_ttl_sec must be positive int, got {ttl!r}")
        return cls(policy_name=name, wallclock_ttl_sec=ttl)


@dataclass
class _CacheEntry:
    """Mutable per-model cache state. Internal to :class:`PromptCacheModel`."""

    messages: tuple[Mapping[str, Any], ...]
    prefix_token_count: int
    last_call_ts: float
    prefix_hash: str


def messages_fingerprint(messages: tuple[Mapping[str, Any], ...]) -> str:
    """Stable fingerprint of a message list for cache identity (ignores cache_control)."""
    # Use CRB's _semantic_fingerprint-equivalent via json of only semantic keys.
    semantic_keys = ("role", "content", "tool_calls", "tool_call_id", "name")
    normalized = [
        {k: m[k] for k in semantic_keys if k in m}
        for m in messages
    ]
    blob = json.dumps(normalized, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CacheLookupResult:
    """Result of :meth:`PromptCacheModel.lookup`, consumed by the agent loop
    for tracing and by the baseline re-simulator for cost estimation.
    """

    hit: bool
    reason: str
    cached_prefix_token_count: int
    last_call_ts: float | None


class PromptCacheModel:
    """Per-instance, mutable cache model shared across steps.

    Thread safety: instances are NOT thread-safe. Each SWE-bench instance runs
    in its own worker and owns its own :class:`PromptCacheModel`.
    """

    def __init__(self, ttl: TTLPolicy) -> None:
        self._ttl = ttl
        self._by_model: dict[str, _CacheEntry] = {}

    @property
    def ttl(self) -> TTLPolicy:
        return self._ttl

    def lookup(
        self,
        *,
        model_id: str,
        messages: tuple[Mapping[str, Any], ...],
        now_ts: float,
    ) -> CacheLookupResult:
        """Decide whether the upcoming request is a cache hit under our TTL model."""

        entry = self._by_model.get(model_id)
        if entry is None:
            return CacheLookupResult(
                hit=False,
                reason="cold_start_first_call",
                cached_prefix_token_count=0,
                last_call_ts=None,
            )
        if now_ts - entry.last_call_ts > self._ttl.wallclock_ttl_sec:
            return CacheLookupResult(
                hit=False,
                reason="ttl_expired",
                cached_prefix_token_count=0,
                last_call_ts=entry.last_call_ts,
            )
        if not is_semantic_prefix(list(entry.messages), list(messages)):
            return CacheLookupResult(
                hit=False,
                reason="prefix_mismatch",
                cached_prefix_token_count=0,
                last_call_ts=entry.last_call_ts,
            )
        return CacheLookupResult(
            hit=True,
            reason="prefix_match",
            cached_prefix_token_count=entry.prefix_token_count,
            last_call_ts=entry.last_call_ts,
        )

    def update(
        self,
        *,
        model_id: str,
        messages: tuple[Mapping[str, Any], ...],
        prefix_token_count: int,
        ts: float,
    ) -> None:
        """Record the state after a successful LLM call so the next lookup can hit."""

        if prefix_token_count < 0:
            raise ValueError(f"prefix_token_count must be non-negative, got {prefix_token_count}")
        self._by_model[model_id] = _CacheEntry(
            messages=tuple(messages),
            prefix_token_count=int(prefix_token_count),
            last_call_ts=float(ts),
            prefix_hash=messages_fingerprint(tuple(messages)),
        )

    def snapshot(
        self,
        *,
        now_ts: float,
        available_models: tuple[str, ...],
    ) -> CacheStateSnapshot:
        """Produce an immutable snapshot for :class:`RouterContext.cache_state`."""

        view: dict[str, ModelCacheView] = {}
        for model_id in available_models:
            entry = self._by_model.get(model_id)
            if entry is None:
                view[model_id] = ModelCacheView(
                    model_id=model_id,
                    last_call_ts=None,
                    prefix_token_count=0,
                    prefix_hash=None,
                )
            else:
                view[model_id] = ModelCacheView(
                    model_id=model_id,
                    last_call_ts=entry.last_call_ts,
                    prefix_token_count=entry.prefix_token_count,
                    prefix_hash=entry.prefix_hash,
                )
        return CacheStateSnapshot(
            wallclock_ttl_sec=self._ttl.wallclock_ttl_sec,
            now_ts=now_ts,
            by_model=view,
        )


@dataclass(frozen=True)
class BaselineStepCacheDecision:
    """Per-step decision produced by the baseline re-simulator.

    ``cold_start`` is True when baseline treats the step as a cache-write from
    scratch (first step, TTL expired). ``cache_read_tokens`` and
    ``cache_write_tokens`` partition the step's total prompt tokens at HIGH
    model rates (see ``docs/scoring_zh.md`` §4).
    """

    step_index: int
    cold_start: bool
    cache_read_tokens: int
    cache_write_tokens: int


def simulate_baseline_cache_sequence(
    *,
    prefix_tokens_by_step: list[int],
    wallclock_ts_by_step: list[float],
    ttl: TTLPolicy,
) -> list[BaselineStepCacheDecision]:
    """Independent re-simulation of the cache chain for the "always-HIGH" baseline.

    Because the baseline uses one model for every step, the cache chain is
    perfect unless TTL expires. We do NOT re-tokenize messages under HIGH's
    tokenizer here; we reuse the per-step prompt token counts from the
    router's actual run (see ``docs/scoring_zh.md`` §4.1 for the first-order
    approximation note).
    """

    n = len(prefix_tokens_by_step)
    if n != len(wallclock_ts_by_step):
        raise ValueError(
            "prefix_tokens_by_step and wallclock_ts_by_step must have equal length"
        )
    decisions: list[BaselineStepCacheDecision] = []
    for i in range(n):
        if i == 0:
            cold = True
        else:
            gap = wallclock_ts_by_step[i] - wallclock_ts_by_step[i - 1]
            if gap < 0:
                raise ValueError(
                    f"wallclock_ts non-monotonic at step {i}: gap={gap}"
                )
            cold = gap > ttl.wallclock_ttl_sec
        prefix = prefix_tokens_by_step[i]
        if prefix < 0:
            raise ValueError(f"prefix_tokens at step {i} is negative: {prefix}")
        if cold:
            decisions.append(
                BaselineStepCacheDecision(
                    step_index=i,
                    cold_start=True,
                    cache_read_tokens=0,
                    cache_write_tokens=prefix,
                )
            )
        else:
            prev = prefix_tokens_by_step[i - 1]
            cache_write = max(0, prefix - prev)
            cache_read = prefix - cache_write
            decisions.append(
                BaselineStepCacheDecision(
                    step_index=i,
                    cold_start=False,
                    cache_read_tokens=cache_read,
                    cache_write_tokens=cache_write,
                )
            )
    return decisions
