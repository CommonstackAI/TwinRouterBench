"""Per-instance routing session that any benchmark harness can call.

The SWE-bench dynamic track wires the router into mini-swe-agent through
``MiniRouterAgent.query`` + ``RouterAwareModel.query_as``. This module factors
the benchmark-independent half of that wiring so a new benchmark only has to
replace its own LLM call with :meth:`RoutedSession.call` and report the
official pass/fail verdict through :meth:`RoutedSession.finish`.

Per call the session:

1. builds a :class:`swerouter.router.RouterContext` from the live prefix,
   tool schemas, harness-side cache state, and spend so far;
2. asks the router for one ``model_id`` and validates it against the locked
   pool (fail fast, no default model);
3. dispatches the unchanged request through the harness-supplied
   ``call_model`` callable;
4. normalizes the provider ``usage`` payload into four buckets and prices it
   with the locked ``model_pricing.json`` (missing usage is an error);
5. appends one trace row in the layout ``swerouter.leaderboard.score`` reads.

:meth:`finish` writes the ``loop_summary`` marker and
``results/<instance_id>.json`` so ``twinrouterbench dynamic score`` grades the
run directory without benchmark-specific code.
"""

from __future__ import annotations

import copy
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from swerouter.agent.loop import ModelPoolEntry, load_model_pool
from swerouter.cache import PromptCacheModel, TTLPolicy
from swerouter.pricing import PricingTable, load_pricing_table, step_real_cost_usd
from swerouter.router import Router, RouterContext, RunConfig, validate_decision
from swerouter.usage import normalize_usage

REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCKED_DATA = REPO_ROOT / "data" / "dynamic"
DEFAULT_POOL = _LOCKED_DATA / "model_pool.json"
DEFAULT_PRICING = _LOCKED_DATA / "model_pricing.json"
DEFAULT_TTL = _LOCKED_DATA / "ttl_policy.json"


@dataclass(frozen=True)
class ModelCallResult:
    """What the harness-side ``call_model`` returns for one LLM request.

    ``message`` is the assistant message exactly as the benchmark harness will
    append it to its history. ``raw_usage`` is the provider's ``usage`` object
    from the same response; it is required because cost is computed from it.
    """

    message: Mapping[str, Any]
    raw_usage: Mapping[str, Any]


ModelCaller = Callable[
    [str, list[dict[str, Any]], tuple[Mapping[str, Any], ...]], ModelCallResult
]
"""``call_model(model_id, messages, tools) -> ModelCallResult``."""


@dataclass(frozen=True)
class SessionLimits:
    """Limits the benchmark harness already enforces, exposed to the router.

    The session does not enforce them; the harness keeps its own official
    step/cost limits so routed and unrouted runs stop under identical rules.
    """

    max_steps: int
    budget_usd: float
    select_timeout_sec: float = 30.0


class RoutedSession:
    """Route every LLM call of one benchmark instance and record the trace."""

    def __init__(
        self,
        *,
        router: Router,
        instance_id: str,
        run_dir: str | Path,
        call_model: ModelCaller,
        limits: SessionLimits,
        pool_path: str | Path = DEFAULT_POOL,
        pricing_path: str | Path = DEFAULT_PRICING,
        ttl_path: str | Path = DEFAULT_TTL,
    ) -> None:
        if not instance_id:
            raise ValueError("instance_id must be non-empty")
        self._router = router
        self._instance_id = instance_id
        self._run_dir = Path(run_dir)
        self._call_model = call_model
        self._limits = limits

        self._pool: list[ModelPoolEntry] = load_model_pool(pool_path)
        self._pricing: PricingTable = load_pricing_table(pricing_path)
        self._ttl = TTLPolicy.load(ttl_path)
        for entry in self._pool:
            if entry.model_id not in self._pricing:
                raise KeyError(
                    f"pool model {entry.model_id!r} missing from pricing table "
                    f"schema v{self._pricing.schema_version}"
                )
        self._provider_by_model = {e.model_id: e.provider for e in self._pool}
        self._available_models = tuple(e.model_id for e in self._pool)
        self._cache = PromptCacheModel(self._ttl)

        self._trace_path = self._run_dir / f"{instance_id}.trace.jsonl"
        self._trace_path.parent.mkdir(parents=True, exist_ok=True)
        if self._trace_path.exists():
            self._trace_path.unlink()

        self._step_costs: list[float] = []
        self._model_counts: Counter[str] = Counter()
        self._finished = False

    @property
    def available_models(self) -> tuple[str, ...]:
        return self._available_models

    @property
    def total_router_cost_usd(self) -> float:
        return sum(self._step_costs)

    @property
    def trace_path(self) -> Path:
        return self._trace_path

    def call(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> Mapping[str, Any]:
        """Route and execute one LLM call; return the assistant message."""

        if self._finished:
            raise RuntimeError("RoutedSession.call() after finish()")
        if not messages:
            raise ValueError("messages must be non-empty")
        step_index = len(self._step_costs)
        prefix = tuple(copy.deepcopy(list(messages)))
        tool_schemas = tuple(copy.deepcopy(list(tools)))

        now_ts = time.time()
        ctx = RouterContext(
            instance_id=self._instance_id,
            step_index=step_index,
            messages=prefix,
            tools=tool_schemas,
            available_models=self._available_models,
            cache_state=self._cache.snapshot(
                now_ts=now_ts, available_models=self._available_models
            ),
            budget_so_far_usd=self.total_router_cost_usd,
            run_config=RunConfig(
                max_steps=self._limits.max_steps,
                budget_usd=self._limits.budget_usd,
                wallclock_ttl_sec=self._ttl.wallclock_ttl_sec,
                select_timeout_sec=self._limits.select_timeout_sec,
            ),
        )
        decision = validate_decision(
            self._router.select(ctx), available_models=self._available_models
        )
        model_id = decision.model_id
        cache_lookup = self._cache.lookup(model_id=model_id, messages=prefix, now_ts=now_ts)

        started_at = time.time()
        t0 = time.perf_counter()
        result = self._call_model(model_id, copy.deepcopy(list(messages)), tool_schemas)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        finished_at = time.time()

        if not isinstance(result, ModelCallResult):
            raise TypeError(
                f"call_model must return ModelCallResult, got {type(result).__name__}"
            )
        if result.raw_usage is None:
            raise RuntimeError(
                f"model {model_id!r} returned no usage payload; refusing to "
                "assume zero cost"
            )
        provider = self._provider_by_model[model_id]
        buckets = normalize_usage(provider, dict(result.raw_usage))
        step_cost = step_real_cost_usd(buckets, self._pricing.get(model_id))

        self._step_costs.append(step_cost)
        self._model_counts[model_id] += 1
        self._cache.update(
            model_id=model_id,
            messages=prefix + (dict(result.message),),
            prefix_token_count=buckets.total_prompt_tokens,
            ts=finished_at,
        )

        content = result.message.get("content")
        tool_calls = result.message.get("tool_calls") or []
        row = {
            "instance_id": self._instance_id,
            "step_index": step_index,
            "started_at": started_at,
            "finished_at": finished_at,
            "model_id": model_id,
            "provider": provider,
            "rationale": decision.rationale,
            "latency_ms": latency_ms,
            "prompt_messages_count": len(prefix),
            "tool_schema_count": len(tool_schemas),
            "usage": {
                "input_tokens": buckets.input_tokens,
                "cache_read_tokens": buckets.cache_read_tokens,
                "cache_write_tokens": buckets.cache_write_tokens,
                "output_tokens": buckets.output_tokens,
            },
            "raw_usage": dict(result.raw_usage),
            "step_cost_usd": step_cost,
            "cumulative_cost_usd": self.total_router_cost_usd,
            "assistant_content_len": len(content) if isinstance(content, str) else 0,
            "tool_call_count": len(tool_calls),
            "cache_lookup": {
                "hit": cache_lookup.hit,
                "reason": cache_lookup.reason,
                "cached_prefix_token_count": cache_lookup.cached_prefix_token_count,
            },
        }
        with self._trace_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return result.message

    def finish(
        self,
        *,
        resolved: bool,
        finished_by: str,
        agent_error: str | None = None,
        eval_error: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> Path:
        """Write the loop summary and ``results/<instance_id>.json``.

        ``resolved`` must come from the benchmark's official pass predicate.
        Errors keep their original message so ``swerouter.infra_errors`` can
        classify transport failures.
        """

        if self._finished:
            raise RuntimeError("RoutedSession.finish() called twice")
        if not isinstance(resolved, bool):
            raise TypeError(f"resolved must be bool, got {type(resolved).__name__}")
        if not finished_by:
            raise ValueError("finished_by must be non-empty")
        self._finished = True

        summary = {
            "__marker__": "loop_summary",
            "instance_id": self._instance_id,
            "step_count": len(self._step_costs),
            "finished_by": finished_by,
            "total_router_cost_usd": self.total_router_cost_usd,
            "error": agent_error,
            "model_distribution": dict(self._model_counts),
        }
        with self._trace_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(summary, ensure_ascii=False) + "\n")

        result = {
            "instance_id": self._instance_id,
            "resolved": resolved,
            "trace_path": str(self._trace_path),
            "step_count": len(self._step_costs),
            "total_router_cost_usd": self.total_router_cost_usd,
            "finished_by": finished_by,
            "model_distribution": dict(self._model_counts),
            "agent_error": agent_error,
            "eval_error": eval_error,
            "pool_fingerprint": "|".join(sorted(self._available_models)),
            "pricing_schema_version": self._pricing.schema_version,
            "ttl_policy_name": self._ttl.policy_name,
            "extra": dict(extra or {}),
        }
        path = self._run_dir / "results" / f"{self._instance_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)
        return path


__all__ = ["DEFAULT_POOL", "DEFAULT_PRICING", "DEFAULT_TTL", "ModelCallResult", "ModelCaller", "RoutedSession", "SessionLimits"]
