"""mini-swe-agent ``Model`` implementation with per-step router dispatch
and SWERouterBench-locked four-bucket pricing.

mini's own ``LitellmModel`` (``minisweagent/models/litellm_model.py``) assumes
a single model for the whole run and computes cost via litellm's bundled
price table. That's a problem for leaderboard fairness because:

1. Routers need to pick a different ``model_id`` on every step, but
   ``LitellmModel`` is constructed with one ``model_name``.
2. litellm's prices are not pinned to SWERouterBench's ``pricing_fingerprint``
   (the published OpenRouter snapshot in
   ``data/dynamic/model_pricing.json``), so costs could drift between
   runs without any code change.

:class:`RouterAwareModel` fixes both:

* It pre-instantiates one ``LitellmModel`` per pool entry (cheap: no network
  calls), so we can dispatch to the right sub-model for each step.
* It disables litellm's cost calculator
  (``cost_tracking="ignore_errors"``) and re-computes USD locally from the
  provider's ``usage`` payload using :mod:`swerouter.pricing` +
  :mod:`swerouter.usage`. The resulting ``message["extra"]["cost"]`` is the
  number the leaderboard sums.

Template-level operations (``format_message`` /
``format_observation_messages`` / ``get_template_vars``) are delegated to a
single template sub-model because their behaviour (multimodal regex,
observation template) doesn't depend on which pool member runs a given
step -- all pool members share the same mini BASH_TOOL schema.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig

from swerouter.agent.loop import ModelPoolEntry
from swerouter.pricing import PricingTable
from swerouter.usage import UsageBuckets, normalize_usage


@dataclass
class RouterAwareModelConfig:
    """Config surface echoed into mini's trajectory serialization.

    mini stores ``config`` on every Model instance and dumps it under
    ``info.config.model`` in the saved trajectory. We keep the same shape so
    the trajectory inspector renders correctly.
    """

    pool_model_ids: list[str] = field(default_factory=list)
    pricing_schema_version: int = 0
    default_model_kwargs: dict[str, Any] = field(default_factory=dict)

    def model_dump(self, *, mode: str = "python") -> dict[str, Any]:
        _ = mode
        return {
            "pool_model_ids": list(self.pool_model_ids),
            "pricing_schema_version": self.pricing_schema_version,
            "default_model_kwargs": dict(self.default_model_kwargs),
        }


@dataclass
class StepRecord:
    """Internal per-step log kept by :class:`RouterAwareModel` so the agent
    can write SWERouterBench-compatible trace rows after ``query_as``.
    """

    step_index: int
    model_id: str
    provider: str
    usage: UsageBuckets
    raw_usage: dict[str, Any]
    step_cost_usd: float
    latency_ms: float
    started_at: float
    finished_at: float


class RouterAwareModel:
    """mini ``Model`` Protocol implementation that dispatches per step."""

    # Sub-model construction kwargs that we always set. Callers can merge
    # more kwargs via the ``default_model_kwargs`` argument.
    _PROVIDER_FOR_PRICING = "openai_compat"

    def __init__(
        self,
        pool: list[ModelPoolEntry],
        pricing: PricingTable,
        *,
        base_url: str,
        api_key: str,
        default_model_kwargs: Mapping[str, Any] | None = None,
        observation_template: str | None = None,
        format_error_template: str | None = None,
    ) -> None:
        if not pool:
            raise ValueError("RouterAwareModel requires a non-empty pool")
        for entry in pool:
            if entry.model_id not in pricing:
                raise KeyError(
                    f"pool model {entry.model_id!r} missing from pricing "
                    f"table schema v{pricing.schema_version}"
                )
        self._pool = {p.model_id: p for p in pool}
        self._pricing = pricing
        self._base_url = base_url
        self._api_key = api_key
        self._default_model_kwargs = dict(default_model_kwargs or {})
        self._observation_template = observation_template
        self._format_error_template = format_error_template

        self._submodels: dict[str, LitellmModel] = {}
        for entry in pool:
            self._submodels[entry.model_id] = self._build_submodel(entry)
        # Pick a stable template sub-model for formatting (first in pool).
        self._template_submodel = self._submodels[pool[0].model_id]

        self.config = RouterAwareModelConfig(
            pool_model_ids=list(self._submodels.keys()),
            pricing_schema_version=pricing.schema_version,
            default_model_kwargs=dict(self._default_model_kwargs),
        )

        self._step_log: list[StepRecord] = []

    @property
    def pricing_table(self) -> PricingTable:
        """Locked :class:`PricingTable` (for trace-side audit helpers)."""

        return self._pricing

    # ------------------------------------------------------------------
    # sub-model construction
    # ------------------------------------------------------------------

    def _openai_compat_litellm_model_id(self, pool_model_id: str) -> str:
        """Optional remap of pool ``model_id`` for OpenAI-compat gateways.

        Some proxies expect different vendor strings than OpenRouter's catalogue
        (e.g. hyphen vs dot in Claude IDs). Set::

            MINI_OPENAI_COMPAT_MODEL_ID_ALIASES_JSON='{"pool-id":"gateway-id"}'

        (single-line JSON object). Unmapped IDs pass through unchanged.
        """

        raw = (os.environ.get("MINI_OPENAI_COMPAT_MODEL_ID_ALIASES_JSON") or "").strip()
        if not raw:
            return pool_model_id
        try:
            aliases = json.loads(raw)
        except json.JSONDecodeError:
            return pool_model_id
        if not isinstance(aliases, dict):
            return pool_model_id
        out = aliases.get(pool_model_id)
        return str(out) if isinstance(out, str) and out else pool_model_id

    def _litellm_model_name(self, entry: ModelPoolEntry) -> str:
        """Pick the LiteLLM ``model=`` prefix for ``model_kwargs.api_base``.

        ``openrouter/...`` keeps OpenRouter-specific client behaviour (headers,
        transforms). For OpenAI-compatible gateways (CommonStack, local vLLM,
        etc.) use ``openai/...`` so LiteLLM sends a plain ``/v1/chat/completions``
        POST with ``Authorization: Bearer <api_key>`` and ``model`` equal to
        the pool ``model_id`` string after the prefix.

        Override: ``SWEROUTER_LITELLM_PROVIDER_PREFIX=openai`` or ``openrouter``.
        """

        mid = self._openai_compat_litellm_model_id(entry.model_id)
        if mid.startswith("openrouter/") or mid.startswith("openai/"):
            return mid
        override = (os.environ.get("SWEROUTER_LITELLM_PROVIDER_PREFIX") or "").strip().lower()
        if override == "openai":
            return f"openai/{mid}"
        if override == "openrouter":
            return f"openrouter/{mid}"
        base = (self._base_url or "").lower()
        if "openrouter.ai" in base:
            return f"openrouter/{mid}"
        return f"openai/{mid}"

    def _build_submodel(self, entry: ModelPoolEntry) -> LitellmModel:
        """Construct one ``LitellmModel`` per pool member.

        We always set ``cost_tracking="ignore_errors"`` because we compute
        USD ourselves from :mod:`swerouter.pricing`. Anthropic-style entries
        also get ``set_cache_control="default_end"`` so litellm injects
        ``cache_control: ephemeral`` markers automatically, matching the
        locked prompt-cache semantics documented in
        ``swerouter/docs/pricing_and_cache_zh.md``.
        """

        cfg_kwargs: dict[str, Any] = {
            "model_name": self._litellm_model_name(entry),
            "model_kwargs": {
                "api_base": self._base_url,
                "api_key": self._api_key,
                **self._default_model_kwargs,
            },
            "cost_tracking": "ignore_errors",
        }
        if entry.cache_control_style == "anthropic":
            cfg_kwargs["set_cache_control"] = "default_end"
        if self._observation_template is not None:
            cfg_kwargs["observation_template"] = self._observation_template
        if self._format_error_template is not None:
            cfg_kwargs["format_error_template"] = self._format_error_template

        return LitellmModel(config_class=LitellmModelConfig, **cfg_kwargs)

    # ------------------------------------------------------------------
    # mini Model Protocol
    # ------------------------------------------------------------------

    @property
    def available_models(self) -> tuple[str, ...]:
        return tuple(self._submodels.keys())

    @property
    def step_log(self) -> list[StepRecord]:
        return list(self._step_log)

    def query(self, messages: list[dict[str, Any]], **kwargs) -> dict:
        """mini calls this when no router override is in effect.

        We require the agent to always go through :meth:`query_as` so every
        step has an explicit ``model_id``. Plain ``query()`` is therefore a
        hard error -- silently falling back to a default model would make
        the leaderboard number depend on which pool entry happened to be
        first, which violates the "fail fast" contract.
        """

        raise RuntimeError(
            "RouterAwareModel.query() must not be called directly; "
            "use MiniRouterAgent which dispatches via query_as(model_id, ...). "
            "If you're seeing this, DefaultAgent.query() wasn't overridden."
        )

    def query_as(
        self,
        model_id: str,
        messages: list[dict[str, Any]],
        *,
        step_index: int,
        **kwargs,
    ) -> dict:
        """Dispatch one step to a specific ``model_id``.

        Returns the same message shape mini's ``LitellmModel.query`` produces
        (``{"role", "content"/"tool_calls", "extra": {"actions", "response",
        "cost", "timestamp"}}``) but with ``extra["cost"]`` overwritten by
        our :mod:`swerouter.pricing` computation and an extra
        ``extra["swerouter"]`` section containing normalized usage buckets
        and step bookkeeping for trace emission.
        """

        if model_id not in self._submodels:
            raise KeyError(
                f"router picked model_id={model_id!r} but it is not in the "
                f"pool {list(self._submodels)}"
            )
        submodel = self._submodels[model_id]
        pool_entry = self._pool[model_id]

        started_at = time.time()
        t0 = time.perf_counter()
        message = submodel.query(messages, **kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        finished_at = time.time()

        extra = dict(message.get("extra") or {})
        raw_response = extra.get("response") or {}
        raw_usage = raw_response.get("usage") if isinstance(raw_response, dict) else None
        if raw_usage is None:
            raise RuntimeError(
                f"model {model_id!r} returned no usage payload; refusing to "
                "silently assume zero cost"
            )

        buckets = normalize_usage(pool_entry.provider, raw_usage)
        step_cost_usd = _step_cost_from_buckets(self._pricing, model_id, buckets)

        extra["cost"] = step_cost_usd
        extra["swerouter"] = {
            "step_index": step_index,
            "model_id": model_id,
            "provider": pool_entry.provider,
            "usage": {
                "input_tokens": buckets.input_tokens,
                "cache_read_tokens": buckets.cache_read_tokens,
                "cache_write_tokens": buckets.cache_write_tokens,
                "output_tokens": buckets.output_tokens,
            },
            "raw_usage": dict(raw_usage),
            "step_cost_usd": step_cost_usd,
            "latency_ms": latency_ms,
            "started_at": started_at,
            "finished_at": finished_at,
            "pricing_schema_version": self._pricing.schema_version,
        }
        message["extra"] = extra

        self._step_log.append(
            StepRecord(
                step_index=step_index,
                model_id=model_id,
                provider=pool_entry.provider,
                usage=buckets,
                raw_usage=dict(raw_usage),
                step_cost_usd=step_cost_usd,
                latency_ms=latency_ms,
                started_at=started_at,
                finished_at=finished_at,
            )
        )
        return message

    def format_message(self, **kwargs) -> dict:
        """Delegate to template sub-model -- all pool members share the
        same bash tool schema, so formatting is model-agnostic."""

        return self._template_submodel.format_message(**kwargs)

    def format_observation_messages(
        self,
        message: dict,
        outputs: list[dict],
        template_vars: dict | None = None,
    ) -> list[dict]:
        return self._template_submodel.format_observation_messages(
            message, outputs, template_vars
        )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        base = self._template_submodel.get_template_vars(**kwargs)
        base["router_pool_model_ids"] = list(self._submodels.keys())
        return base

    def serialize(self) -> dict:
        """Payload stored under ``info.config.model`` in mini's trajectory."""

        return {
            "info": {
                "config": {
                    "model": self.config.model_dump(mode="json"),
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
            }
        }


def _step_cost_from_buckets(
    pricing: PricingTable, model_id: str, buckets: UsageBuckets
) -> float:
    """Four-bucket cost in USD using the locked pricing table.

    Duplicates the arithmetic of :func:`swerouter.pricing.step_real_cost_usd`
    but keeps the implementation local so this module stays independent of
    the harness-side per-step API (which takes a whole cache lookup result).
    """

    mp = pricing.get(model_id)
    return (
        buckets.input_tokens * mp.input_per_m
        + buckets.cache_read_tokens * mp.cache_read_per_m
        + buckets.cache_write_tokens * mp.cache_write_per_m
        + buckets.output_tokens * mp.output_per_m
    ) / 1_000_000.0


__all__ = [
    "RouterAwareModel",
    "RouterAwareModelConfig",
    "StepRecord",
]
