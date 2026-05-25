"""Router protocol, context, decision and reference wrappers.

This module is intentionally dependency-free (standard library only) so router
implementers can unit-test against it without pulling in ``docker`` or
``swebench``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, runtime_checkable


@dataclass(frozen=True)
class ModelCacheView:
    """Read-only view of a single model's cache state at the top of one step.

    ``prefix_token_count`` is harness-side accounting and may differ from what
    the provider eventually reports in ``usage`` (providers may count tokens
    differently). Routers should treat it as a rough hint, not a billing
    number.
    """

    model_id: str
    last_call_ts: float | None
    prefix_token_count: int
    prefix_hash: str | None


@dataclass(frozen=True)
class CacheStateSnapshot:
    """Per-step, per-model cache state snapshot passed to ``Router.select``."""

    wallclock_ttl_sec: int
    now_ts: float
    by_model: Mapping[str, ModelCacheView]

    def get(self, model_id: str) -> ModelCacheView | None:
        return self.by_model.get(model_id)


@dataclass(frozen=True)
class RunConfig:
    """Run-wide configuration visible to the router for budget-aware routing."""

    max_steps: int
    budget_usd: float
    wallclock_ttl_sec: int
    select_timeout_sec: float


@dataclass(frozen=True)
class RouterContext:
    """Context object passed to the router on every step.

    Fields are immutable. A router MUST NOT keep a reference to ``messages`` or
    ``tools`` across calls and mutate them; build internal state from your own
    copies.
    """

    instance_id: str
    step_index: int
    messages: tuple[Mapping[str, Any], ...]
    tools: tuple[Mapping[str, Any], ...]
    available_models: tuple[str, ...]
    cache_state: CacheStateSnapshot
    budget_so_far_usd: float
    run_config: RunConfig


@dataclass(frozen=True)
class RouterDecision:
    """Router output: one concrete ``model_id`` per step.

    ``model_id`` must be one of ``ctx.available_models``. ``rationale`` is a
    free-form string written into the per-step trace for analysis; it does not
    affect scoring.
    """

    model_id: str
    rationale: str | None = None


@runtime_checkable
class Router(Protocol):
    """Implement this to plug a custom router into SWERouterBench."""

    def select(self, ctx: RouterContext) -> RouterDecision:  # pragma: no cover - protocol body
        ...


def validate_decision(
    decision: Any,
    *,
    available_models: tuple[str, ...],
) -> RouterDecision:
    """Fail-fast validation for ``Router.select`` return values.

    Raises
    ------
    TypeError
        If ``decision`` is not a :class:`RouterDecision` instance.
    ValueError
        If ``decision.model_id`` is falsy or not in ``available_models``.
    """

    if not isinstance(decision, RouterDecision):
        raise TypeError(
            f"Router.select must return RouterDecision, got {type(decision).__name__}"
        )
    if not decision.model_id:
        raise ValueError("RouterDecision.model_id is empty or None")
    if decision.model_id not in available_models:
        raise ValueError(
            f"RouterDecision.model_id={decision.model_id!r} is not in the official "
            f"pool (size={len(available_models)})"
        )
    return decision


DecisionLike = str | RouterDecision


@dataclass
class FunctionRouter:
    """Wrap any callable ``f(ctx) -> str | RouterDecision`` as a :class:`Router`.

    Convenience for rule-based or sklearn-style routers that want a one-liner
    without writing a class. A bare string is auto-coerced into
    ``RouterDecision(model_id=str, rationale=None)``.
    """

    label: str
    func: Callable[[RouterContext], DecisionLike]
    rationale_prefix: str | None = field(default=None)

    def select(self, ctx: RouterContext) -> RouterDecision:
        raw = self.func(ctx)
        if isinstance(raw, RouterDecision):
            return raw
        if isinstance(raw, str):
            rationale = self.rationale_prefix
            return RouterDecision(model_id=raw, rationale=rationale)
        raise TypeError(
            f"FunctionRouter[{self.label!r}] callable must return str | RouterDecision, "
            f"got {type(raw).__name__}"
        )
