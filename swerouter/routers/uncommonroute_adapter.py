"""``UncommonRouteRouter``: drive per-step routing with UncommonRoute v2.

Wraps :func:`uncommon_route.route` so the SWERouterBench harness can delegate
model selection to the public UncommonRoute Python API without going through
its OpenAI-compatible proxy. Pricing, capabilities, and the candidate set are
pinned to SWERouterBench's official four-model pool so upstream's default
model table cannot leak in.

Behaviour:

* Load ``data/model_pricing.json`` once at construction and translate each row
  into UncommonRoute's :class:`ModelPricing` + inferred
  :class:`ModelCapabilities`. UncommonRoute keys strictly by ``model_id``.
* At each step build a chat prompt from the full :attr:`RouterContext.messages`
  sequence (preserving tool-call turns) and call
  ``route(..., available_models=list(ctx.available_models), pricing=..., model_capabilities=...)``.
* ``RoutingDecision.model`` must be a member of ``ctx.available_models`` — any
  other value (including ``None`` or an upstream suggestion outside the pool)
  is a fail-fast ``ValueError``. No silent fallback.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from uncommon_route import route as ur_route
from uncommon_route.model_map import infer_capabilities
from uncommon_route.router.types import (
    ModelCapabilities as URModelCapabilities,
    ModelPricing as URModelPricing,
    RoutingMode,
)

from swerouter.router import RouterContext, RouterDecision


def _flatten_message_content(content: Any) -> str:
    """Return a plain-text rendering of an OpenAI-style message ``content``.

    OpenAI / Anthropic tool calls can package content as a list of blocks
    (``{"type": "text", "text": ...}``); we join just the text. Non-text
    blocks are serialized to JSON so the router still sees their structure.
    """

    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
                    continue
                parts.append(json.dumps(block, ensure_ascii=False))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def _build_prompt(messages: tuple[Mapping[str, Any], ...]) -> tuple[str, str | None]:
    """Extract (prompt, system_prompt) for UncommonRoute from chat messages.

    ``system_prompt`` is the first system turn (if any). ``prompt`` is the
    last non-system turn's text so UncommonRoute's token estimate + structural
    signal see the request-side text rather than only early context. Tool /
    assistant history is still passed in via ``messages=`` so the v2 embedding
    signal can score the full trajectory.
    """

    system_prompt: str | None = None
    last_text = ""
    for msg in messages:
        role = msg.get("role")
        text = _flatten_message_content(msg.get("content"))
        if role == "system" and system_prompt is None:
            system_prompt = text
            continue
        if text:
            last_text = text
    return last_text, system_prompt


def _messages_for_ur(messages: tuple[Mapping[str, Any], ...]) -> list[dict[str, Any]]:
    """Deep-copy frozen chat messages into plain ``list[dict]`` for UncommonRoute.

    UncommonRoute expects a mutable list of dicts; the harness supplies a
    tuple of ``Mapping`` objects. We do a shallow copy of each mapping so
    UncommonRoute cannot inadvertently mutate the harness-owned context.
    """

    out: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, Mapping):
            raise TypeError(f"expected Mapping in messages, got {type(msg).__name__}")
        out.append(dict(msg))
    return out


def _pricing_from_pool(
    pool_model_ids: list[str],
    pricing_path: Path,
) -> tuple[dict[str, URModelPricing], dict[str, URModelCapabilities]]:
    """Load SWERouterBench pricing JSON and build UncommonRoute-native tables.

    Raises
    ------
    FileNotFoundError / ValueError / KeyError
        If the pricing file is missing or does not cover every pool entry.
    """

    if not pricing_path.is_file():
        raise FileNotFoundError(f"model_pricing.json not found: {pricing_path}")
    with pricing_path.open("r", encoding="utf-8") as fh:
        doc = json.load(fh)
    raw = doc.get("pricing")
    if not isinstance(raw, dict):
        raise ValueError("model_pricing.json missing .pricing object")

    missing = [mid for mid in pool_model_ids if mid not in raw]
    if missing:
        raise ValueError(
            f"model_pricing.json missing entries for pool models: {missing}"
        )

    ur_pricing: dict[str, URModelPricing] = {}
    ur_caps: dict[str, URModelCapabilities] = {}
    for mid in pool_model_ids:
        entry = raw[mid]
        price = URModelPricing(
            input_price=float(entry["input_per_m"]),
            output_price=float(entry["output_per_m"]),
            cached_input_price=float(entry["cache_read_per_m"]),
            cache_write_price=float(entry["cache_write_per_m"]),
        )
        ur_pricing[mid] = price
        ur_caps[mid] = infer_capabilities(mid, price, has_explicit_pricing=True)
    return ur_pricing, ur_caps


@dataclass
class UncommonRouteRouter:
    """Adapter from SWERouterBench's per-step protocol to UncommonRoute's ``route``.

    Parameters
    ----------
    pricing_path
        Path to ``data/dynamic/model_pricing.json`` (under ``TwinRouterBench/``). The adapter reads
        this once to build UncommonRoute-native pricing + capability tables so
        UncommonRoute scores candidates against the official pool's real cost
        structure rather than its bundled defaults.
    routing_mode
        ``"auto"`` (default), ``"fast"``, or ``"best"``. Forwarded verbatim to
        ``uncommon_route.route(routing_mode=...)``.
    label
        Human-readable identifier written into per-step ``rationale``.
    """

    pricing_path: Path
    routing_mode: str = "auto"
    label: str = "uncommon_route"

    def __post_init__(self) -> None:
        try:
            RoutingMode(self.routing_mode)
        except ValueError as exc:
            raise ValueError(
                f"UncommonRouteRouter[{self.label!r}]: invalid routing_mode="
                f"{self.routing_mode!r}; expected one of "
                f"{tuple(m.value for m in RoutingMode)}"
            ) from exc

        object.__setattr__(self, "_pricing_path", Path(self.pricing_path))
        object.__setattr__(self, "_pricing_cache", None)

    def _ensure_tables(
        self, available_models: tuple[str, ...]
    ) -> tuple[dict[str, URModelPricing], dict[str, URModelCapabilities]]:
        cache = getattr(self, "_pricing_cache")
        if cache is not None and cache[0] == available_models:
            return cache[1], cache[2]
        pool_ids = list(available_models)
        ur_pricing, ur_caps = _pricing_from_pool(pool_ids, self._pricing_path)
        object.__setattr__(
            self,
            "_pricing_cache",
            (available_models, ur_pricing, ur_caps),
        )
        return ur_pricing, ur_caps

    def select(self, ctx: RouterContext) -> RouterDecision:
        ur_pricing, ur_caps = self._ensure_tables(ctx.available_models)
        prompt, system_prompt = _build_prompt(ctx.messages)
        messages = _messages_for_ur(ctx.messages)

        started = time.perf_counter()
        decision = ur_route(
            prompt=prompt,
            system_prompt=system_prompt,
            messages=messages,
            routing_mode=self.routing_mode,
            available_models=list(ctx.available_models),
            pricing=ur_pricing,
            model_capabilities=ur_caps,
            max_output_tokens=4096,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        model_id = decision.model
        if not isinstance(model_id, str) or not model_id:
            raise ValueError(
                f"UncommonRouteRouter[{self.label!r}] step={ctx.step_index} "
                f"returned empty model in RoutingDecision"
            )
        if model_id not in ctx.available_models:
            raise ValueError(
                f"UncommonRouteRouter[{self.label!r}] step={ctx.step_index} "
                f"picked model_id={model_id!r} which is not in pool "
                f"{list(ctx.available_models)}"
            )

        rationale = (
            f"ur[{self.routing_mode}] tier={decision.tier.value} "
            f"conf={decision.confidence:.3f} elapsed_ms={elapsed_ms:.1f} "
            f"signals={decision.reasoning[:200]}"
        )
        return RouterDecision(model_id=model_id, rationale=rationale)

    @classmethod
    def from_cli_args(
        cls,
        *,
        pricing_path: str,
        routing_mode: str = "auto",
        label: str = "uncommon_route",
    ) -> "UncommonRouteRouter":
        """CLI-friendly factory; all kwargs are forwarded as strings."""

        return cls(
            pricing_path=Path(pricing_path),
            routing_mode=routing_mode,
            label=label,
        )
