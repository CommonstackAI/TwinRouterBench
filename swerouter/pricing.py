"""Real per-model pricing and 4-bucket USD cost computation.

Prices live exclusively in ``data/model_pricing.json``. This module does not
embed any vendor price as a Python literal (user rule: no hardcoded business
values). Any unknown ``model_id`` or malformed entry raises (user rule: fail
fast, no silent fallbacks).

See ``docs/pricing_and_cache_zh.md`` for schema documentation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from swerouter.usage import UsageBuckets

SUPPORTED_PRICING_SCHEMA_VERSIONS: frozenset[int] = frozenset({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12})


@dataclass(frozen=True)
class ModelPricing:
    """Four published unit prices for one ``model_id`` in USD / 1M tokens."""

    model_id: str
    input_per_m: float
    output_per_m: float
    cache_read_per_m: float
    cache_write_per_m: float
    source_url: str

    @classmethod
    def from_json(cls, model_id: str, obj: Mapping[str, object]) -> "ModelPricing":
        required = (
            "input_per_m",
            "output_per_m",
            "cache_read_per_m",
            "cache_write_per_m",
            "source_url",
        )
        missing = [k for k in required if k not in obj]
        if missing:
            raise ValueError(
                f"pricing entry for {model_id!r} missing required keys: {missing}"
            )
        for k in required[:-1]:
            v = obj[k]
            if not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0:
                raise ValueError(
                    f"pricing entry for {model_id!r}: {k} must be non-negative number, got {v!r}"
                )
        source_url = obj["source_url"]
        if not isinstance(source_url, str) or not source_url:
            raise ValueError(
                f"pricing entry for {model_id!r}: source_url must be a non-empty string"
            )
        return cls(
            model_id=model_id,
            input_per_m=float(obj["input_per_m"]),
            output_per_m=float(obj["output_per_m"]),
            cache_read_per_m=float(obj["cache_read_per_m"]),
            cache_write_per_m=float(obj["cache_write_per_m"]),
            source_url=source_url,
        )


@dataclass(frozen=True)
class PricingTable:
    """Immutable lookup table loaded from ``data/model_pricing.json``."""

    schema_version: int
    fetched_at: str
    currency: str
    unit: str
    _by_model: Mapping[str, ModelPricing]

    def __contains__(self, model_id: object) -> bool:
        return isinstance(model_id, str) and model_id in self._by_model

    def __iter__(self):
        return iter(self._by_model)

    def items(self):
        return self._by_model.items()

    def get(self, model_id: str) -> ModelPricing:
        """Return pricing for ``model_id`` or raise ``KeyError`` (fail fast)."""
        if model_id not in self._by_model:
            raise KeyError(
                f"Unknown model_id {model_id!r} for pricing table (schema v{self.schema_version})."
                f" Pool size={len(self._by_model)}. Pricing must be added to data/model_pricing.json"
                f" before routing may use this model."
            )
        return self._by_model[model_id]


def load_pricing_table(path: str | Path) -> PricingTable:
    """Parse ``data/model_pricing.json`` and return an immutable :class:`PricingTable`.

    Raises ``FileNotFoundError``, ``json.JSONDecodeError``, or ``ValueError`` on
    any malformed content. There is no default fallback table.
    """

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"pricing file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        doc = json.load(fh)

    if not isinstance(doc, dict):
        raise ValueError(f"pricing file root must be object, got {type(doc).__name__}")

    schema_version = doc.get("schema_version")
    if schema_version not in SUPPORTED_PRICING_SCHEMA_VERSIONS:
        raise ValueError(
            f"pricing schema_version {schema_version!r} not supported "
            f"(supported: {sorted(SUPPORTED_PRICING_SCHEMA_VERSIONS)})"
        )

    for key in ("fetched_at", "currency", "unit", "pricing"):
        if key not in doc:
            raise ValueError(f"pricing file missing top-level key: {key!r}")

    raw_pricing = doc["pricing"]
    if not isinstance(raw_pricing, dict) or not raw_pricing:
        raise ValueError("pricing.pricing must be a non-empty object")

    entries: dict[str, ModelPricing] = {}
    for model_id, obj in raw_pricing.items():
        if not isinstance(obj, dict):
            raise ValueError(
                f"pricing entry {model_id!r} must be an object, got {type(obj).__name__}"
            )
        entries[model_id] = ModelPricing.from_json(model_id, obj)

    return PricingTable(
        schema_version=int(schema_version),
        fetched_at=str(doc["fetched_at"]),
        currency=str(doc["currency"]),
        unit=str(doc["unit"]),
        _by_model=entries,
    )


def step_real_cost_usd(usage: UsageBuckets, pricing: ModelPricing) -> float:
    """Compute per-step USD cost from 4-bucket usage + per-model prices.

    All token counts must be non-negative integers (already enforced in
    :class:`UsageBuckets.__post_init__`). Returns cost in USD.
    """

    return (
        usage.input_tokens * pricing.input_per_m
        + usage.cache_read_tokens * pricing.cache_read_per_m
        + usage.cache_write_tokens * pricing.cache_write_per_m
        + usage.output_tokens * pricing.output_per_m
    ) / 1_000_000.0
