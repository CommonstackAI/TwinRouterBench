"""Normalize provider-side ``usage`` payloads into 4 canonical buckets.

Different vendors report token usage differently. The leaderboard requires one
uniform 4-bucket view (input / cache_read / cache_write / output) so that
``step_real_cost_usd`` can apply per-model prices unambiguously.

Vendor mapping rules are documented in ``docs/pricing_and_cache_zh.md`` §2.
Unknown providers, missing fields, or negative counts raise (fail fast).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

SUPPORTED_PROVIDERS: frozenset[str] = frozenset(
    {"anthropic", "openai", "deepseek", "gemini", "openai_compat"}
)


@dataclass(frozen=True)
class UsageBuckets:
    """Canonical 4-bucket token usage for one LLM call.

    All four counts are non-negative integers. The sum over input + cache_read +
    cache_write is the total number of prompt tokens the provider saw; output
    is newly generated completion tokens.
    """

    input_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    output_tokens: int

    def __post_init__(self) -> None:
        for name, value in (
            ("input_tokens", self.input_tokens),
            ("cache_read_tokens", self.cache_read_tokens),
            ("cache_write_tokens", self.cache_write_tokens),
            ("output_tokens", self.output_tokens),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"UsageBuckets.{name} must be int, got {type(value).__name__}")
            if value < 0:
                raise ValueError(f"UsageBuckets.{name} must be non-negative, got {value}")

    @property
    def total_prompt_tokens(self) -> int:
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens


def _json_count(value: Any, *, field: str, provider: str) -> int:
    """Coerce JSON numeric usage counts to ``int`` (OpenRouter may emit ``1.0``)."""

    if value is None:
        return 0
    if isinstance(value, bool):
        raise TypeError(
            f"usage.{field} from provider {provider!r} must be numeric, got bool"
        )
    if isinstance(value, int):
        n = value
    elif isinstance(value, float):
        if not value.is_integer():
            raise ValueError(
                f"usage.{field} from provider {provider!r} must be whole-number tokens, got {value!r}"
            )
        n = int(value)
    else:
        raise TypeError(
            f"usage.{field} from provider {provider!r} must be int or float, got {type(value).__name__}"
        )
    if n < 0:
        raise ValueError(f"usage.{field} from provider {provider!r} is negative: {n}")
    return n


def _require_int(raw: Mapping[str, Any], key: str, *, provider: str) -> int:
    if key not in raw:
        raise ValueError(
            f"usage payload from provider {provider!r} missing required key {key!r}"
        )
    return _json_count(raw[key], field=key, provider=provider)


def _optional_int(raw: Mapping[str, Any], key: str, *, provider: str) -> int:
    if key not in raw or raw[key] is None:
        return 0
    return _json_count(raw[key], field=key, provider=provider)


def _normalize_openai(raw: Mapping[str, Any], provider_label: str) -> UsageBuckets:
    """OpenAI-style envelope plus OpenRouter extensions.

    OpenRouter forwards ``prompt_tokens_details.cached_tokens`` (cache read) and
    ``prompt_tokens_details.cache_write_tokens`` (e.g. Anthropic ephemeral
    writes). Previously ``cache_write_tokens`` was forced to ``0``, which
    mis-billed cache writes at ``input_per_m`` and diverged from OpenRouter's
    own ``usage.cost``.

    Some OpenAI-compatible stacks and LiteLLM's ``Usage.model_dump()`` attach
    Anthropic-style ``input_tokens`` / ``output_tokens`` while leaving
    ``prompt_tokens`` / ``completion_tokens`` at zero. When both canonical
    counts are zero but the aliases are non-zero, treat the aliases as the
    OpenAI prompt/completion totals for billing (no silent all-zero cost).
    """

    pt_opt = _optional_int(raw, "prompt_tokens", provider=provider_label)
    ct_opt = _optional_int(raw, "completion_tokens", provider=provider_label)
    if pt_opt == 0 and ct_opt == 0:
        it = _optional_int(raw, "input_tokens", provider=provider_label)
        ot = _optional_int(raw, "output_tokens", provider=provider_label)
        if it or ot:
            prompt_tokens, completion_tokens = it, ot
        else:
            prompt_tokens = _require_int(raw, "prompt_tokens", provider=provider_label)
            completion_tokens = _require_int(raw, "completion_tokens", provider=provider_label)
    else:
        prompt_tokens = _require_int(raw, "prompt_tokens", provider=provider_label)
        completion_tokens = _require_int(raw, "completion_tokens", provider=provider_label)
    cached = 0
    cache_write = 0
    details = raw.get("prompt_tokens_details")
    if isinstance(details, Mapping):
        cached = _optional_int(details, "cached_tokens", provider=provider_label)
        cache_write = _optional_int(
            details, "cache_write_tokens", provider=provider_label
        )
    elif details is not None:
        raise TypeError(
            f"usage.prompt_tokens_details from provider {provider_label!r} must be object, "
            f"got {type(details).__name__}"
        )
    billed_prefix = cached + cache_write
    if billed_prefix > prompt_tokens:
        raise ValueError(
            f"usage cached_tokens+cache_write_tokens ({cached}+{cache_write}) exceeds "
            f"prompt_tokens ({prompt_tokens}) for provider {provider_label!r}"
        )
    return UsageBuckets(
        input_tokens=prompt_tokens - billed_prefix,
        cache_read_tokens=cached,
        cache_write_tokens=cache_write,
        output_tokens=completion_tokens,
    )


def _normalize_anthropic(raw: Mapping[str, Any]) -> UsageBuckets:
    return UsageBuckets(
        input_tokens=_require_int(raw, "input_tokens", provider="anthropic"),
        cache_read_tokens=_optional_int(raw, "cache_read_input_tokens", provider="anthropic"),
        cache_write_tokens=_optional_int(raw, "cache_creation_input_tokens", provider="anthropic"),
        output_tokens=_require_int(raw, "output_tokens", provider="anthropic"),
    )


def _normalize_deepseek(raw: Mapping[str, Any]) -> UsageBuckets:
    hit = _optional_int(raw, "prompt_cache_hit_tokens", provider="deepseek")
    miss_key = (
        "prompt_cache_miss_tokens"
        if "prompt_cache_miss_tokens" in raw
        else "prompt_tokens"
    )
    if miss_key == "prompt_tokens":
        total = _require_int(raw, "prompt_tokens", provider="deepseek")
        if hit > total:
            raise ValueError(
                f"deepseek prompt_cache_hit_tokens ({hit}) exceeds prompt_tokens ({total})"
            )
        miss = total - hit
    else:
        miss = _require_int(raw, "prompt_cache_miss_tokens", provider="deepseek")
    completion = _require_int(raw, "completion_tokens", provider="deepseek")
    return UsageBuckets(
        input_tokens=miss,
        cache_read_tokens=hit,
        cache_write_tokens=0,
        output_tokens=completion,
    )


def _normalize_gemini(raw: Mapping[str, Any]) -> UsageBuckets:
    prompt = _require_int(raw, "prompt_token_count", provider="gemini")
    cached = _optional_int(raw, "cached_content_token_count", provider="gemini")
    if cached > prompt:
        raise ValueError(
            f"gemini cached_content_token_count ({cached}) exceeds prompt_token_count ({prompt})"
        )
    # Gemini may report candidates_token_count OR output_token_count across
    # API variants; prefer explicit candidates_token_count.
    if "candidates_token_count" in raw:
        output = _require_int(raw, "candidates_token_count", provider="gemini")
    else:
        output = _require_int(raw, "output_token_count", provider="gemini")
    return UsageBuckets(
        input_tokens=prompt - cached,
        cache_read_tokens=cached,
        cache_write_tokens=0,
        output_tokens=output,
    )


def normalize_usage(provider: str, raw_usage: Mapping[str, Any] | None) -> UsageBuckets:
    """Map a vendor-specific ``usage`` payload to :class:`UsageBuckets`.

    Parameters
    ----------
    provider
        One of ``anthropic``, ``openai``, ``deepseek``, ``gemini``, ``openai_compat``.
    raw_usage
        The ``usage`` object returned by the vendor (or the aggregator).

    Raises
    ------
    ValueError
        For unknown providers, missing required keys, or negative counts.
    TypeError
        For wrong-typed fields.
    """

    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported provider {provider!r}; supported: {sorted(SUPPORTED_PROVIDERS)}"
        )
    if raw_usage is None:
        raise ValueError(f"provider {provider!r} returned no usage payload")
    if not isinstance(raw_usage, Mapping):
        raise TypeError(
            f"raw_usage must be a mapping, got {type(raw_usage).__name__}"
        )

    if provider == "anthropic":
        return _normalize_anthropic(raw_usage)
    if provider == "deepseek":
        return _normalize_deepseek(raw_usage)
    if provider == "gemini":
        return _normalize_gemini(raw_usage)
    return _normalize_openai(raw_usage, provider)
