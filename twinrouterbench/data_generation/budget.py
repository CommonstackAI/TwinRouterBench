"""A process-safe, append-only spend cap for live validation requests.

An uncertain/failed request keeps its full reservation. It is never silently
refunded or retried. Pricing is supplied from a saved provider catalog.
"""

from __future__ import annotations

import json
import math
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

try:
    import fcntl
except ImportError:  # Optional live budgeting must not break other backends.
    fcntl = None


class BudgetExceeded(RuntimeError):
    pass


class ApiBudget:
    def __init__(self, path: str | Path, limit_usd: float) -> None:
        if fcntl is None:
            raise RuntimeError("API budgeting requires POSIX file locking (Linux/macOS)")
        if not math.isfinite(limit_usd) or limit_usd <= 0:
            raise ValueError("API budget must be finite and positive")
        self.path = Path(path).expanduser()
        self.limit_usd = limit_usd
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _locked(self) -> Iterator[Any]:
        with self.path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield handle
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    @staticmethod
    def _entries(handle: Any) -> dict[str, dict[str, Any]]:
        handle.seek(0)
        entries: dict[str, dict[str, Any]] = {}
        for line in handle:
            row = json.loads(line)
            if row["event"] == "reserve":
                entries[row["id"]] = row
            else:
                entries[row["id"]].update(row)
        return entries

    @staticmethod
    def _append(handle: Any, row: dict[str, Any]) -> None:
        handle.seek(0, 2)
        handle.write(json.dumps({"timestamp": time.time(), **row}, sort_keys=True) + "\n")
        handle.flush()
        import os
        os.fsync(handle.fileno())

    def reserve(self, model: str, maximum_usd: float) -> str:
        if not math.isfinite(maximum_usd) or maximum_usd <= 0:
            raise ValueError("request reservation must be finite and positive")
        with self._locked() as handle:
            entries = self._entries(handle)
            spent = sum(row.get("accounted_usd", row["reserved_usd"]) for row in entries.values())
            if spent + maximum_usd > self.limit_usd:
                raise BudgetExceeded(
                    f"API cap ${self.limit_usd:.2f}: ${spent:.6f} already accounted, "
                    f"next request needs at most ${maximum_usd:.6f}"
                )
            request_id = uuid.uuid4().hex
            self._append(handle, dict(event="reserve", id=request_id, model=model,
                                      reserved_usd=maximum_usd, limit_usd=self.limit_usd))
            return request_id

    def finish(self, request_id: str, *, usage: dict[str, Any], pricing: dict[str, Any],
               provider: str = "") -> None:
        prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
        output = usage.get("completion_tokens", usage.get("output_tokens"))
        if type(prompt) is not int or type(output) is not int or min(prompt, output) < 0:
            raise RuntimeError("missing/invalid provider usage; full reservation retained")
        if "prompt_tokens" not in usage:
            for field in ("cache_read_input_tokens", "cache_creation_input_tokens"):
                cached = usage.get(field, 0)
                if type(cached) is not int or cached < 0:
                    raise RuntimeError("invalid cached input usage; full reservation retained")
                prompt += cached
        # Count cached input at the full input rate conservatively. Cache writes
        # can carry an extra charge; reserve/settle at the largest published rate.
        input_rate = max(float(pricing.get(key, 0)) for key in (
            "prompt", "input_cache_writes", "input_cache_writes_1h"
        ))
        computed = prompt * input_rate + output * float(pricing["completion"])
        reported = usage.get("cost")
        if reported is not None:
            reported = float(reported)
            if not math.isfinite(reported) or reported < 0:
                raise RuntimeError("invalid reported cost; full reservation retained")
        accounted = max(computed, reported or 0)
        with self._locked() as handle:
            entry = self._entries(handle)[request_id]
            if entry["event"] != "reserve":
                raise RuntimeError("request already settled")
            if accounted > entry["reserved_usd"]:
                # Keep the larger observed expense and block further calls.
                self._append(handle, dict(event="overrun", id=request_id,
                    accounted_usd=max(accounted, self.limit_usd), usage=usage,
                    reported_cost_usd=reported, conservative_cost_usd=computed))
                raise BudgetExceeded("provider usage exceeded reservation; further calls blocked")
            self._append(handle, dict(event="settle", id=request_id,
                accounted_usd=accounted, reported_cost_usd=reported,
                conservative_cost_usd=computed, usage=usage, provider=provider))

    def summary(self) -> dict[str, Any]:
        with self._locked() as handle:
            entries = self._entries(handle)
            return {
                "limit_usd": self.limit_usd,
                "requests": len(entries),
                "pending_or_failed": sum(row["event"] == "reserve" for row in entries.values()),
                "accounted_usd": sum(row.get("accounted_usd", row["reserved_usd"]) for row in entries.values()),
                "reported_usd": sum(row.get("reported_cost_usd") or 0 for row in entries.values()),
            }


def request_reservation(body: dict[str, Any], pricing: dict[str, Any]) -> float:
    if pricing.get("currency", "USD") != "USD":
        raise ValueError("budget requires USD provider prices")
    # UTF-8 bytes upper-bound text tokens; the margin covers message/tool
    # framing. This validator is deliberately restricted to textual requests.
    if any(marker in json.dumps(body) for marker in ('"image_url"', '"input_audio"')):
        raise ValueError("multimodal requests need a separate pricing bound")
    input_bound = len(json.dumps(body, ensure_ascii=False).encode()) + 4096
    if "max_tokens" in body and "max_completion_tokens" in body:
        raise ValueError("ambiguous output token limits")
    max_output = body.get("max_tokens", body.get("max_completion_tokens"))
    if type(max_output) is not int or max_output <= 0:
        raise ValueError("budgeted requests need a positive output token limit")
    rates = [float(pricing.get(key, 0)) for key in (
        "prompt", "input_cache_writes", "input_cache_writes_1h"
    )]
    completion = float(pricing["completion"])
    if any(not math.isfinite(rate) or rate < 0 for rate in rates + [completion]):
        raise ValueError("invalid provider pricing")
    return max(0.000001, (input_bound * max(rates) + max_output * completion) * 1.1)
