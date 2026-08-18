"""Response-cache bookkeeping for causal mixed-prefix search."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence

from .types import Assignment, ExecutionResult


@dataclass(frozen=True)
class CacheKey:
    instance_id: str
    target_step: int
    assignments: tuple[tuple[str, str], ...]
    generation_parameters: str


class TrialCache:
    def __init__(self) -> None:
        self._items: dict[CacheKey, ExecutionResult] = {}
        self.hits = 0
        self.misses = 0
        self.invalidations = 0

    @staticmethod
    def make_key(
        instance_id: str,
        target_step: int,
        assignments: Sequence[Assignment],
        generation_parameters: dict[str, Any],
    ) -> CacheKey:
        return CacheKey(
            instance_id=instance_id,
            target_step=target_step,
            assignments=tuple((item.tier, item.model) for item in assignments),
            generation_parameters=json.dumps(
                generation_parameters, sort_keys=True, separators=(",", ":")
            ),
        )

    def get(self, key: CacheKey) -> ExecutionResult | None:
        result = self._items.get(key)
        if result is None:
            self.misses += 1
        else:
            self.hits += 1
        return result

    def put(self, key: CacheKey, result: ExecutionResult) -> None:
        self._items[key] = result

    def invalidate_downstream(self, instance_id: str, changed_step: int) -> int:
        stale = [
            key
            for key in self._items
            if key.instance_id == instance_id and key.target_step > changed_step
        ]
        for key in stale:
            del self._items[key]
        self.invalidations += len(stale)
        return len(stale)

    def stats(self) -> dict[str, int]:
        return {
            "entries": len(self._items),
            "hits": self.hits,
            "misses": self.misses,
            "invalidations": self.invalidations,
        }
