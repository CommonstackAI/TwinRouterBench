"""Question-bank row selection: full file or manifest-proportional reservoir sample."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from main.dataset import DATA_DIR, QUESTION_BANK_PATH, load_manifest


def manifest_proportional_quotas(manifest: dict[str, Any], total: int) -> dict[str, int]:
    """Largest-remainder integer quotas per benchmark source (manifest ``sources.*.line_count``)."""
    sources = manifest.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("manifest.json: missing or invalid sources")
    weights: dict[str, int] = {}
    for name in sorted(sources.keys()):
        info = sources[name]
        if not isinstance(info, dict):
            raise ValueError(f"manifest sources[{name!r}] must be an object")
        lc = info.get("line_count")
        if lc is None:
            raise ValueError(f"manifest sources[{name!r}] missing line_count")
        weights[name] = int(lc)
    wsum = sum(weights.values())
    if wsum <= 0:
        raise ValueError("sum of line_count must be positive")
    quotas: dict[str, int] = {}
    fracs: list[tuple[float, str]] = []
    for name in sorted(weights.keys()):
        exact = total * weights[name] / wsum
        q = int(exact)
        quotas[name] = q
        fracs.append((exact - q, name))
    deficit = total - sum(quotas.values())
    fracs.sort(key=lambda x: -x[0])
    for i in range(deficit):
        quotas[fracs[i][1]] += 1
    return quotas


def proportional_reservoir_sample(
    shard: Path,
    rng: random.Random,
    quotas: dict[str, int],
) -> list[dict[str, Any]]:
    """One-pass stratified reservoir sample; shuffles the combined list before return."""
    active = {b: k for b, k in quotas.items() if k > 0}
    reservoirs: dict[str, list[dict[str, Any]]] = {b: [] for b in active}
    seen: dict[str, int] = {b: 0 for b in active}

    with shard.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            b = row.get("benchmark")
            if b not in active:
                continue
            k = active[b]
            seen[b] += 1
            n = seen[b]
            pool = reservoirs[b]
            if len(pool) < k:
                pool.append(row)
                continue
            j = rng.randint(1, n)
            if j <= k:
                pool[j - 1] = row

    for b, k in active.items():
        got = len(reservoirs[b])
        if got != k:
            raise ValueError(
                f"Not enough rows for benchmark {b!r}: sampled {got}, quota {k} "
                f"(check question_bank vs manifest line_count)"
            )

    out: list[dict[str, Any]] = []
    for b in sorted(active.keys()):
        out.extend(reservoirs[b])
    rng.shuffle(out)
    return out


def load_all_question_bank_rows(shard: Path | None = None) -> list[dict[str, Any]]:
    """Load every non-empty JSONL object from the question bank (file order)."""
    path = shard if shard is not None else QUESTION_BANK_PATH
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No non-empty rows in {path}")
    return rows


def rows_per_benchmark(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Count rows per ``benchmark`` string (sorted keys)."""
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        b = r.get("benchmark")
        if not isinstance(b, str):
            rid = r.get("id")
            raise ValueError(f"row id={rid!r} missing string benchmark")
        counts[b] += 1
    return dict(sorted(counts.items()))


def select_question_bank_rows(
    *,
    n: int | None,
    seed: int,
    shard: Path | None = None,
    manifest: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], str, dict[str, int] | None]:
    """
    Return ``(rows, sample_mode, proportional_quotas)``.

    * ``n is None``: all rows in file order (``sample_mode == "full_bank"``).
    * ``n`` positive: proportional sample (``sample_mode == "proportional_sample"``).
    """
    path = shard if shard is not None else QUESTION_BANK_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Question bank not found: {path}")

    if n is None:
        return load_all_question_bank_rows(path), "full_bank", None
    if n <= 0:
        raise ValueError("n must be positive when set")

    m = manifest if manifest is not None else load_manifest()
    quotas = manifest_proportional_quotas(m, n)
    rng = random.Random(seed)
    sample = proportional_reservoir_sample(path, rng, quotas)
    return sample, "proportional_sample", quotas


def default_manifest_path() -> Path:
    """``data/static/manifest.json`` next to the shipped question bank."""
    return DATA_DIR / "manifest.json"
