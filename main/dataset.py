"""Stream tier-only records from the single-file question bank under data/static/."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# TwinRouterBench install root: parent of the inner `main` package directory.
_OPEN_ROOT = Path(__file__).resolve().parent.parent
# Static-track shipped data (question bank + manifest) lives next to dynamic lock files.
STATIC_DATA_DIR = _OPEN_ROOT / "data" / "static"
DATA_DIR = STATIC_DATA_DIR
QUESTION_BANK_NAME = "question_bank.jsonl"
QUESTION_BANK_PATH = DATA_DIR / QUESTION_BANK_NAME


def _question_bank_path() -> Path:
    if not QUESTION_BANK_PATH.is_file():
        raise FileNotFoundError(
            f"Missing question bank: {QUESTION_BANK_PATH} "
            "(install or restore data/static/question_bank.jsonl under TwinRouterBench/)"
        )
    return QUESTION_BANK_PATH


def list_question_bank_sources() -> list[str]:
    """Logical source names from manifest (e.g. swebench, mtrag), not filesystem folders."""
    try:
        m = load_manifest()
        src = m.get("sources")
        if isinstance(src, dict):
            return sorted(src.keys())
    except FileNotFoundError:
        pass
    return []


def list_benchmarks() -> list[str]:
    """Deprecated alias for :func:`list_question_bank_sources` (same logical names as row[\"benchmark\"])."""
    return list_question_bank_sources()


def iter_question_bank(*, benchmark: str | None = None) -> Iterator[dict[str, Any]]:
    """
    Stream JSON objects from ``data/static/question_bank.jsonl``.

    If ``benchmark`` is set, only yield rows whose ``benchmark`` field equals that value
    (e.g. ``\"swebench\"``, ``\"mtrag\"``). The bank file itself is not split by directory.
    """
    path = _question_bank_path()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if benchmark is not None and row.get("benchmark") != benchmark:
                continue
            yield row


def iter_routing_supervision(benchmark: str) -> Iterator[dict[str, Any]]:
    """Yield rows for one logical ``benchmark`` value (same as ``iter_question_bank(benchmark=...)``)."""
    return iter_question_bank(benchmark=benchmark)


def load_manifest() -> dict[str, Any]:
    m = DATA_DIR / "manifest.json"
    if not m.is_file():
        raise FileNotFoundError(f"Missing manifest: {m}")
    with m.open(encoding="utf-8") as f:
        return json.load(f)
