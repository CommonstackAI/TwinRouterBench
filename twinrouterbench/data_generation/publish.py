"""Manual-review and publication gates for generated supervision."""

from __future__ import annotations

import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from .source import SourceSpec
from .types import TIER_ORDER, TIER_TO_ID, lower_tier


REVIEW_VERDICTS = {"tight", "uncertain", "further_downgradeable"}


def _json_text(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=indent,
        separators=None if indent is not None else (",", ":"),
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row at {path}:{line_number} is not an object")
        rows.append(row)
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_text(value, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_json_text(row) + "\n")


def export_review_queue(
    run_dir: str | Path, output_path: str | Path | None = None
) -> Path:
    run_dir = Path(run_dir)
    queue = _load_jsonl(run_dir / "review_queue.jsonl")
    output = Path(output_path) if output_path else run_dir / "manual_reviews.jsonl"
    if output.exists():
        raise FileExistsError(f"review export already exists: {output}")
    _write_jsonl(output, queue)
    return output


def apply_reviews(run_dir: str | Path, reviews_path: str | Path) -> dict[str, Any]:
    """Apply human verdicts and emit final labels only when all samples resolve."""

    run_dir = Path(run_dir)
    pre_rows = _load_jsonl(run_dir / "labels.pre_review.jsonl")
    queue = _load_jsonl(run_dir / "review_queue.jsonl")
    reviews = _load_jsonl(Path(reviews_path))

    queue_by_id = {str(row["id"]): row for row in queue}
    if len(queue_by_id) != len(queue):
        raise ValueError("review queue contains duplicate ids")
    reviews_by_id: dict[str, dict[str, Any]] = {}
    for review in reviews:
        row_id = str(review.get("id", ""))
        if row_id not in queue_by_id:
            raise ValueError(f"review id is not in the queue: {row_id!r}")
        if row_id in reviews_by_id:
            raise ValueError(f"duplicate review id: {row_id}")
        verdict = str(review.get("verdict", "")).strip()
        if verdict and verdict not in REVIEW_VERDICTS:
            raise ValueError(
                f"invalid verdict {verdict!r} for {row_id}; expected {REVIEW_VERDICTS}"
            )
        reviews_by_id[row_id] = review

    final_rows: list[dict[str, Any]] = []
    applied_rows: list[dict[str, Any]] = []
    unresolved: list[str] = []
    for source in pre_rows:
        row = copy.deepcopy(source)
        row_id = str(row["id"])
        if row_id not in queue_by_id:
            row["pipeline_stage"] = "ground_truth_ready"
            final_rows.append(row)
            continue

        review = reviews_by_id.get(row_id)
        verdict = str((review or {}).get("verdict", "")).strip()
        if not verdict or verdict == "uncertain":
            unresolved.append(row_id)
            applied_rows.append(
                {
                    "id": row_id,
                    "verdict": verdict or "pending",
                    "reviewer": str((review or {}).get("reviewer", "")),
                    "notes": str((review or {}).get("notes", "")),
                    "status": "unresolved",
                }
            )
            continue

        original_tier = str(row["target_tier"])
        if verdict == "further_downgradeable":
            adjusted = lower_tier(original_tier)
            row["review_original_tier"] = original_tier
            row["target_tier"] = adjusted
            row["target_tier_id"] = TIER_TO_ID[adjusted]
            row["selected_model"] = None
            row["review_adjusted"] = True
        else:
            row["review_adjusted"] = False
        row["review_status"] = verdict
        row["reviewer"] = str(review.get("reviewer", ""))
        row["review_notes"] = str(review.get("notes", ""))
        row["pipeline_stage"] = "ground_truth_ready"
        final_rows.append(row)
        applied_rows.append(
            {
                "id": row_id,
                "verdict": verdict,
                "reviewer": row["reviewer"],
                "notes": row["review_notes"],
                "original_tier": original_tier,
                "final_tier": row["target_tier"],
                "status": "applied",
            }
        )

    _write_jsonl(run_dir / "reviews.applied.jsonl", applied_rows)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("artifacts", {})["applied_reviews"] = "reviews.applied.jsonl"
    manifest["review"] = {
        "queued": len(queue),
        "submitted": len(reviews_by_id),
        "unresolved": unresolved,
        "applied": len([row for row in applied_rows if row["status"] == "applied"]),
    }
    if unresolved:
        manifest["status"] = "review_incomplete"
        manifest["artifacts"]["final_labels"] = None
        stale_final = run_dir / "labels.final.jsonl"
        if stale_final.exists():
            stale_final.unlink()
    else:
        _write_jsonl(run_dir / "labels.final.jsonl", final_rows)
        manifest["status"] = "ready"
        manifest["artifacts"]["final_labels"] = "labels.final.jsonl"
    _write_json(manifest_path, manifest)
    return manifest


def _validate_internal_rows(rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("no final labels to publish")
    ids: set[str] = set()
    by_instance: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows, 1):
        required = {
            "id",
            "benchmark",
            "scenario",
            "instance_id",
            "step_index",
            "total_steps",
            "messages",
            "target_tier",
            "target_tier_id",
            "source",
        }
        missing = required.difference(row)
        if missing:
            raise ValueError(f"final row {index} missing fields: {sorted(missing)}")
        row_id = str(row["id"])
        if row_id in ids:
            raise ValueError(f"duplicate label id: {row_id}")
        ids.add(row_id)
        tier = str(row["target_tier"])
        if tier not in TIER_TO_ID or int(row["target_tier_id"]) != TIER_TO_ID[tier]:
            raise ValueError(f"tier/id mismatch in {row_id}")
        if row.get("pipeline_stage") != "ground_truth_ready":
            raise ValueError(f"row is not finalized: {row_id}")
        if row.get("review_status") in {"pending", "uncertain"}:
            raise ValueError(f"row has unresolved review: {row_id}")
        if not isinstance(row["messages"], list) or not row["messages"]:
            raise ValueError(f"row has no router-visible messages: {row_id}")
        source = row["source"]
        if not isinstance(source, dict) or not all(
            str(source.get(field, "")).strip() for field in ("uri", "license", "version")
        ):
            raise ValueError(f"row has incomplete source metadata: {row_id}")
        if not SourceSpec.parse(str(source["uri"])).is_pinned:
            raise ValueError(f"row has an unpinned remote source: {row_id}")
        key = (str(row["benchmark"]), str(row["instance_id"]))
        by_instance[key].append(row)

    for key, instance_rows in by_instance.items():
        totals = {int(row["total_steps"]) for row in instance_rows}
        if len(totals) != 1:
            raise ValueError(f"inconsistent total_steps for {key}: {totals}")
        expected = list(range(1, next(iter(totals)) + 1))
        actual = sorted(int(row["step_index"]) for row in instance_rows)
        if actual != expected:
            raise ValueError(f"non-contiguous steps for {key}: {actual}, expected {expected}")


def _public_row(row: dict[str, Any]) -> dict[str, Any]:
    public: dict[str, Any] = {
        "id": row["id"],
        "benchmark": row["benchmark"],
        "scenario": row["scenario"],
        "instance_id": row["instance_id"],
        "step_index": row["step_index"],
        "total_steps": row["total_steps"],
        "messages": copy.deepcopy(row["messages"]),
        "target_tier": row["target_tier"],
        "target_tier_id": row["target_tier_id"],
        "benchmark_display": row.get("benchmark_display", row["benchmark"]),
        "benchmark_subset": row.get("benchmark_subset", ""),
        "benchmark_version": row.get("benchmark_version", "unknown"),
        "pipeline_stage": "ground_truth_ready",
        "collector": row.get("collector", "twinrouterbench-reference-pipeline"),
        "collected_at": row.get("collected_at", "unknown"),
        "source": copy.deepcopy(row["source"]),
        "notes": "Generated through the versioned downgrade-and-cascade construction pipeline.",
    }
    if row.get("functions"):
        public["functions"] = copy.deepcopy(row["functions"])
    return public


def publish_runs(
    run_dirs: Sequence[str | Path], output_dir: str | Path
) -> dict[str, Any]:
    """Merge finalized runs into a validated, isolated release candidate."""

    if not run_dirs:
        raise ValueError("publish requires at least one run directory")
    output = Path(output_dir)
    repo_root = Path(__file__).resolve().parents[2]
    official = repo_root / "data" / "static"
    if output.resolve() == official.resolve():
        raise ValueError(
            "publish refuses to overwrite data/static; use a separate candidate directory"
        )
    if (output / "question_bank.jsonl").exists() or (output / "manifest.json").exists():
        raise FileExistsError(f"publish output already contains release artifacts: {output}")

    rows: list[dict[str, Any]] = []
    run_manifests: list[dict[str, Any]] = []
    for raw_dir in run_dirs:
        run_dir = Path(raw_dir)
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "ready":
            raise ValueError(
                f"run {run_dir} is not ready (status={manifest.get('status')!r})"
            )
        rows.extend(_load_jsonl(run_dir / "labels.final.jsonl"))
        run_manifests.append(manifest)

    _validate_internal_rows(rows)
    rows.sort(
        key=lambda row: (
            str(row["benchmark"]),
            str(row["instance_id"]),
            int(row["step_index"]),
        )
    )
    public_rows = [_public_row(row) for row in rows]
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "question_bank.jsonl", public_rows)

    benchmark_counts = Counter(str(row["benchmark"]) for row in public_rows)
    source_registry: dict[str, list[dict[str, Any]]] = defaultdict(list)
    prompt_versions: dict[str, str] = {}
    model_pool_versions: set[str] = set()
    for manifest in run_manifests:
        prompt_versions.update(manifest.get("prompt_versions", {}))
        model_pool_versions.add(str(manifest.get("model_pool_version", "unknown")))
        for benchmark, sources in manifest.get("source_registry", {}).items():
            for source in sources:
                if source not in source_registry[benchmark]:
                    source_registry[benchmark].append(source)

    manifest = {
        "schema": "tier_only_question_bank",
        "question_bank": "question_bank.jsonl",
        "total_line_count": len(public_rows),
        "target_fields": ["target_tier", "target_tier_id"],
        "no_model_ids_in_records": True,
        "sources": {
            benchmark: {
                "line_count": benchmark_counts[benchmark],
                "registry": source_registry.get(benchmark, []),
            }
            for benchmark in sorted(benchmark_counts)
        },
        "prompt_versions": prompt_versions,
        "model_pool_versions": sorted(model_pool_versions),
        "input_run_ids": [str(item.get("run_id", "unknown")) for item in run_manifests],
    }
    _write_json(output / "manifest.json", manifest)
    validate_public_dataset(output)
    return manifest


def validate_public_dataset(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.is_dir():
        bank_path = path / "question_bank.jsonl"
        manifest_path = path / "manifest.json"
    else:
        bank_path = path
        manifest_path = path.with_name("manifest.json")
    rows = _load_jsonl(bank_path)
    ids: set[str] = set()
    by_instance: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    forbidden_fields = {
        "selected_model",
        "baseline_model",
        "optimal_model",
        "reviewer",
        "review_notes",
    }
    for index, row in enumerate(rows, 1):
        missing = {
            "id",
            "benchmark",
            "scenario",
            "instance_id",
            "step_index",
            "total_steps",
            "messages",
            "target_tier",
            "target_tier_id",
        }.difference(row)
        if missing:
            raise ValueError(f"public row {index} missing fields: {sorted(missing)}")
        leaked = forbidden_fields.intersection(row)
        if leaked:
            raise ValueError(f"public row {row['id']} leaks internal fields: {sorted(leaked)}")
        row_id = str(row["id"])
        if row_id in ids:
            raise ValueError(f"duplicate public id: {row_id}")
        ids.add(row_id)
        tier = str(row["target_tier"])
        if tier not in TIER_ORDER or int(row["target_tier_id"]) != TIER_TO_ID[tier]:
            raise ValueError(f"public tier mismatch: {row_id}")
        source = row.get("source")
        if not isinstance(source, dict) or not all(
            str(source.get(field, "")).strip() for field in ("uri", "license", "version")
        ):
            raise ValueError(f"public source metadata is incomplete: {row_id}")
        if not SourceSpec.parse(str(source["uri"])).is_pinned:
            raise ValueError(f"public source is not pinned: {row_id}")
        by_instance[(str(row["benchmark"]), str(row["instance_id"]))].append(row)
    for key, instance_rows in by_instance.items():
        total = {int(row["total_steps"]) for row in instance_rows}
        if len(total) != 1:
            raise ValueError(f"public total_steps mismatch for {key}")
        expected = list(range(1, next(iter(total)) + 1))
        actual = sorted(int(row["step_index"]) for row in instance_rows)
        if actual != expected:
            raise ValueError(f"public step sequence mismatch for {key}: {actual}")

    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("total_line_count", -1)) != len(rows):
            raise ValueError("manifest total_line_count does not match question bank")
    return {
        "rows": len(rows),
        "instances": len(by_instance),
        "benchmarks": dict(sorted(Counter(row["benchmark"] for row in rows).items())),
    }
