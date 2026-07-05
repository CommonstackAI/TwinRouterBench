"""Concurrently run many SWE-bench Verified instances through
:func:`swerouter.harness.run_instance.run_instance` and aggregate results.

Resume-from-last-completed is driven by the presence of
``<output_dir>/results/<instance_id>.json`` on disk: any instance with a valid
result file is skipped on subsequent invocations unless ``force_rerun=True``.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from swerouter.harness.run_instance import (
    DEFAULT_DATASET_NAME,
    DEFAULT_DATASET_SPLIT,
    DEFAULT_POOL,
    DEFAULT_PRICING,
    DEFAULT_TTL,
    InstanceResult,
    RunInstanceRequest,
    run_instance,
)
from swerouter.router import Router


def _json_default(o: object) -> object:
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


@dataclass
class EvalRequest:
    """Run-wide inputs for :func:`run_eval`."""

    router: Router
    base_url: str
    api_key: str
    output_dir: Path
    dataset_name: str = DEFAULT_DATASET_NAME
    dataset_split: str = DEFAULT_DATASET_SPLIT
    pool_path: Path = DEFAULT_POOL
    pricing_path: Path = DEFAULT_PRICING
    ttl_path: Path = DEFAULT_TTL
    instance_ids: tuple[str, ...] | None = None
    limit: int | None = None
    max_workers: int = 8
    max_steps: int = 40
    # Optional per-instance override of ``max_steps`` (missing keys fall back to
    # the global ``max_steps``). Lets the caller cap each instance at e.g. its
    # GT trajectory length + 1 without sharing one loose global budget.
    max_steps_by_instance: dict[str, int] | None = None
    budget_usd: float = 5.0
    run_id: str = "swerouter_default"
    force_rerun: bool = False
    rm_image: bool = False
    image_namespace: str | None = "swebench"
    windows_compat: bool = False


@dataclass
class EvalSummary:
    """Aggregate written to ``output_dir/eval_summary.json`` when the run ends."""

    router_label: str
    run_id: str
    started_at: float
    finished_at: float
    dataset_name: str
    dataset_split: str
    pool_fingerprint: str
    pricing_schema_version: int
    ttl_policy_name: str
    total_instances: int
    completed: int
    resolved_count: int
    resolved_rate: float
    total_router_cost_usd: float
    per_instance_paths: list[str] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)


def _resolve_instance_list(
    dataset_name: str,
    dataset_split: str,
    selected: Iterable[str] | None,
    limit: int | None,
) -> list[str]:
    from swebench.harness.utils import load_swebench_dataset

    dataset = list(load_swebench_dataset(dataset_name, dataset_split))
    all_ids = [row["instance_id"] for row in dataset]
    if selected is not None:
        wanted = set(selected)
        missing = wanted - set(all_ids)
        if missing:
            raise ValueError(
                f"requested instance_ids not in dataset: {sorted(missing)[:5]}..."
            )
        ordered = [i for i in all_ids if i in wanted]
    else:
        ordered = all_ids
    if limit is not None:
        ordered = ordered[:limit]
    return ordered


def _result_path(output_dir: Path, instance_id: str) -> Path:
    return output_dir / "results" / f"{instance_id}.json"


def _persist_result(output_dir: Path, result: InstanceResult) -> Path:
    path = _result_path(output_dir, result.instance_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(asdict(result), fh, default=_json_default, indent=2)
    return path


def _load_existing_result(output_dir: Path, instance_id: str) -> InstanceResult | None:
    path = _result_path(output_dir, instance_id)
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            blob = json.load(fh)
    except Exception:
        return None
    # Any persisted ``results/<instance_id>.json`` that parses and rehydrates
    # is treated as a finished harness outcome (resolved true/false, agent or
    # eval errors, or infra/API strings in the error fields). Resume must not
    # silently re-run those cases; operators delete the JSON or pass
    # ``force_rerun`` when they intentionally want a fresh attempt.
    # Rehydrate minimal InstanceResult shape. Fields added in future versions
    # get defaulted via ``dict.get``.
    try:
        return InstanceResult(
            instance_id=blob["instance_id"],
            resolved=bool(blob.get("resolved", False)),
            patch=blob.get("patch"),
            patch_applied=bool(blob.get("patch_applied", False)),
            trace_path=Path(blob.get("trace_path", "")),
            step_count=int(blob.get("step_count", 0)),
            total_router_cost_usd=float(blob.get("total_router_cost_usd", 0.0)),
            finished_by=str(blob.get("finished_by", "unknown")),
            model_distribution=dict(blob.get("model_distribution", {})),
            agent_error=blob.get("agent_error"),
            eval_error=blob.get("eval_error"),
            eval_report_path=Path(blob["eval_report_path"])
            if blob.get("eval_report_path")
            else None,
            pool_fingerprint=str(blob.get("pool_fingerprint", "")),
            pricing_schema_version=int(blob.get("pricing_schema_version", 0)),
            ttl_policy_name=str(blob.get("ttl_policy_name", "")),
            fail_to_pass_pass_count=blob.get("fail_to_pass_pass_count"),
            fail_to_pass_fail_count=blob.get("fail_to_pass_fail_count"),
            pass_to_pass_pass_count=blob.get("pass_to_pass_pass_count"),
            pass_to_pass_fail_count=blob.get("pass_to_pass_fail_count"),
            extra=dict(blob.get("extra") or {}),
        )
    except Exception:
        return None


def run_eval(request: EvalRequest, *, router_label: str) -> EvalSummary:
    """Run all instances in ``instance_ids`` concurrently and write a summary."""

    output_dir = Path(request.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    instance_ids = _resolve_instance_list(
        request.dataset_name,
        request.dataset_split,
        request.instance_ids,
        request.limit,
    )

    started_at = time.time()
    results: list[InstanceResult] = []
    errors: list[dict] = []

    pending: list[str] = []
    for iid in instance_ids:
        cached = None if request.force_rerun else _load_existing_result(output_dir, iid)
        if cached is not None:
            results.append(cached)
        else:
            pending.append(iid)

    def _execute_one(instance_id: str) -> InstanceResult:
        per_instance_cap = request.max_steps
        override_map = request.max_steps_by_instance or {}
        if instance_id in override_map:
            per_instance_cap = int(override_map[instance_id])
            if per_instance_cap <= 0:
                raise ValueError(
                    f"max_steps_by_instance[{instance_id!r}] must be > 0, got "
                    f"{override_map[instance_id]!r}"
                )
        req = RunInstanceRequest(
            instance_id=instance_id,
            router=request.router,
            base_url=request.base_url,
            api_key=request.api_key,
            output_dir=output_dir,
            dataset_name=request.dataset_name,
            dataset_split=request.dataset_split,
            pool_path=request.pool_path,
            pricing_path=request.pricing_path,
            ttl_path=request.ttl_path,
            max_steps=per_instance_cap,
            budget_usd=request.budget_usd,
            run_id=request.run_id,
            rm_image=request.rm_image,
            image_namespace=request.image_namespace,
            windows_compat=request.windows_compat,
        )
        return run_instance(req)

    if pending:
        n_pending = len(pending)
        print(
            f"[swerouterbench] run_eval: executing {n_pending} pending instance(s) "
            f"(max_workers={request.max_workers})",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=request.max_workers) as pool_ex:
            futures = {pool_ex.submit(_execute_one, iid): iid for iid in pending}
            done_n = 0
            for fut in as_completed(futures):
                iid = futures[fut]
                try:
                    result = fut.result()
                except Exception as ex:  # noqa: BLE001 — record per-instance failure and continue
                    done_n += 1
                    errors.append({"instance_id": iid, "error": f"{type(ex).__name__}: {ex}"})
                    print(
                        f"[swerouterbench] instance {done_n}/{n_pending} {iid}: "
                        f"FAILED {type(ex).__name__}",
                        flush=True,
                    )
                    continue
                _persist_result(output_dir, result)
                results.append(result)
                done_n += 1
                print(
                    f"[swerouterbench] instance {done_n}/{n_pending} {iid}: "
                    f"ok resolved={result.resolved} cost_usd={result.total_router_cost_usd:.4f}",
                    flush=True,
                )

    finished_at = time.time()
    resolved_count = sum(1 for r in results if r.resolved)
    total_router_cost = sum(r.total_router_cost_usd for r in results)

    pool_fp = next((r.pool_fingerprint for r in results if r.pool_fingerprint), "")
    pricing_v = next((r.pricing_schema_version for r in results if r.pricing_schema_version), 0)
    ttl_name = next((r.ttl_policy_name for r in results if r.ttl_policy_name), "")

    summary = EvalSummary(
        router_label=router_label,
        run_id=request.run_id,
        started_at=started_at,
        finished_at=finished_at,
        dataset_name=request.dataset_name,
        dataset_split=request.dataset_split,
        pool_fingerprint=pool_fp,
        pricing_schema_version=pricing_v,
        ttl_policy_name=ttl_name,
        total_instances=len(instance_ids),
        completed=len(results),
        resolved_count=resolved_count,
        resolved_rate=(resolved_count / len(results)) if results else 0.0,
        total_router_cost_usd=total_router_cost,
        per_instance_paths=[
            str(_result_path(output_dir, r.instance_id)) for r in results
        ],
        errors=errors,
    )

    summary_path = output_dir / "eval_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(asdict(summary), fh, default=_json_default, indent=2)

    return summary


__all__ = ["EvalRequest", "EvalSummary", "run_eval"]
