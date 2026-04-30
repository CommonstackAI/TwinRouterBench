"""Run one SWE-bench Verified instance end-to-end (editor scaffold).

Flow:

1. Load the dataset row and build a TestSpec via
   :mod:`swerouter.harness.container_runner`.
2. Start the upstream work container.
3. Run the tool-use agent loop (``swerouter.agent.loop.run_agent_loop``)
   inside that container. The router picks a model at every step.
4. Extract the final patch with ``git diff`` (via ``container_runner``).
5. Grade the patch through upstream's official evaluator (via
   ``container_runner``) so our ``resolved`` numbers match the SWE-bench
   Verified leaderboard.
6. Clean up.

All Docker/SWE-bench plumbing is delegated to ``container_runner`` so the
MiniSWERouterBench bridge can reuse the exact same code path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from swerouter.agent.loop import (
    AgentConfig,
    AgentRunResult,
    ModelPoolEntry,
    load_model_pool,
    run_agent_loop,
)
from swerouter.cache import TTLPolicy
from swerouter.harness.container_runner import (
    DEFAULT_DATASET_NAME,
    DEFAULT_DATASET_SPLIT,
    DEFAULT_EVAL_MODEL_NAME,
    DEFAULT_IMAGE_NAMESPACE,
    SwebenchContainerHandle,
    extract_git_diff,
    load_dataset_instance,
    make_test_spec_for_instance,
    run_upstream_eval,
)
from swerouter.llm_client import LLMClient
from swerouter.pricing import PricingTable, load_pricing_table
from swerouter.router import Router

REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCKED_DATA = REPO_ROOT / "data" / "dynamic"
DEFAULT_POOL = _LOCKED_DATA / "model_pool.json"
DEFAULT_PRICING = _LOCKED_DATA / "model_pricing.json"
DEFAULT_TTL = _LOCKED_DATA / "ttl_policy.json"
DEFAULT_TIER_MAP = _LOCKED_DATA / "tier_to_model.json"


def _recover_agent_metrics_from_trace(trace_path: Path) -> dict[str, Any] | None:
    """When ``run_agent_loop`` raised before returning, read its ``finally`` block
    ``loop_summary`` row so ``total_router_cost_usd`` matches ``*.trace.jsonl``.

    Without this, persisted ``results/<id>.json`` used ``0.0`` for router cost
    while :func:`swerouter.leaderboard.score.score_run_dir` summed non-zero
    ``step_cost_usd`` lines, inflating headline ``total_router_cost_usd``.
    """

    if not trace_path.is_file():
        return None
    last_summary: dict[str, Any] | None = None
    with trace_path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("__marker__") == "loop_summary":
                last_summary = row
    return last_summary


@dataclass
class RunInstanceRequest:
    """All inputs needed to run one instance."""

    instance_id: str
    router: Router
    base_url: str
    api_key: str
    output_dir: Path
    dataset_name: str = DEFAULT_DATASET_NAME
    dataset_split: str = DEFAULT_DATASET_SPLIT
    pool_path: Path = DEFAULT_POOL
    pricing_path: Path = DEFAULT_PRICING
    ttl_path: Path = DEFAULT_TTL
    tier_map_path: Path = DEFAULT_TIER_MAP
    max_steps: int = 40
    budget_usd: float = 5.0
    max_response_tokens: int = 4096
    temperature: float = 0.0
    select_timeout_sec: float = 30.0
    run_id: str = "swerouter_default"
    eval_timeout_sec: int = 1800
    force_rebuild: bool = False
    rm_image: bool = False
    # When set, ``make_test_spec(..., namespace=...)`` asks swebench to pull
    # pre-built instance images from Docker Hub under that namespace (the
    # official registry is ``swebench``). Avoids multi-hour local rebuilds.
    image_namespace: str | None = DEFAULT_IMAGE_NAMESPACE


@dataclass
class InstanceResult:
    """Single-instance output consumed by :mod:`swerouter.harness.run_eval`
    and :mod:`swerouter.leaderboard.score`.
    """

    instance_id: str
    resolved: bool
    patch: str | None
    patch_applied: bool
    trace_path: Path
    step_count: int
    total_router_cost_usd: float
    finished_by: str
    model_distribution: dict[str, int]
    agent_error: str | None
    eval_error: str | None
    eval_report_path: Path | None
    pool_fingerprint: str
    pricing_schema_version: int
    ttl_policy_name: str
    fail_to_pass_pass_count: int | None = None
    fail_to_pass_fail_count: int | None = None
    pass_to_pass_pass_count: int | None = None
    pass_to_pass_fail_count: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _pool_fingerprint(pool: list[ModelPoolEntry]) -> str:
    sorted_ids = sorted(p.model_id for p in pool)
    return "|".join(sorted_ids)


def _load_tier_reverse_map(path: Path) -> dict[str, str]:
    """Invert ``data/tier_to_model.json`` to ``{model_id: tier_name}``.

    Used only for human-facing case summaries; missing IDs fall back to
    ``None`` rather than raising so a pool that is intentionally richer than
    the canonical four tiers still writes a summary.
    """
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        doc = json.load(fh)
    fwd = doc.get("map") if isinstance(doc, dict) else None
    if not isinstance(fwd, dict):
        return {}
    rev: dict[str, str] = {}
    for tier_name, model_id in fwd.items():
        if isinstance(tier_name, str) and isinstance(model_id, str):
            rev[model_id] = tier_name
    return rev


def _parse_trace_for_summary(trace_path: Path) -> list[dict[str, Any]]:
    """Read ``<iid>.trace.jsonl`` and pair each step row with its
    following ``tool_results`` marker (when present).

    Returns a list of per-step dicts in natural ``step_index`` order.
    """
    if not trace_path.is_file():
        return []
    step_rows: dict[int, dict[str, Any]] = {}
    tool_results: dict[int, list[dict[str, Any]]] = {}
    with trace_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            marker = row.get("__marker__")
            if marker == "tool_results":
                idx = row.get("step_index")
                if isinstance(idx, int):
                    tool_results[idx] = list(row.get("tool_results") or [])
            elif marker is None and "step_index" in row and "model_id" in row:
                idx = row["step_index"]
                if isinstance(idx, int):
                    step_rows[idx] = row
    out: list[dict[str, Any]] = []
    for idx in sorted(step_rows.keys()):
        row = step_rows[idx]
        row = dict(row)
        row["tool_results"] = tool_results.get(idx, [])
        out.append(row)
    return out


def _write_case_summary(
    *,
    output_dir: Path,
    result: "InstanceResult",
    tier_by_model: dict[str, str],
) -> Path:
    """Persist a compact per-case summary to ``case_summaries/<iid>.summary.json``.

    Pulls per-step data from the trace file so the summary stays in sync with
    what the scorer consumes, and enriches each step with its pool tier.
    """
    steps_raw = _parse_trace_for_summary(result.trace_path)
    per_step: list[dict[str, Any]] = []
    tier_dist: dict[str, int] = {}
    for row in steps_raw:
        model_id = row.get("model_id")
        tier = tier_by_model.get(model_id) if isinstance(model_id, str) else None
        if tier is not None:
            tier_dist[tier] = tier_dist.get(tier, 0) + 1
        per_step.append(
            {
                "step_index": row.get("step_index"),
                "model_id": model_id,
                "tier": tier,
                "provider": row.get("provider"),
                "rationale": row.get("rationale"),
                "latency_ms": row.get("latency_ms"),
                "started_at": row.get("started_at"),
                "finished_at": row.get("finished_at"),
                "usage": row.get("usage"),
                "step_cost_usd": row.get("step_cost_usd"),
                "cumulative_cost_usd": row.get("cumulative_cost_usd"),
                "cache_lookup": row.get("cache_lookup"),
                "tool_call_count": row.get("tool_call_count"),
                "assistant_content_len": row.get("assistant_content_len"),
                "tool_calls_preview": row.get("tool_calls_preview"),
                "tool_results": [
                    {
                        "tool_name": tr.get("tool_name"),
                        "ok": tr.get("ok"),
                        "content_length": tr.get("content_length"),
                    }
                    for tr in (row.get("tool_results") or [])
                ],
            }
        )
    summary = {
        "instance_id": result.instance_id,
        "resolved": result.resolved,
        "patch_applied": result.patch_applied,
        "step_count": result.step_count,
        "finished_by": result.finished_by,
        "total_router_cost_usd": result.total_router_cost_usd,
        "agent_error": result.agent_error,
        "eval_error": result.eval_error,
        "eval": {
            "fail_to_pass_pass_count": result.fail_to_pass_pass_count,
            "fail_to_pass_fail_count": result.fail_to_pass_fail_count,
            "pass_to_pass_pass_count": result.pass_to_pass_pass_count,
            "pass_to_pass_fail_count": result.pass_to_pass_fail_count,
            "eval_report_path": (
                str(result.eval_report_path) if result.eval_report_path else None
            ),
        },
        "pool_fingerprint": result.pool_fingerprint,
        "pricing_schema_version": result.pricing_schema_version,
        "ttl_policy_name": result.ttl_policy_name,
        "model_distribution": dict(result.model_distribution),
        "tier_distribution": tier_dist,
        "trace_path": str(result.trace_path),
        "io_log_path": str(output_dir / "llm_io" / f"{result.instance_id}.io.jsonl"),
        "per_step": per_step,
    }
    path = output_dir / "case_summaries" / f"{result.instance_id}.summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    return path


def run_instance(request: RunInstanceRequest) -> InstanceResult:
    """Drive one SWE-bench Verified instance end-to-end."""

    # Resolve to an absolute path so downstream consumers (score.py, leaderboard)
    # never have to guess the base directory of a persisted trace_path.
    output_dir = Path(request.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / f"{request.instance_id}.trace.jsonl"
    io_path = output_dir / "llm_io" / f"{request.instance_id}.io.jsonl"

    pool = load_model_pool(request.pool_path)
    pricing = load_pricing_table(request.pricing_path)
    ttl = TTLPolicy.load(request.ttl_path)
    tier_by_model = _load_tier_reverse_map(Path(request.tier_map_path))

    # Sanity: every pool model must have a pricing entry before we even touch the cluster.
    for entry in pool:
        if entry.model_id not in pricing:
            raise ValueError(
                f"pool model {entry.model_id!r} missing from pricing table schema v{pricing.schema_version}"
            )

    instance = load_dataset_instance(
        request.instance_id,
        dataset_name=request.dataset_name,
        dataset_split=request.dataset_split,
    )
    test_spec = make_test_spec_for_instance(
        instance, image_namespace=request.image_namespace
    )

    llm = LLMClient(
        base_url=request.base_url,
        api_key=request.api_key,
    )

    log_path = output_dir / "agent_logs" / request.instance_id / "agent.log"
    handle = SwebenchContainerHandle(
        test_spec=test_spec,
        run_id=request.run_id,
        log_path=log_path,
        force_rebuild=request.force_rebuild,
    )

    agent_error: str | None = None
    agent_summary: AgentRunResult | None = None
    patch_text = ""
    try:
        handle.start()
        try:
            agent_summary = run_agent_loop(
                instance_id=request.instance_id,
                repo=str(instance.get("repo", "")),
                base_commit=str(instance.get("base_commit", "")),
                problem_statement=str(instance.get("problem_statement", "")),
                container=handle.container,
                router=request.router,
                pool=pool,
                pricing=pricing,
                ttl=ttl,
                llm=llm,
                config=AgentConfig(
                    max_steps=request.max_steps,
                    budget_usd=request.budget_usd,
                    max_response_tokens=request.max_response_tokens,
                    temperature=request.temperature,
                    select_timeout_sec=request.select_timeout_sec,
                ),
                trace_path=trace_path,
                io_path=io_path,
            )
            patch_text = extract_git_diff(handle.container)
        except Exception as ex:
            agent_error = f"{type(ex).__name__}: {ex}"
            patch_text = ""
    finally:
        handle.stop()

    # Feed the patch into upstream's official evaluation path.
    # container_runner builds a fresh docker client internally because we've
    # already torn down the work container's client in ``handle.stop()``.
    eval_report = run_upstream_eval(
        test_spec=test_spec,
        instance_id=request.instance_id,
        patch_text=patch_text,
        run_id=request.run_id,
        timeout_sec=request.eval_timeout_sec,
        rm_image=request.rm_image,
        model_name=DEFAULT_EVAL_MODEL_NAME,
    )

    if agent_summary is not None:
        step_count = agent_summary.step_count
        total_router_cost_usd = agent_summary.total_router_cost_usd
        finished_by = agent_summary.finished_by
        model_distribution = dict(agent_summary.model_distribution)
    else:
        step_count = 0
        total_router_cost_usd = 0.0
        finished_by = "error_before_loop"
        model_distribution = {}
        recovered = _recover_agent_metrics_from_trace(trace_path)
        if recovered is not None:
            step_count = int(recovered.get("step_count", 0))
            total_router_cost_usd = float(recovered.get("total_router_cost_usd", 0.0))
            finished_by = str(recovered.get("finished_by") or finished_by)
            md = recovered.get("model_distribution")
            if isinstance(md, dict):
                model_distribution = {str(k): int(v) for k, v in md.items()}

    result = InstanceResult(
        instance_id=request.instance_id,
        resolved=eval_report.resolved,
        patch=patch_text or None,
        patch_applied=eval_report.patch_applied,
        trace_path=trace_path,
        step_count=step_count,
        total_router_cost_usd=total_router_cost_usd,
        finished_by=finished_by,
        model_distribution=model_distribution,
        agent_error=agent_error,
        eval_error=eval_report.error,
        eval_report_path=eval_report.report_path,
        pool_fingerprint=_pool_fingerprint(pool),
        pricing_schema_version=pricing.schema_version,
        ttl_policy_name=ttl.policy_name,
        fail_to_pass_pass_count=eval_report.test_counts.get("FAIL_TO_PASS.success"),
        fail_to_pass_fail_count=eval_report.test_counts.get("FAIL_TO_PASS.failure"),
        pass_to_pass_pass_count=eval_report.test_counts.get("PASS_TO_PASS.success"),
        pass_to_pass_fail_count=eval_report.test_counts.get("PASS_TO_PASS.failure"),
        extra={"instance_repo": instance.get("repo", "")},
    )

    # Emit the per-case summary next to the raw trace. This is human-facing
    # (not consumed by the scorer) so we write it best-effort and never let
    # a summary failure clobber a real evaluation outcome.
    try:
        _write_case_summary(
            output_dir=output_dir,
            result=result,
            tier_by_model=tier_by_model,
        )
    except Exception as ex:  # noqa: BLE001 — best-effort human log
        result.extra["case_summary_error"] = f"{type(ex).__name__}: {ex}"

    return result


__all__ = [
    "RunInstanceRequest",
    "InstanceResult",
    "run_instance",
    "DEFAULT_DATASET_NAME",
    "DEFAULT_DATASET_SPLIT",
    "DEFAULT_POOL",
    "DEFAULT_PRICING",
    "DEFAULT_TTL",
    "DEFAULT_TIER_MAP",
]
