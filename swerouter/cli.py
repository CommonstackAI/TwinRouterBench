"""Twin Router Bench — dynamic track (editor scaffold) command-line entrypoint.

Subcommands:

* ``run``         -- run a router on SWE-bench Verified and write a run summary.
* ``score``       -- score an existing run directory (``total_actual_bill_usd``).
* ``audit-infra`` -- scan ``results/*.json`` for fair-metrics infra / harness exclusions.
* ``audit-trace-cost`` -- sum ``step_cost_usd`` vs ``raw_usage.cost`` from ``*.trace.jsonl``.
* ``render``      -- render a markdown leaderboard from one or more score files.

Example::

    swerouterbench run \\
        --router-import swerouter.routers.always_model:AlwaysModelRouter \\
        --router-arg model_id=anthropic/claude-opus-4.6 \\
        --router-arg label=always_opus \\
        --base-url https://openrouter.ai/api/v1 \\
        --api-key $OPENROUTER_API_KEY \\
        --output-dir runs/always_opus \\
        --router-label always_opus \\
        --limit 5
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

from swerouter.router import Router


def _swerouter_repo_root() -> Path:
    """``TwinRouterBench/`` install root (parent of the ``swerouter`` package)."""

    return Path(__file__).resolve().parent.parent


def _load_swerouter_dotenv() -> None:
    """Populate ``os.environ`` from ``TwinRouterBench/.env`` when keys are unset.

    Does not add a ``python-dotenv`` dependency: minimal ``KEY=value`` parsing
    only. Existing non-empty environment variables win.
    """

    path = _swerouter_repo_root() / ".env"
    if not path.is_file():
        return
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        if not key:
            continue
        cur = os.environ.get(key)
        if cur is None or cur == "":
            os.environ[key] = val


def _apply_gateway_aliases() -> None:
    """Map CommonStack-style env names to ``OPENROUTER_*`` / ``SWEROUTER_*``.

    When ``COMMONSTACK_API_BASE`` / ``COMMONSTACK_API_KEY`` are set, they take
    precedence so a single gateway block in ``TwinRouterBench/.env`` is enough.
    """

    base = (os.environ.get("COMMONSTACK_API_BASE") or "").strip()
    if base:
        os.environ["OPENROUTER_BASE_URL"] = base
        os.environ["SWEROUTER_BASE_URL"] = base
    ck = (os.environ.get("COMMONSTACK_API_KEY") or "").strip()
    if ck:
        os.environ["OPENROUTER_API_KEY_EXP"] = ck
        os.environ["OPENROUTER_API_KEY"] = ck
        os.environ["SWEROUTER_API_KEY"] = ck
    if not (os.environ.get("OPENROUTER_API_KEY") or "").strip():
        exp = (os.environ.get("OPENROUTER_API_KEY_EXP") or "").strip()
        if exp:
            os.environ["OPENROUTER_API_KEY"] = exp


def _bootstrap_swerouter_env() -> None:
    _load_swerouter_dotenv()
    _apply_gateway_aliases()


def _parse_router_args(pairs: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in pairs or []:
        if "=" not in p:
            raise ValueError(f"--router-arg expects key=value, got {p!r}")
        k, v = p.split("=", 1)
        if not k:
            raise ValueError(f"--router-arg key empty in {p!r}")
        out[k] = v
    return out


def _instantiate_router(router_import: str, router_args: dict[str, str]) -> Router:
    """Import ``module:attr`` (or ``module:Attr.sub``) and call it with
    ``router_args`` to produce a Router.

    The dotted sub-attribute form lets callers target a classmethod factory,
    e.g. ``swerouter.routers.gold_tier:GoldTierRouter.from_cli_args``, which is
    how routers with non-string construction args (paths, tuples) stay
    CLI-addressable without inventing a parallel factory module.
    """

    if ":" not in router_import:
        raise ValueError(
            f"--router-import expects 'module:ClassOrFactory', got {router_import!r}"
        )
    module_name, attr_path = router_import.split(":", 1)
    if not attr_path:
        raise ValueError(
            f"--router-import attribute path is empty in {router_import!r}"
        )
    module = importlib.import_module(module_name)
    obj: object = module
    for part in attr_path.split("."):
        if not part:
            raise ValueError(
                f"--router-import attribute path has empty segment in {router_import!r}"
            )
        if not hasattr(obj, part):
            raise ValueError(
                f"{router_import!r}: object {type(obj).__name__} has no attribute {part!r}"
            )
        obj = getattr(obj, part)
    if not callable(obj):
        raise TypeError(
            f"--router-import {router_import!r} resolved to non-callable {type(obj).__name__}"
        )
    instance = obj(**router_args)
    if not hasattr(instance, "select") or not callable(instance.select):
        raise TypeError(
            f"{router_import!r} produced {type(instance).__name__}, which has no callable .select"
        )
    return instance


def _parse_max_steps_by_instance(
    json_literal: str | None, json_file: str | None
) -> dict[str, int] | None:
    """Load the optional per-instance step cap override.

    Accepts either inline JSON (``--max-steps-json '{"id": 9}'``) or a path
    to a JSON file (``--max-steps-json-file caps.json``); specifying both is
    an error.
    """

    if json_literal and json_file:
        raise ValueError("pass either --max-steps-json or --max-steps-json-file, not both")
    raw: str | None = None
    if json_literal:
        raw = json_literal
    elif json_file:
        raw = Path(json_file).read_text(encoding="utf-8")
    if raw is None:
        return None
    doc = json.loads(raw)
    if not isinstance(doc, dict):
        raise ValueError("--max-steps-json must decode to an object mapping instance_id -> int")
    out: dict[str, int] = {}
    for k, v in doc.items():
        if not isinstance(k, str) or not k:
            raise ValueError(f"--max-steps-json key must be non-empty string, got {k!r}")
        if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
            raise ValueError(
                f"--max-steps-json[{k!r}] must be positive int, got {v!r}"
            )
        out[k] = v
    return out


def _cmd_run(args: argparse.Namespace) -> int:
    from swerouter.harness.run_eval import EvalRequest, run_eval

    router = _instantiate_router(args.router_import, _parse_router_args(args.router_arg))

    instance_ids: tuple[str, ...] | None = None
    if args.instances:
        instance_ids = tuple(i for i in args.instances if i)

    max_steps_by_instance = _parse_max_steps_by_instance(
        args.max_steps_json, args.max_steps_json_file
    )

    req = EvalRequest(
        router=router,
        base_url=args.base_url,
        api_key=args.api_key,
        output_dir=Path(args.output_dir),
        instance_ids=instance_ids,
        limit=args.limit,
        max_workers=args.workers,
        max_steps=args.max_steps,
        max_steps_by_instance=max_steps_by_instance,
        budget_usd=args.budget_usd,
        run_id=args.run_id,
        force_rerun=args.force_rerun,
        rm_image=args.rm_image,
    )
    summary = run_eval(req, router_label=args.router_label)
    print(
        json.dumps(
            {
                "router_label": summary.router_label,
                "resolved_count": summary.resolved_count,
                "resolved_rate": summary.resolved_rate,
                "completed": summary.completed,
                "total_router_cost_usd": summary.total_router_cost_usd,
                "pool_fingerprint": summary.pool_fingerprint,
                "pricing_schema_version": summary.pricing_schema_version,
            },
            indent=2,
        )
    )
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    from swerouter.leaderboard.score import score_run_dir

    score = score_run_dir(
        run_dir=Path(args.run_dir),
        router_label=args.router_label,
        pricing_path=Path(args.pricing) if args.pricing else None,
        ttl_path=Path(args.ttl) if args.ttl else None,
        pool_path=Path(args.pool) if args.pool else None,
        exclude_infra_failures=bool(args.exclude_infra_failures),
        reprice_from_raw_usage=bool(args.reprice_from_raw_usage),
    )
    out_path = Path(args.out) if args.out else Path(args.run_dir) / "score.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(score, fh, indent=2)
    summary = {
        "router_label": score["router_label"],
        "total_actual_bill_usd": score["total_actual_bill_usd"],
        "resolved_count": score["resolved_count"],
        "resolved_rate": score["resolved_rate"],
        "instances": score["instance_count"],
        "pricing_fingerprint": score["pricing_fingerprint"],
        "score_path": str(out_path),
    }
    if score.get("exclude_infra_failures"):
        summary["raw_instance_count"] = score["raw_instance_count"]
        summary["infra_excluded_count"] = score["infra_excluded_count"]
    if score.get("reprice_from_raw_usage"):
        summary["reprice_from_raw_usage"] = True
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_audit_infra(args: argparse.Namespace) -> int:
    """Summarise which instances ``--exclude-infra-failures`` would drop."""

    from swerouter.infra_errors import (
        is_excluded_from_fair_metrics,
        is_transport_or_infra_failure,
    )

    run_dir = Path(args.run_dir).resolve()
    results_dir = run_dir / "results"
    if not results_dir.is_dir():
        raise SystemExit(f"no results directory: {results_dir}")

    total = 0
    fair_excluded = 0
    subset_transport = 0
    excluded_ids: list[str] = []

    for p in sorted(results_dir.glob("*.json")):
        total += 1
        with p.open("r", encoding="utf-8") as fh:
            doc = json.load(fh)
        iid = str(doc.get("instance_id") or p.stem)
        ae = doc.get("agent_error")
        ee = doc.get("eval_error")
        ae_s = ae if isinstance(ae, str) else None
        ee_s = ee if isinstance(ee, str) else None
        if not is_excluded_from_fair_metrics(ae_s, ee_s):
            continue
        fair_excluded += 1
        excluded_ids.append(iid)
        if is_transport_or_infra_failure(ae_s) or is_transport_or_infra_failure(ee_s):
            subset_transport += 1

    report = {
        "run_dir": str(run_dir),
        "results_json_count": total,
        "fair_metrics_excluded_count": fair_excluded,
        "fair_metrics_excluded_transport_or_quota_subset": subset_transport,
        "fair_metrics_excluded_env_harness_provider_subset": fair_excluded
        - subset_transport,
        "eligible_for_fair_headline_count": total - fair_excluded,
    }
    if args.list_ids:
        report["excluded_instance_ids"] = sorted(excluded_ids)

    out_path = Path(args.out) if args.out else None
    text = json.dumps(report, indent=2)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"wrote {out_path}")
    else:
        print(text)
    return 0


def _cmd_audit_trace_cost(args: argparse.Namespace) -> int:
    """Sum step_cost_usd vs raw_usage.cost across all ``*.trace.jsonl`` in a run."""

    from swerouter.trace_cost_audit import audit_trace_cost_metrics

    report = audit_trace_cost_metrics(Path(args.run_dir))
    text = json.dumps(report, indent=2)
    out_path = Path(args.out) if args.out else None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"wrote {out_path}")
    else:
        print(text)
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    from swerouter.leaderboard.render import render_leaderboard

    score_files = [Path(p) for p in args.score]
    markdown = render_leaderboard(score_files)
    out = Path(args.out) if args.out else None
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown, encoding="utf-8")
        print(f"wrote {out}")
    else:
        sys.stdout.write(markdown)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="swerouterbench")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run a router on SWE-bench Verified.")
    run.add_argument("--router-import", required=True, help="module:ClassOrFactory")
    run.add_argument(
        "--router-arg",
        action="append",
        default=[],
        help="key=value; may be repeated; passed as kwargs to the factory",
    )
    run.add_argument("--router-label", required=True)
    run.add_argument(
        "--base-url",
        default=os.environ.get("SWEROUTER_BASE_URL")
        or os.environ.get("OPENROUTER_BASE_URL"),
    )
    run.add_argument(
        "--api-key",
        default=os.environ.get("SWEROUTER_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY"),
    )
    run.add_argument("--output-dir", required=True)
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--instances", nargs="*", default=None)
    run.add_argument("--workers", type=int, default=8)
    run.add_argument("--max-steps", type=int, default=40)
    run.add_argument(
        "--max-steps-json",
        default=None,
        help='Optional inline JSON mapping instance_id -> max_steps, e.g. \'{"django__django-11133": 9}\'. '
        "Missing keys fall back to --max-steps.",
    )
    run.add_argument(
        "--max-steps-json-file",
        default=None,
        help="Path to a JSON file with the same shape as --max-steps-json.",
    )
    run.add_argument("--budget-usd", type=float, default=5.0)
    run.add_argument("--run-id", default="swerouter_default")
    run.add_argument("--force-rerun", action="store_true")
    run.add_argument("--rm-image", action="store_true")
    run.set_defaults(func=_cmd_run)

    score = sub.add_parser("score", help="Score an existing run directory.")
    score.add_argument("--run-dir", required=True)
    score.add_argument("--router-label", required=True)
    score.add_argument("--pricing", default=None)
    score.add_argument("--ttl", default=None)
    score.add_argument("--pool", default=None)
    score.add_argument("--out", default=None)
    score.add_argument(
        "--exclude-infra-failures",
        action="store_true",
        help="Exclude instances matching swerouter.infra_errors.is_excluded_from_fair_metrics "
        "(transport/quota + env/harness/provider-wrap) from headline metrics.",
    )
    score.add_argument(
        "--reprice-from-raw-usage",
        action="store_true",
        help="Recompute per-step USD and baseline inputs from trace raw_usage "
        "using current normalize_usage + pricing (historical trace correction).",
    )
    score.set_defaults(func=_cmd_score)

    audit = sub.add_parser(
        "audit-infra",
        help="Scan results/*.json for instances excluded by --exclude-infra-failures.",
    )
    audit.add_argument("--run-dir", required=True)
    audit.add_argument(
        "--list-ids",
        action="store_true",
        help="Include sorted excluded_instance_ids in the JSON output.",
    )
    audit.add_argument(
        "--out",
        default=None,
        help="Write JSON report to this path instead of stdout.",
    )
    audit.set_defaults(func=_cmd_audit_infra)

    audit_cost = sub.add_parser(
        "audit-trace-cost",
        help="Sum step_cost_usd and raw_usage.cost from run_dir/*.trace.jsonl (OpenRouter vs table).",
    )
    audit_cost.add_argument("--run-dir", required=True)
    audit_cost.add_argument(
        "--out",
        default=None,
        help="Write JSON report to this path instead of stdout.",
    )
    audit_cost.set_defaults(func=_cmd_audit_trace_cost)

    render = sub.add_parser("render", help="Render a markdown leaderboard.")
    render.add_argument("--score", nargs="+", required=True)
    render.add_argument("--out", default=None)
    render.set_defaults(func=_cmd_render)

    return p


def main(argv: list[str] | None = None) -> int:
    _bootstrap_swerouter_env()
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        missing = [k for k in ("base_url", "api_key") if not getattr(args, k)]
        if missing:
            parser.error(
                f"missing required connection settings: {missing}. Pass --base-url / --api-key "
                f"or set SWEROUTER_BASE_URL / SWEROUTER_API_KEY (or OPENROUTER_* / COMMONSTACK_*; "
                f"see README)."
            )

    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
