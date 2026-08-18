"""Command-line interface for the static data-construction pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .api import PipelineRequest, run_pipeline
from .backends import MockBackend, ReplayBackend, load_live_backend
from .benchmarking import BenchmarkRegistry, builtin_registry
from .pipeline import GenerationConfig, GenerationPipeline, load_model_pool
from .publish import (
    apply_reviews,
    export_review_queue,
    publish_runs,
    validate_public_dataset,
)
from .source import SourceSpec


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def _parse_sources(
    values: list[str], registry: BenchmarkRegistry
) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit("--source must use benchmark=URI syntax")
        benchmark, uri = value.split("=", 1)
        registry.get(benchmark)
        SourceSpec.parse(uri)
        result[benchmark] = uri
    return result


def _load_generation_config(
    args: argparse.Namespace, registry: BenchmarkRegistry
) -> GenerationConfig:
    config = (
        GenerationConfig.from_file(args.config)
        if args.config
        else GenerationConfig()
    )
    config.backend_name = args.backend
    if args.run_id:
        config.run_id = args.run_id
    if args.seed is not None:
        config.seed = args.seed
    if args.review_rate is not None:
        config.review_rate = args.review_rate
    if args.max_cases is not None:
        config.max_cases_per_benchmark = args.max_cases
    if args.source:
        config.source_uris.update(_parse_sources(args.source, registry))
    config.__post_init__()
    return config


def _cmd_generate(args: argparse.Namespace) -> int:
    registry = (
        BenchmarkRegistry.from_file(args.suite_config)
        if args.suite_config
        else builtin_registry()
    )
    config = _load_generation_config(args, registry)
    _, model_pool = load_model_pool(config.model_pool_path or None)
    if args.backend == "mock":
        backend = MockBackend(model_pool)
    elif args.backend == "replay":
        if not args.replay_log:
            raise SystemExit("--backend replay requires --replay-log")
        backend = ReplayBackend(args.replay_log)
    else:
        if not args.live_backend:
            raise SystemExit("--backend live requires --live-backend module:factory")
        backend = load_live_backend(
            args.live_backend,
            {**config.to_dict(), "benchmark_suite": registry.to_dict()},
        )

    pipeline = GenerationPipeline(
        output_dir=args.output_dir,
        backend=backend,
        config=config,
        registry=registry,
    )
    manifest = pipeline.generate(args.benchmark)
    if args.reviews:
        manifest = apply_reviews(args.output_dir, args.reviews)
    _print_json(manifest)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    _print_json(run_pipeline(args.pipeline_config, output_dir=args.output_dir))
    return 0


def _cmd_config_validate(args: argparse.Namespace) -> int:
    request = PipelineRequest.from_file(
        args.pipeline_config, output_dir=args.output_dir or ".validation-output"
    )
    _print_json(request.summary())
    return 0


def _cmd_suite_validate(args: argparse.Namespace) -> int:
    registry = BenchmarkRegistry.from_file(args.suite_config)
    _print_json(
        {
            "schema": "twinrouterbench.benchmark_suite_validation.v1",
            "version": registry.version,
            "fingerprint": registry.fingerprint,
            "benchmarks": list(registry.names),
            "definitions": {
                name: registry.get(name).to_dict() for name in registry.names
            },
        }
    )
    return 0


def _cmd_review_export(args: argparse.Namespace) -> int:
    output = export_review_queue(args.run_dir, args.output)
    print(output)
    return 0


def _cmd_review_apply(args: argparse.Namespace) -> int:
    _print_json(apply_reviews(args.run_dir, args.reviews))
    return 0


def _cmd_publish(args: argparse.Namespace) -> int:
    _print_json(publish_runs(args.runs, args.output_dir))
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    _print_json(validate_public_dataset(args.path))
    return 0


def _cmd_source_inspect(args: argparse.Namespace) -> int:
    source = SourceSpec.parse(args.uri)
    result = source.to_dict()
    if source.kind == "local":
        path = Path(source.locator).expanduser()
        result["exists"] = path.exists()
        result["resolved_path"] = str(path.resolve())
    else:
        result["materialization"] = (
            "remote registry entry; download explicitly and rerun with local://"
        )
    _print_json(result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="twinrouterbench data",
        description="Construct, review, and publish static routing supervision.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="Run the construction pipeline.")
    generate.add_argument("--benchmark", default="all")
    generate.add_argument("--backend", choices=("mock", "replay", "live"), default="mock")
    generate.add_argument("--output-dir", required=True)
    generate.add_argument("--config", help="Optional JSON generation config.")
    generate.add_argument(
        "--suite-config",
        "--benchmark-config",
        dest="suite_config",
        help="Config-driven benchmark suite JSON; defaults to built-ins.",
    )
    generate.add_argument("--source", action="append", default=[], help="benchmark=URI")
    generate.add_argument("--replay-log")
    generate.add_argument("--live-backend", help="module:factory plugin")
    generate.add_argument("--reviews", help="Apply a completed review JSONL after generation.")
    generate.add_argument("--run-id")
    generate.add_argument("--seed", type=int)
    generate.add_argument("--review-rate", type=float)
    generate.add_argument(
        "--max-cases",
        type=int,
        help="Limit each selected benchmark to its first N normalized cases.",
    )
    generate.set_defaults(func=_cmd_generate)

    run = sub.add_parser(
        "run", help="Run a complete benchmark-agnostic pipeline config."
    )
    run.add_argument("--config", dest="pipeline_config", required=True)
    run.add_argument("--output-dir", help="Override run.output_dir from the config.")
    run.set_defaults(func=_cmd_run)

    config = sub.add_parser("config", help="Validate a complete pipeline config.")
    config_sub = config.add_subparsers(dest="config_command", required=True)
    config_validate = config_sub.add_parser("validate")
    config_validate.add_argument("--config", dest="pipeline_config", required=True)
    config_validate.add_argument("--output-dir")
    config_validate.set_defaults(func=_cmd_config_validate)

    suite = sub.add_parser("suite", help="Inspect or validate a benchmark suite.")
    suite_sub = suite.add_subparsers(dest="suite_command", required=True)
    suite_validate = suite_sub.add_parser("validate")
    suite_validate.add_argument("--config", dest="suite_config", required=True)
    suite_validate.set_defaults(func=_cmd_suite_validate)

    review = sub.add_parser("review", help="Export or apply manual audit verdicts.")
    review_sub = review.add_subparsers(dest="review_command", required=True)
    review_export = review_sub.add_parser("export")
    review_export.add_argument("--run-dir", required=True)
    review_export.add_argument("--output")
    review_export.set_defaults(func=_cmd_review_export)
    review_apply = review_sub.add_parser("apply")
    review_apply.add_argument("--run-dir", required=True)
    review_apply.add_argument("--reviews", required=True)
    review_apply.set_defaults(func=_cmd_review_apply)

    publish = sub.add_parser("publish", help="Merge ready runs into a release candidate.")
    publish.add_argument("--runs", nargs="+", required=True)
    publish.add_argument("--output-dir", required=True)
    publish.set_defaults(func=_cmd_publish)

    validate = sub.add_parser("validate", help="Validate a public question bank.")
    validate.add_argument("--path", required=True)
    validate.set_defaults(func=_cmd_validate)

    source = sub.add_parser("source", help="Inspect a versioned source URI.")
    source_sub = source.add_subparsers(dest="source_command", required=True)
    inspect = source_sub.add_parser("inspect")
    inspect.add_argument("uri")
    inspect.set_defaults(func=_cmd_source_inspect)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
