"""Single public entry point for config-driven data generation."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .backends import ExecutionBackend, MockBackend, ReplayBackend, load_live_backend
from .benchmarking import BenchmarkRegistry, SUITE_SCHEMA, builtin_registry
from .pipeline import GenerationConfig, GenerationPipeline, load_model_pool
from .publish import apply_reviews
from .source import SourceSpec


PIPELINE_SCHEMA = "twinrouterbench.data_pipeline.v1"
_SECRET_KEYS = {"api_key", "apikey", "token", "password", "secret", "authorization"}


def _reject_inline_credentials(value: Any, path: str = "config") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}"
            if str(key).lower().replace("-", "_") in _SECRET_KEYS and str(item).strip():
                raise ValueError(
                    f"credentials are not allowed in pipeline config ({child}); "
                    "read them from the environment or a secret manager"
                )
            _reject_inline_credentials(item, child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_inline_credentials(item, f"{path}[{index}]")


def _local_path(value: str, base_dir: Path) -> str:
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else (base_dir / path).resolve())


def _local_uri(value: str, base_dir: Path) -> str:
    source = SourceSpec.parse(value)
    if source.kind != "local":
        return value
    return f"local://{_local_path(source.locator, base_dir)}"


@dataclass
class PipelineRequest:
    registry: BenchmarkRegistry
    generation: GenerationConfig
    backend_config: dict[str, Any]
    benchmarks: str | list[str]
    output_dir: Path
    reviews: Path | None = None

    @classmethod
    def from_dict(
        cls,
        raw: dict[str, Any],
        *,
        base_dir: str | Path = ".",
        output_dir: str | Path | None = None,
    ) -> "PipelineRequest":
        if not isinstance(raw, dict):
            raise ValueError("pipeline config must be a JSON object")
        if str(raw.get("schema", PIPELINE_SCHEMA)) != PIPELINE_SCHEMA:
            raise ValueError(f"unsupported pipeline schema {raw.get('schema')!r}")
        _reject_inline_credentials(raw)
        base = Path(base_dir).expanduser().resolve()

        suite_raw = raw.get("suite")
        if suite_raw in (None, "builtin") and "benchmarks" not in raw:
            registry = builtin_registry()
        elif isinstance(suite_raw, str):
            registry = BenchmarkRegistry.from_file(_local_path(suite_raw, base))
        elif isinstance(suite_raw, dict):
            registry = BenchmarkRegistry.from_dict(suite_raw, base_dir=base)
        elif "benchmarks" in raw:
            registry = BenchmarkRegistry.from_dict(
                {
                    "schema": SUITE_SCHEMA,
                    "version": raw.get("suite_version", "inline"),
                    "benchmarks": raw["benchmarks"],
                },
                base_dir=base,
            )
        else:
            raise ValueError("suite must be 'builtin', a JSON path, or an object")

        generation_raw = copy.deepcopy(raw.get("generation") or {})
        if not isinstance(generation_raw, dict):
            raise ValueError("generation must be an object")
        if generation_raw.get("model_pool_path"):
            generation_raw["model_pool_path"] = _local_path(
                str(generation_raw["model_pool_path"]), base
            )
        sources = generation_raw.get("source_uris") or {}
        if not isinstance(sources, dict):
            raise ValueError("generation.source_uris must be an object")
        generation_raw["source_uris"] = {
            str(name): _local_uri(str(uri), base) for name, uri in sources.items()
        }
        generation = GenerationConfig(**generation_raw)

        backend_config = copy.deepcopy(raw.get("backend") or {"type": "mock"})
        if isinstance(backend_config, str):
            backend_config = {"type": backend_config}
        if not isinstance(backend_config, dict):
            raise ValueError("backend must be a string or object")
        backend_type = str(backend_config.get("type", "mock"))
        if backend_type not in {"mock", "replay", "plugin"}:
            raise ValueError("backend.type must be mock, replay, or plugin")
        if backend_type == "replay" and backend_config.get("log"):
            backend_config["log"] = _local_path(str(backend_config["log"]), base)
        generation.backend_name = backend_type

        run = raw.get("run") or {}
        if not isinstance(run, dict):
            raise ValueError("run must be an object")
        selected: str | list[str] = run.get("benchmarks", "all")
        if isinstance(selected, str):
            if selected != "all":
                registry.get(selected)
        elif isinstance(selected, list) and selected:
            selected = [str(item) for item in selected]
            for name in selected:
                registry.get(name)
        else:
            raise ValueError("run.benchmarks must be 'all', a name, or a non-empty list")

        configured_output = output_dir or run.get("output_dir")
        if not configured_output:
            raise ValueError("output_dir is required as an argument or run.output_dir")
        output = Path(configured_output).expanduser()
        if output_dir is None and not output.is_absolute():
            output = (base / output).resolve()
        reviews = run.get("reviews")
        review_path = Path(_local_path(str(reviews), base)) if reviews else None
        return cls(
            registry=registry,
            generation=generation,
            backend_config=backend_config,
            benchmarks=selected,
            output_dir=output,
            reviews=review_path,
        )

    @classmethod
    def from_file(
        cls, path: str | Path, *, output_dir: str | Path | None = None
    ) -> "PipelineRequest":
        path = Path(path).expanduser().resolve()
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(raw, base_dir=path.parent, output_dir=output_dir)

    def summary(self) -> dict[str, Any]:
        return {
            "schema": PIPELINE_SCHEMA,
            "suite_version": self.registry.version,
            "suite_fingerprint": self.registry.fingerprint,
            "benchmarks": list(self.registry.names),
            "selected_benchmarks": self.benchmarks,
            "backend": self.backend_config.get("type", "mock"),
            "output_dir": str(self.output_dir),
            "generation": self.generation.to_dict(),
        }


def _create_backend(request: PipelineRequest) -> ExecutionBackend:
    kind = str(request.backend_config.get("type", "mock"))
    _, model_pool = load_model_pool(request.generation.model_pool_path or None)
    if kind == "mock":
        return MockBackend(model_pool)
    if kind == "replay":
        log = request.backend_config.get("log")
        if not log:
            raise ValueError("replay backend requires backend.log")
        return ReplayBackend(str(log))
    factory = str(request.backend_config.get("factory", ""))
    if not factory:
        raise ValueError("plugin backend requires backend.factory")
    context = {
        **request.generation.to_dict(),
        "backend_options": copy.deepcopy(request.backend_config.get("options") or {}),
        "benchmark_suite": request.registry.to_dict(),
    }
    return load_live_backend(factory, context)


def run_pipeline(
    config: str | Path | dict[str, Any] | PipelineRequest,
    *,
    output_dir: str | Path | None = None,
    backend: ExecutionBackend | None = None,
) -> dict[str, Any]:
    """Run any configured benchmark suite through one stable Python interface.

    ``backend`` is injectable for tests or private harnesses. Otherwise the
    backend is created from the credential-free config and provider plugins
    read secrets from their environment.
    """

    if isinstance(config, PipelineRequest):
        request = config
        if output_dir is not None:
            request = copy.copy(request)
            request.output_dir = Path(output_dir)
    elif isinstance(config, (str, Path)):
        request = PipelineRequest.from_file(config, output_dir=output_dir)
    else:
        request = PipelineRequest.from_dict(
            config, output_dir=output_dir, base_dir=Path.cwd()
        )
    pipeline = GenerationPipeline(
        output_dir=request.output_dir,
        backend=backend or _create_backend(request),
        config=request.generation,
        registry=request.registry,
    )
    manifest = pipeline.generate(request.benchmarks)
    if request.reviews:
        manifest = apply_reviews(request.output_dir, request.reviews)
    return manifest
