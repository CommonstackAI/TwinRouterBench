"""Reference data-construction pipeline for the TwinRouterBench static track.

The package exposes the paper's construction stages as regular Python APIs.  It
is deliberately backend-agnostic: tests use deterministic fixtures, previous
runs can be replayed, and live harnesses can be supplied as plugins.
"""

from .api import PipelineRequest, run_pipeline
from .benchmarking import (
    BenchmarkRegistry,
    BenchmarkSpec,
    ExecutionEvaluator,
    TaskLoader,
)
from .pipeline import GenerationConfig, GenerationPipeline
from .publish import apply_reviews, publish_runs, validate_public_dataset

__all__ = [
    "GenerationConfig",
    "GenerationPipeline",
    "PipelineRequest",
    "BenchmarkRegistry",
    "BenchmarkSpec",
    "TaskLoader",
    "ExecutionEvaluator",
    "run_pipeline",
    "apply_reviews",
    "publish_runs",
    "validate_public_dataset",
]
