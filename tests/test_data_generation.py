from __future__ import annotations

import hashlib
import json
import sys
import types
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

from twinrouterbench.data_generation.adapters import BENCHMARK_ORDER, get_adapter
from twinrouterbench.data_generation.api import PipelineRequest, run_pipeline
from twinrouterbench.data_generation.backends import MockBackend, ReplayBackend
from twinrouterbench.data_generation.benchmarking import BenchmarkRegistry
from twinrouterbench.data_generation.cache import TrialCache
from twinrouterbench.data_generation.commonstack_smoke import CommonStackSmokeBackend
from twinrouterbench.data_generation.cli import main as data_cli_main
from twinrouterbench.data_generation.pipeline import (
    GenerationConfig,
    GenerationPipeline,
    load_model_pool,
)
from twinrouterbench.data_generation.publish import (
    apply_reviews,
    publish_runs,
    validate_public_dataset,
)
from twinrouterbench.data_generation.source import SourceSpec
from twinrouterbench.data_generation.types import (
    Assignment,
    ExecutionResult,
    JudgeResult,
    TaskSpec,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _mock_pipeline(
    output_dir: Path,
    *,
    benchmark: str = "all",
    review_rate: float = 0.1,
    backend: MockBackend | ReplayBackend | None = None,
) -> dict[str, Any]:
    _, pool = load_model_pool()
    config = GenerationConfig(review_rate=review_rate)
    pipeline = GenerationPipeline(
        output_dir=output_dir,
        backend=backend or MockBackend(pool),
        config=config,
    )
    return pipeline.generate(benchmark)


def _complete_reviews(run_dir: Path, *, adjust_benchmark: str | None = None) -> Path:
    queue = _read_jsonl(run_dir / "review_queue.jsonl")
    adjusted = False
    for row in queue:
        if (
            adjust_benchmark
            and row["benchmark"] == adjust_benchmark
            and row["target_tier"] != "low"
            and not adjusted
        ):
            row["verdict"] = "further_downgradeable"
            adjusted = True
        else:
            row["verdict"] = "tight"
        row["reviewer"] = "test-reviewer"
        row["notes"] = "offline fixture audit"
    path = run_dir / "completed_reviews.jsonl"
    _write_jsonl(path, queue)
    return path


def test_source_specs_are_versioned_and_never_auto_download(tmp_path: Path) -> None:
    github = SourceSpec.parse(
        "github://owner/repo@0123456789abcdef0123456789abcdef01234567/data/tasks.jsonl"
    )
    assert github.kind == "github"
    assert github.locator == "owner/repo"
    assert github.is_pinned is True
    with pytest.raises(ValueError, match="materialize"):
        github.require_materialized_path()

    hf = SourceSpec.parse("hf://org/dataset@main/data.jsonl")
    assert hf.kind == "hf"
    assert hf.is_pinned is False

    local_file = tmp_path / "tasks.jsonl"
    local_file.write_text("", encoding="utf-8")
    local = SourceSpec.parse(f"local://{local_file}")
    assert local.require_materialized_path() == local_file


def test_v2_model_pool_matches_paper_shape() -> None:
    version, pool = load_model_pool()
    assert version == "twinrouterbench-static-v2-paper-2026-05-08"
    assert tuple(pool) == ("low", "mid", "mid_high", "high")
    assert [len(pool[tier]) for tier in pool] == [3, 3, 3, 2]
    assert len({model for models in pool.values() for model in models}) == 11


def test_all_five_adapters_generate_and_hardened_judge_rejects_pseudo_pass(
    tmp_path: Path,
) -> None:
    manifest = _mock_pipeline(tmp_path / "run")
    assert manifest["benchmarks"] == list(BENCHMARK_ORDER)
    assert manifest["counts"]["emitted_rows"] == {
        "bfcl": 2,
        "mtrag": 1,
        "pinchbench": 2,
        "qmsum": 1,
        "swebench": 3,
    }
    assert manifest["counts"]["rejected_instances"] == {"mtrag": 1}
    rejections = _read_jsonl(tmp_path / "run" / "rejections.jsonl")
    assert rejections[0]["stage"] == "open_ended_judge"
    assert "Evidence conflict" in rejections[0]["reason"]


def test_sequential_search_uses_cascade_and_respects_not_downgradeable(
    tmp_path: Path,
) -> None:
    _mock_pipeline(tmp_path / "run", benchmark="swebench")
    trials = _read_jsonl(tmp_path / "run" / "execution_trials.jsonl")
    step_one_mid = [
        row
        for row in trials
        if row["target_step"] == 1 and row["candidate_tier"] == "mid"
    ]
    assert [row["cascade_index"] for row in step_one_mid] == [1, 2]
    assert step_one_mid[0]["passed"] is False
    assert step_one_mid[1]["accepted"] is True
    assert not [row for row in trials if row["target_step"] == 3]

    labels = _read_jsonl(tmp_path / "run" / "labels.pre_review.jsonl")
    assert [row["target_tier"] for row in labels] == ["mid", "low", "high"]


def test_mixed_prefixes_use_locked_outputs_and_cross_step_failures_are_retried(
    tmp_path: Path,
) -> None:
    _mock_pipeline(tmp_path / "run", benchmark="pinchbench")
    labels = _read_jsonl(tmp_path / "run" / "labels.pre_review.jsonl")
    assert [row["target_tier"] for row in labels] == ["mid", "mid"]
    second_prefix = labels[1]["messages"]
    assert any(
        message.get("content") == "synthetic response for step 1 at tier mid"
        for message in second_prefix
    )
    trials = _read_jsonl(tmp_path / "run" / "execution_trials.jsonl")
    low_second = next(
        row
        for row in trials
        if row["target_step"] == 2 and row["candidate_tier"] == "low"
    )
    assert low_second["passed"] is False
    assert "cross-step" in low_second["reason"]


def test_trial_cache_invalidates_only_downstream_entries() -> None:
    cache = TrialCache()
    params = {"temperature": 0}
    assignments = [Assignment("high", "h"), Assignment("high", "h")]
    result = ExecutionResult(True, 2, ((), ()), ({}, {}), "ok")
    first = cache.make_key("case", 1, assignments, params)
    second = cache.make_key("case", 2, assignments, params)
    cache.put(first, result)
    cache.put(second, result)
    assert cache.invalidate_downstream("case", 1) == 1
    assert cache.get(first) == result
    assert cache.get(second) is None


def test_passing_candidate_with_changed_step_count_is_not_accepted(tmp_path: Path) -> None:
    _, pool = load_model_pool()

    class ChangedLengthBackend(MockBackend):
        def run_mixed(
            self,
            task: Any,
            assignments: Any,
            *,
            target_step: int,
            generation_parameters: dict[str, Any],
        ) -> ExecutionResult:
            result = super().run_mixed(
                task,
                assignments,
                target_step=target_step,
                generation_parameters=generation_parameters,
            )
            if target_step == 2 and assignments[1].tier == "low":
                return ExecutionResult(
                    passed=True,
                    step_count=result.step_count + 1,
                    prefixes=result.prefixes,
                    responses=result.responses,
                    reason="fixture changes the number of trajectory steps",
                )
            return result

    run = tmp_path / "run"
    pipeline = GenerationPipeline(
        output_dir=run,
        backend=ChangedLengthBackend(pool),
        config=GenerationConfig(),
    )
    pipeline.generate("swebench")
    trials = _read_jsonl(run / "execution_trials.jsonl")
    changed = next(
        row
        for row in trials
        if row["target_step"] == 2 and row["candidate_tier"] == "low"
    )
    assert changed["passed"] is True
    assert changed["same_step_count"] is False
    assert changed["accepted"] is False
    labels = _read_jsonl(run / "labels.pre_review.jsonl")
    assert labels[1]["target_tier"] == "mid"


def test_review_can_lower_one_tier_and_uncertain_blocks_publication(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    _mock_pipeline(run, benchmark="pinchbench", review_rate=1.0)
    reviews = _complete_reviews(run, adjust_benchmark="pinchbench")
    manifest = apply_reviews(run, reviews)
    assert manifest["status"] == "ready"
    final_rows = _read_jsonl(run / "labels.final.jsonl")
    adjusted = [row for row in final_rows if row.get("review_adjusted")]
    assert len(adjusted) == 1
    assert adjusted[0]["review_original_tier"] == "mid"
    assert adjusted[0]["target_tier"] == "low"

    blocked_run = tmp_path / "blocked"
    _mock_pipeline(blocked_run, benchmark="swebench", review_rate=1.0)
    queue = _read_jsonl(blocked_run / "review_queue.jsonl")
    for row in queue:
        row["verdict"] = "uncertain"
    uncertain = blocked_run / "uncertain.jsonl"
    _write_jsonl(uncertain, queue)
    blocked = apply_reviews(blocked_run, uncertain)
    assert blocked["status"] == "review_incomplete"
    assert not (blocked_run / "labels.final.jsonl").exists()
    with pytest.raises(ValueError, match="not ready"):
        publish_runs([blocked_run], tmp_path / "blocked-publish")


def test_publish_is_isolated_validated_and_strips_internal_model_fields(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    _mock_pipeline(run, review_rate=1.0)
    apply_reviews(run, _complete_reviews(run))
    candidate = tmp_path / "candidate"
    manifest = publish_runs([run], candidate)
    assert manifest["total_line_count"] == 9
    summary = validate_public_dataset(candidate)
    assert summary["instances"] == 5
    for row in _read_jsonl(candidate / "question_bank.jsonl"):
        assert "selected_model" not in row
        assert "reviewer" not in row
        assert row["pipeline_stage"] == "ground_truth_ready"

    repo_static = Path(__file__).resolve().parents[1] / "data" / "static"
    with pytest.raises(ValueError, match="refuses to overwrite"):
        publish_runs([run], repo_static)


def test_publish_rejects_unpinned_remote_provenance(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _mock_pipeline(run, benchmark="qmsum")
    rows = _read_jsonl(run / "labels.final.jsonl")
    rows[0]["source"]["uri"] = "hf://org/dataset@main/tasks.jsonl"
    _write_jsonl(run / "labels.final.jsonl", rows)
    with pytest.raises(ValueError, match="unpinned remote source"):
        publish_runs([run], tmp_path / "candidate")


def test_replay_reproduces_mock_labels_and_trials_byte_for_byte(tmp_path: Path) -> None:
    mock_run = tmp_path / "mock"
    _mock_pipeline(mock_run)
    replay_run = tmp_path / "replay"
    _mock_pipeline(
        replay_run,
        backend=ReplayBackend(mock_run / "backend_events.jsonl"),
    )
    for name in ("labels.pre_review.jsonl", "execution_trials.jsonl"):
        assert (mock_run / name).read_bytes() == (replay_run / name).read_bytes()


def test_mock_golden_hashes_detect_protocol_drift(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _mock_pipeline(run)
    expected = {
        "labels.pre_review.jsonl": "6d23241b0d11d9e7bbd837b397dac26ab333d9a7152dd5f7e5b588c08b647f0b",
        "execution_trials.jsonl": "cad5067e9e9e740834a88e73524f4466e461aaf63e5c1ca8dbf7610b24348bb6",
    }
    for name, digest in expected.items():
        assert hashlib.sha256((run / name).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("benchmark", BENCHMARK_ORDER)
def test_single_benchmark_generation(benchmark: str, tmp_path: Path) -> None:
    run = tmp_path / benchmark
    manifest = _mock_pipeline(run, benchmark=benchmark)
    assert manifest["benchmarks"] == [benchmark]
    labels = _read_jsonl(run / "labels.pre_review.jsonl")
    assert labels and {row["benchmark"] for row in labels} == {benchmark}


def test_max_cases_limits_each_benchmark(tmp_path: Path) -> None:
    _, pool = load_model_pool()
    pipeline = GenerationPipeline(
        output_dir=tmp_path / "limited",
        backend=MockBackend(pool),
        config=GenerationConfig(max_cases_per_benchmark=1, review_rate=0),
    )
    manifest = pipeline.generate("all")
    assert manifest["counts"]["seed_instances"] == {
        benchmark: 1 for benchmark in BENCHMARK_ORDER
    }
    assert manifest["counts"]["rejected_instances"] == {}


def test_commonstack_smoke_backend_calls_provider_without_recording_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, pool = load_model_pool()
    calls: list[dict[str, Any]] = []

    class Response(BytesIO):
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: Any) -> None:
            self.close()

    def fake_urlopen(request: Any, timeout: float) -> Response:
        calls.append(
            {
                "url": request.full_url,
                "authorization": request.headers["Authorization"],
                "body": json.loads(request.data),
                "timeout": timeout,
            }
        )
        return Response(b'{"choices":[{"message":{"content":"OK"}}]}')

    monkeypatch.setattr(
        "twinrouterbench.data_generation.commonstack_smoke.urllib.request.urlopen",
        fake_urlopen,
    )
    backend = CommonStackSmokeBackend(pool, api_key="secret-test-key")
    task = get_adapter("qmsum").load_tasks()[0]
    result = backend.run_seed(task, pool["high"][0])
    assert result.passed is True
    assert calls[0]["url"].endswith("/v1/chat/completions")
    assert calls[0]["authorization"] == "Bearer secret-test-key"
    assert "secret-test-key" not in json.dumps(calls[0]["body"])
    assert "What cache decision was made?" in calls[0]["body"]["messages"][1]["content"]


def test_local_normalized_source_and_cli_entrypoint(tmp_path: Path) -> None:
    fixture_task = get_adapter("qmsum").load_tasks()[0].to_dict()
    source_path = tmp_path / "tasks.jsonl"
    _write_jsonl(source_path, [fixture_task])
    output = tmp_path / "cli-run"
    assert (
        data_cli_main(
            [
                "generate",
                "--benchmark",
                "qmsum",
                "--backend",
                "mock",
                "--source",
                f"qmsum=local://{source_path}",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    assert (output / "labels.final.jsonl").is_file()
    manifest_text = (output / "manifest.json").read_text(encoding="utf-8")
    assert str(source_path) not in manifest_text


def test_generation_refuses_to_overwrite_an_existing_run(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _mock_pipeline(run, benchmark="qmsum")
    with pytest.raises(FileExistsError, match="already contains artifacts"):
        _mock_pipeline(run, benchmark="qmsum")


def test_config_only_custom_benchmark_runs_through_public_api(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    config = repo / "configs" / "data_generation" / "custom_qa_pipeline.json"
    output = tmp_path / "custom-qa"
    manifest = run_pipeline(config, output_dir=output)
    assert manifest["benchmarks"] == ["custom_qa"]
    assert manifest["benchmark_suite_version"] == "custom-qa-example-v1"
    assert manifest["counts"]["seed_instances"] == {"custom_qa": 2}
    rows = _read_jsonl(output / "labels.final.jsonl")
    assert [row["target_tier"] for row in rows] == ["low", "mid"]
    assert {row["benchmark"] for row in rows} == {"custom_qa"}
    assert str(config.parent) not in (output / "manifest.json").read_text()


def test_config_only_exact_match_uses_generic_openai_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "qa.jsonl"
    _write_jsonl(
        source,
        [
            {
                "id": "capital",
                "question": "What is the capital of France?",
                "answer": "Paris",
                "hint": "low",
            }
        ],
    )
    pool = tmp_path / "pool.json"
    pool.write_text(
        json.dumps(
            {
                "version": "unit-pool",
                "tiers": {
                    "low": ["model-low"],
                    "mid": ["model-mid"],
                    "mid_high": ["model-mid-high"],
                    "high": ["model-high"],
                },
            }
        )
    )
    calls: list[str] = []

    class Response(BytesIO):
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: Any) -> None:
            self.close()

    def fake_urlopen(request: Any, timeout: float) -> Response:
        del timeout
        body = json.loads(request.data)
        model = body["model"]
        calls.append(model)
        content = "Lyon" if model == "model-low" else "Paris"
        return Response(
            json.dumps({"choices": [{"message": {"content": content}}]}).encode()
        )

    monkeypatch.setenv("UNIT_TEST_API_KEY", "environment-only-secret")
    monkeypatch.setattr(
        "twinrouterbench.data_generation.openai_backend.urllib.request.urlopen",
        fake_urlopen,
    )
    config = {
        "schema": "twinrouterbench.data_pipeline.v1",
        "suite_version": "unit-custom-suite",
        "generation": {
            "review_rate": 0,
            "cascade_size": 1,
            "model_pool_path": str(pool),
        },
        "backend": {
            "type": "plugin",
            "factory": "twinrouterbench.data_generation.openai_backend:create_backend",
            "options": {
                "api_key_env": "UNIT_TEST_API_KEY",
                "default_base_url": "https://unit.test/v1",
            },
        },
        "benchmarks": {
            "unseen_qa": {
                "display_name": "Previously Unseen QA",
                "scenario": "qa",
                "source": {
                    "uri": f"local://{source}",
                    "license": "CC0-1.0",
                    "version": "unit-v1",
                },
                "loader": {"type": "single_turn"},
                "evaluation": {
                    "trial": "exact_match",
                    "final": "exact_match",
                },
            }
        },
        "run": {"output_dir": str(tmp_path / "ignored")},
    }
    output = tmp_path / "exact-run"
    manifest = run_pipeline(config, output_dir=output)
    assert manifest["status"] == "ready"
    assert calls == ["model-high", "model-low", "model-mid"]
    trials = _read_jsonl(output / "execution_trials.jsonl")
    assert [(row["candidate_tier"], row["passed"]) for row in trials] == [
        ("low", False),
        ("mid", True),
    ]
    assert _read_jsonl(output / "labels.final.jsonl")[0]["target_tier"] == "mid"
    assert "environment-only-secret" not in "".join(
        path.read_text(errors="ignore") for path in output.iterdir()
    )


def test_loader_and_evaluator_plugins_add_benchmark_without_core_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = types.ModuleType("trb_unit_plugin")

    class Loader:
        def load_tasks(self, benchmark: Any, source: Any) -> list[TaskSpec]:
            del source
            return [
                TaskSpec.from_dict(
                    {
                        "benchmark": benchmark.name,
                        "benchmark_display": benchmark.display_name,
                        "scenario": benchmark.default_scenario,
                        "instance_id": "plugin-case",
                        "initial_messages": [
                            {"role": "user", "content": "plugin task"}
                        ],
                        "steps": [{"hint": "low", "minimum_tier": "low"}],
                        "source": {
                            "uri": "fixture://plugin-case",
                            "license": "CC0-1.0",
                            "version": "plugin-v1",
                        },
                        "benchmark_version": "plugin-v1",
                    }
                )
            ]

    class Evaluator:
        stage = "plugin_evaluator"

        def evaluate(
            self, task: TaskSpec, execution: ExecutionResult, backend: Any
        ) -> JudgeResult:
            del task, backend
            return JudgeResult(
                passed=execution.passed,
                faithfulness=execution.passed,
                appropriateness=execution.passed,
                completeness=execution.passed,
                reason="plugin verdict",
            )

    module.create_loader = lambda options: Loader()
    module.create_evaluator = lambda options: Evaluator()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    materialized = tmp_path / "placeholder.json"
    materialized.write_text("[]")
    registry = BenchmarkRegistry.from_dict(
        {
            "schema": "twinrouterbench.benchmark_suite.v1",
            "version": "plugin-suite",
            "benchmarks": {
                "plugin_bench": {
                    "display_name": "Plugin Bench",
                    "scenario": "plugin",
                    "source": f"local://{materialized}",
                    "loader": {
                        "type": "plugin",
                        "factory": "trb_unit_plugin:create_loader",
                    },
                    "evaluation": {
                        "trial": {
                            "type": "plugin",
                            "factory": "trb_unit_plugin:create_evaluator",
                        },
                        "final": {
                            "type": "plugin",
                            "factory": "trb_unit_plugin:create_evaluator",
                        },
                    },
                }
            },
        }
    )
    _, pool_data = load_model_pool()
    output = tmp_path / "plugin-run"
    manifest = GenerationPipeline(
        output_dir=output,
        backend=MockBackend(pool_data),
        registry=registry,
        config=GenerationConfig(review_rate=0),
    ).generate("plugin_bench")
    assert manifest["counts"]["emitted_rows"] == {"plugin_bench": 1}


def test_pipeline_config_rejects_inline_credentials(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="credentials are not allowed"):
        PipelineRequest.from_dict(
            {
                "schema": "twinrouterbench.data_pipeline.v1",
                "suite": "builtin",
                "backend": {"type": "plugin", "api_key": "do-not-store-me"},
                "run": {"output_dir": str(tmp_path / "run")},
            }
        )


def test_config_driven_cli_run(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    config = repo / "configs" / "data_generation" / "custom_qa_pipeline.json"
    output = tmp_path / "cli-config-run"
    assert (
        data_cli_main(
            ["run", "--config", str(config), "--output-dir", str(output)]
        )
        == 0
    )
    assert _read_jsonl(output / "labels.final.jsonl")[1]["target_tier"] == "mid"
