"""Regression tests for verified search; no external services or paid calls."""

from __future__ import annotations

import copy
import hashlib
import json
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from twinrouterbench.data_generation.adapters import get_adapter
from twinrouterbench.data_generation.backends import (
    MockBackend, RecordingBackend, ReplayBackend, locked_prefix_key,
)
from twinrouterbench.data_generation.benchmarking import PassThroughEvaluator
from twinrouterbench.data_generation.cache import TrialCache
from twinrouterbench.data_generation.openai_backend import OpenAICompatibleBackend
from twinrouterbench.data_generation.pipeline import (
    GenerationConfig, GenerationPipeline, load_model_pool,
)
from twinrouterbench.data_generation.types import Assignment, ExecutionResult, JudgeResult


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


@pytest.fixture
def generic() -> OpenAICompatibleBackend:
    _, pool = load_model_pool()
    return OpenAICompatibleBackend(pool, api_key="local-test", base_url="http://127.0.0.1:1")


@pytest.mark.parametrize("benchmark", ["swebench", "bfcl", "pinchbench"])
def test_generic_rejects_agentic_benchmarks_before_any_request(
    generic: OpenAICompatibleBackend, monkeypatch: Any, tmp_path: Path, benchmark: str,
) -> None:
    def unexpected(**kwargs: Any) -> Any:
        pytest.fail("unsupported benchmark must fail before making a provider request")

    monkeypatch.setattr(generic, "_chat", unexpected)
    run = tmp_path / benchmark
    with pytest.raises(ValueError, match="multi-step environments"):
        GenerationPipeline(output_dir=run, backend=generic).generate(benchmark)
    assert not run.exists()
    with pytest.raises(ValueError, match="single-turn"):
        generic.run_seed(get_adapter(benchmark).load_tasks()[0], generic.model_pool["high"][0])


def test_transport_completion_cannot_be_used_as_execution_verdict(
    generic: OpenAICompatibleBackend, monkeypatch: Any, tmp_path: Path,
) -> None:
    monkeypatch.setattr(generic, "_chat", lambda **kwargs: {"role": "assistant", "content": "wrong"})
    task = get_adapter("qmsum").load_tasks()[0]
    execution = generic.run_seed(task, generic.model_pool["high"][0])
    assert execution.verified is False
    with pytest.raises(ValueError, match="official harness verdict"):
        PassThroughEvaluator().evaluate(task, execution, generic)
    adapter = replace(get_adapter("qmsum"), trial_evaluator_config={"type": "execution"})
    with pytest.raises(ValueError, match="independent trial and final"):
        GenerationPipeline(
            output_dir=tmp_path / "unsafe", backend=generic, adapters={"qmsum": adapter},
        ).generate("qmsum")


def test_generic_refuses_unexecuted_tools(generic: OpenAICompatibleBackend, monkeypatch: Any) -> None:
    task = get_adapter("qmsum").load_tasks()[0]
    with pytest.raises(ValueError, match="tool execution"):
        generic.run_seed(replace(task, functions=({"name": "bash"},)), generic.model_pool["high"][0])
    monkeypatch.setattr(generic, "_chat", lambda **kwargs: {
        "role": "assistant", "tool_calls": [{"function": {"name": "bash"}}],
    })
    with pytest.raises(ValueError, match="cannot execute tool calls"):
        generic.run_seed(task, generic.model_pool["high"][0])


@pytest.mark.parametrize("bad", [
    {"passed": "false"}, {"uncertain": "false"}, {"faithfulness": 1},
    {"completeness": None}, {"evidence_conflict": 0}, {"appropriateness": []},
])
def test_judge_rejects_non_boolean_fields(
    generic: OpenAICompatibleBackend, monkeypatch: Any, bad: dict[str, Any],
) -> None:
    verdict = dict(passed=True, faithfulness=True, appropriateness=True,
                   completeness=True, evidence_conflict=False, uncertain=False)
    verdict.update(bad)
    monkeypatch.setattr(generic, "_chat", lambda **kwargs: {"content": json.dumps(verdict)})
    task = get_adapter("qmsum").load_tasks()[0]
    execution = MockBackend(generic.model_pool).run_seed(task, generic.model_pool["high"][0])
    with pytest.raises(ValueError, match="JSON boolean"):
        generic.judge_open_ended(task, execution)


def test_judge_rejects_missing_fields(generic: OpenAICompatibleBackend, monkeypatch: Any) -> None:
    monkeypatch.setattr(generic, "_chat", lambda **kwargs: {"content": '{"passed":true}'})
    task = get_adapter("qmsum").load_tasks()[0]
    execution = MockBackend(generic.model_pool).run_seed(task, generic.model_pool["high"][0])
    with pytest.raises(ValueError, match="faithfulness"):
        generic.judge_open_ended(task, execution)


def test_single_turn_http_search_loads_prompts_and_rejects_wrong_cheaper_answer(tmp_path: Path) -> None:
    """Exercise real HTTP serialization against a local deterministic server."""
    _, pool = load_model_pool()
    calls: list[dict[str, Any]] = []
    prompts = Path(__file__).resolve().parents[1] / "twinrouterbench/data_generation/prompts"
    proposer = (prompts / "tier_proposer_v1.txt").read_text()
    judge = (prompts / "open_ended_judge_v1.txt").read_text()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            pass

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(body)
            system = body["messages"][0].get("content")
            if system == proposer:
                content = json.dumps({"steps": [{"step_index": 1, "start_tier": "low", "reason": "probe"}]})
            elif system == judge:
                payload = json.loads(body["messages"][1]["content"])
                passed = payload["candidate_responses"][0]["content"] == "correct"
                content = json.dumps(dict(passed=passed, faithfulness=passed,
                    appropriateness=passed, completeness=passed,
                    evidence_conflict=False, uncertain=False, reason="local verdict"))
            else:
                content = "wrong" if body["model"] in pool["low"] else "correct"
            response = json.dumps({"choices": [{"message": {"role": "assistant", "content": content}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        backend = OpenAICompatibleBackend(pool, api_key="local-test", base_url=f"http://127.0.0.1:{server.server_port}")
        run = tmp_path / "http"
        config = GenerationConfig(cascade_size=1, review_rate=0,
            generation_parameters={"temperature": 0.6, "top_p": 0.85, "seed": 17})
        manifest = GenerationPipeline(output_dir=run, backend=backend, config=config).generate("qmsum")
        assert manifest["status"] == "ready"
        assert _rows(run / "labels.final.jsonl")[0]["target_tier"] == "mid"
        trials = _rows(run / "execution_trials.jsonl")
        assert [(row["candidate_tier"], row["accepted"]) for row in trials] == [("low", False), ("mid", True)]
        assert len([body for body in calls if body["messages"][0].get("content") == judge]) == 4
        generation_calls = [body for body in calls if body["messages"][0].get("content") not in (judge, proposer)]
        assert len(generation_calls) == 3  # Seed, low candidate, mid candidate.
        assert all(body["seed"] == 17 and body["temperature"] == 0.6 for body in generation_calls)
        lock = json.loads((run / "config.lock.json").read_text())
        for stage, version in config.prompt_versions.items():
            assert lock["prompt_artifacts"][stage]["sha256"] == hashlib.sha256((prompts / f"{version}.txt").read_bytes()).hexdigest()
        replay = tmp_path / "replay"
        GenerationPipeline(output_dir=replay, backend=ReplayBackend(run / "backend_events.jsonl"), config=config).generate("qmsum")
        assert (replay / "labels.final.jsonl").read_bytes() == (run / "labels.final.jsonl").read_bytes()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_prefix_drift_is_rejected_even_when_backend_reports_passed(tmp_path: Path) -> None:
    _, pool = load_model_pool()

    class DriftingBackend(MockBackend):
        def run_mixed(self, task: Any, assignments: Any, *, target_step: int,
                      generation_parameters: Any, locked_execution: Any = None) -> ExecutionResult:
            result = super().run_mixed(task, assignments, target_step=target_step,
                generation_parameters=generation_parameters, locked_execution=locked_execution)
            if target_step > 1:
                prefixes = copy.deepcopy(result.prefixes)
                prefixes[target_step - 1][0]["content"] = "different upstream output"
                return replace(result, prefixes=prefixes)
            return result

    run = tmp_path / "drift"
    GenerationPipeline(output_dir=run, backend=DriftingBackend(pool)).generate("swebench")
    trials = [row for row in _rows(run / "execution_trials.jsonl") if row["target_step"] == 2]
    assert any(row["passed"] for row in trials)
    assert all(not row["accepted"] and not row["locked_prefix_matches"] for row in trials)
    assert not _rows(run / "labels.pre_review.jsonl")
    assert _rows(run / "rejections.jsonl")[0]["stage"] == "trial_integrity"


def test_legacy_multistep_backend_fails_before_seed_execution(tmp_path: Path) -> None:
    _, pool = load_model_pool()

    class LegacyBackend(MockBackend):
        def run_seed(self, *args: Any) -> Any:
            pytest.fail("legacy backend must fail preflight before seed execution")

        def run_mixed(self, task: Any, assignments: Any, *, target_step: int,
                      generation_parameters: Any) -> Any:
            pytest.fail("legacy backend must fail preflight")

    run = tmp_path / "legacy"
    with pytest.raises(ValueError, match="must accept locked_execution"):
        GenerationPipeline(output_dir=run, backend=LegacyBackend(pool)).generate("swebench")
    assert not (run / "labels.final.jsonl").exists()


def test_cache_distinguishes_benchmarks_and_actual_prefixes() -> None:
    _, pool = load_model_pool()
    task = get_adapter("swebench").load_tasks()[0]
    original = MockBackend(pool).run_seed(task, pool["high"][0])
    changed = replace(original, responses=({"content": "different prefix response"},) + original.responses[1:])
    assignments = [Assignment("high", pool["high"][0]) for _ in task.steps]
    cache = TrialCache()
    a = cache.make_key("same-id", 2, assignments, {}, benchmark="a", locked_prefix_key=locked_prefix_key(original, 2))
    b = cache.make_key("same-id", 2, assignments, {}, benchmark="b", locked_prefix_key=locked_prefix_key(original, 2))
    c = cache.make_key("same-id", 2, assignments, {}, benchmark="a", locked_prefix_key=locked_prefix_key(changed, 2))
    assert len({a, b, c}) == 3
    cache.put(a, original)
    cache.put(b, original)
    assert cache.get(c) is None
    cache.invalidate_downstream("same-id", 1, benchmark="a")
    assert cache.get(a) is None and cache.get(b) == original


def test_replay_preserves_repeated_judge_verdicts_in_order(tmp_path: Path) -> None:
    _, pool = load_model_pool()
    task = get_adapter("qmsum").load_tasks()[0]
    execution = MockBackend(pool).run_seed(task, pool["high"][0])

    class AlternatingJudge(MockBackend):
        counter = 0

        def judge_open_ended(self, task: Any, execution: Any) -> JudgeResult:
            self.counter += 1
            passed = self.counter % 2 == 0
            return JudgeResult(passed, passed, passed, passed)

    recorder = RecordingBackend(AlternatingJudge(pool), tmp_path / "events.jsonl")
    assert recorder.judge_open_ended(task, execution).passed is False
    assert recorder.judge_open_ended(task, execution).passed is True
    replay = ReplayBackend(tmp_path / "events.jsonl")
    assert replay.judge_open_ended(task, execution).passed is False
    assert replay.judge_open_ended(task, execution).passed is True
    with pytest.raises(KeyError, match="no judge event"):
        replay.judge_open_ended(task, execution)


def test_legacy_results_without_verification_are_not_certified() -> None:
    raw = dict(passed=True, step_count=1, prefixes=[[]], responses=[{}])
    execution = ExecutionResult.from_dict(raw)
    assert execution.verified is False
    with pytest.raises(ValueError, match="official harness verdict"):
        PassThroughEvaluator().evaluate(None, execution, None)


def test_locked_responses_are_reused_instead_of_regenerated() -> None:
    _, pool = load_model_pool()
    task = get_adapter("swebench").load_tasks()[0]
    backend = MockBackend(pool)
    seed = backend.run_seed(task, pool["high"][0])
    prefixes = copy.deepcopy(seed.prefixes)
    prefixes[1][-2]["content"] = "previously accepted stochastic response"
    frozen_response = {"role": "assistant", "content": "previously accepted stochastic response"}
    locked = replace(seed, prefixes=prefixes, responses=(frozen_response,) + seed.responses[1:])
    result = backend.run_mixed(task, [Assignment("high", pool["high"][0]) for _ in task.steps],
        target_step=2, generation_parameters={}, locked_execution=locked)
    assert result.responses[0] == frozen_response
    assert result.prefixes[:2] == locked.prefixes[:2]


def test_plugin_cannot_mutate_the_lock_to_hide_prefix_drift(tmp_path: Path) -> None:
    _, pool = load_model_pool()

    class MutatingBackend(MockBackend):
        def run_mixed(self, task: Any, assignments: Any, *, target_step: int,
                      generation_parameters: Any, locked_execution: Any = None) -> ExecutionResult:
            if target_step > 1:
                locked_execution.prefixes[target_step - 1][0]["content"] = "mutated lock"
            return super().run_mixed(task, assignments, target_step=target_step,
                generation_parameters=generation_parameters, locked_execution=locked_execution)

    run = tmp_path / "mutating"
    GenerationPipeline(output_dir=run, backend=MutatingBackend(pool)).generate("swebench")
    trials = [row for row in _rows(run / "execution_trials.jsonl") if row["target_step"] == 2]
    assert trials and all(not row["accepted"] for row in trials)
    assert all(not row["locked_prefix_matches"] for row in trials)
    assert not _rows(run / "labels.pre_review.jsonl")


def test_incomplete_trajectory_is_not_accepted(tmp_path: Path) -> None:
    _, pool = load_model_pool()

    class IncompleteBackend(MockBackend):
        def run_mixed(self, task: Any, assignments: Any, *, target_step: int,
                      generation_parameters: Any, locked_execution: Any = None) -> ExecutionResult:
            result = super().run_mixed(task, assignments, target_step=target_step,
                generation_parameters=generation_parameters, locked_execution=locked_execution)
            return replace(result, passed=True, responses=result.responses[:-1])

    run = tmp_path / "incomplete"
    GenerationPipeline(output_dir=run, backend=IncompleteBackend(pool)).generate("swebench")
    trials = _rows(run / "execution_trials.jsonl")
    assert any(row["passed"] for row in trials)
    assert all(not row["accepted"] and not row["complete_trace"] for row in trials)
    assert not _rows(run / "labels.pre_review.jsonl")


def test_replay_refuses_changed_generation_parameters(tmp_path: Path) -> None:
    _, pool = load_model_pool()
    original = tmp_path / "original"
    GenerationPipeline(output_dir=original, backend=MockBackend(pool)).generate("qmsum")
    with pytest.raises(KeyError, match="no seed event"):
        GenerationPipeline(output_dir=tmp_path / "changed", backend=ReplayBackend(original / "backend_events.jsonl"),
            config=GenerationConfig(generation_parameters={"temperature": 0.8})).generate("qmsum")
