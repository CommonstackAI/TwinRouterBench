import json
from pathlib import Path

import pytest

from swerouter.leaderboard.score import score_run_dir
from swerouter.pricing import load_pricing_table, step_real_cost_usd
from swerouter.router import FunctionRouter, RouterDecision
from swerouter.usage import normalize_usage
from twinrouterbench.dynamic_adapter import ModelCallResult, RoutedSession, SessionLimits
from twinrouterbench.dynamic_adapter.session import DEFAULT_PRICING

LIMITS = SessionLimits(max_steps=10, budget_usd=5.0)


def _usage(prompt, completion, cached=0):
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "prompt_tokens_details": {"cached_tokens": cached},
    }


class FakeHarnessModel:
    """Stands in for a benchmark's own LLM client."""

    def __init__(self, usages):
        self.usages = list(usages)
        self.calls = []

    def __call__(self, model_id, messages, tools):
        self.calls.append((model_id, messages, tools))
        messages.append({"role": "user", "content": "mutated by callee"})
        return ModelCallResult(
            message={"role": "assistant", "content": f"reply from {model_id}"},
            raw_usage=self.usages.pop(0),
        )


def _cycle_router(seen):
    def select(ctx):
        seen.append(ctx)
        return ctx.available_models[ctx.step_index % len(ctx.available_models)]

    return FunctionRouter(label="cycle", func=select)


def _session(tmp_path, router, model, instance_id="bench__case-1"):
    return RoutedSession(
        router=router,
        instance_id=instance_id,
        run_dir=tmp_path,
        call_model=model,
        limits=LIMITS,
    )


def test_routed_run_is_scored_by_existing_leaderboard(tmp_path):
    seen = []
    usages = [_usage(1000, 50), _usage(1200, 40, cached=900), _usage(1500, 30)]
    model = FakeHarnessModel(usages)
    session = _session(tmp_path, _cycle_router(seen), model)
    tools = ({"type": "function", "function": {"name": "shell"}},)

    history = [{"role": "system", "content": "s"}, {"role": "user", "content": "task"}]
    for _ in range(3):
        reply = session.call(history, tools)
        history = history + [dict(reply), {"role": "user", "content": "observation"}]
    session.finish(resolved=True, finished_by="submitted")

    pricing = load_pricing_table(DEFAULT_PRICING)
    expected = sum(
        step_real_cost_usd(normalize_usage("openai_compat", u), pricing.get(m))
        for u, (m, _, _) in zip(
            [_usage(1000, 50), _usage(1200, 40, cached=900), _usage(1500, 30)],
            model.calls,
        )
    )
    report = score_run_dir(run_dir=tmp_path, router_label="cycle")
    assert report["resolved_count"] == 1
    assert report["total_router_cost_usd"] == pytest.approx(expected)
    assert report["total_leaderboard_bill_usd"] == pytest.approx(expected)

    assert [ctx.step_index for ctx in seen] == [0, 1, 2]
    assert seen[0].tools == tools
    assert len(seen[2].messages) == 6
    assert [m for m, _, _ in model.calls] == list(seen[0].available_models[:3])
    assert all(msg["content"] != "mutated by callee" for msg in seen[2].messages)


def test_unresolved_instance_pays_penalty(tmp_path):
    session = _session(tmp_path, _cycle_router([]), FakeHarnessModel([_usage(100, 10)]))
    session.call([{"role": "user", "content": "task"}])
    session.finish(resolved=False, finished_by="step_limit")

    report = score_run_dir(
        run_dir=tmp_path, router_label="cycle", fixed_failure_penalty_usd=0.25
    )
    assert report["resolved_count"] == 0
    assert report["total_penalty_cost_usd"] == pytest.approx(0.25)


def test_cache_lookup_hits_on_same_model_prefix_extension(tmp_path):
    router = FunctionRouter(label="first", func=lambda ctx: ctx.available_models[0])
    session = _session(
        tmp_path, router, FakeHarnessModel([_usage(100, 10), _usage(200, 10)])
    )
    history = [{"role": "user", "content": "task"}]
    reply = session.call(history)
    session.call(history + [dict(reply), {"role": "user", "content": "obs"}])
    rows = [json.loads(line) for line in session.trace_path.read_text().splitlines()]
    assert rows[0]["cache_lookup"]["reason"] == "cold_start_first_call"
    assert rows[1]["cache_lookup"]["hit"] is True


def test_out_of_pool_decision_fails_before_model_call(tmp_path):
    model = FakeHarnessModel([_usage(100, 10)])
    router = FunctionRouter(
        label="bad", func=lambda ctx: RouterDecision(model_id="vendor/not-in-pool")
    )
    session = _session(tmp_path, router, model)
    with pytest.raises(ValueError, match="not in the official"):
        session.call([{"role": "user", "content": "task"}])
    assert model.calls == []
    assert not session.trace_path.exists() or session.trace_path.read_text() == ""


def test_missing_usage_is_an_error_not_zero_cost(tmp_path):
    session = _session(tmp_path, _cycle_router([]), FakeHarnessModel([{}]))
    with pytest.raises(ValueError, match="prompt_tokens"):
        session.call([{"role": "user", "content": "task"}])


def test_finish_requires_explicit_boolean_verdict(tmp_path):
    session = _session(tmp_path, _cycle_router([]), FakeHarnessModel([]))
    with pytest.raises(TypeError):
        session.finish(resolved=1, finished_by="submitted")


def test_agent_error_keeps_original_message(tmp_path):
    session = _session(tmp_path, _cycle_router([]), FakeHarnessModel([]))
    path = session.finish(
        resolved=False,
        finished_by="agent_error",
        agent_error="HTTPError: HTTP 429 from provider",
    )
    blob = json.loads(Path(path).read_text())
    assert blob["agent_error"] == "HTTPError: HTTP 429 from provider"
    report = score_run_dir(
        run_dir=tmp_path, router_label="cycle", exclude_infra_failures=True
    )
    assert report["infra_excluded_count"] == 1
