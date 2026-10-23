from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from twinrouterbench.data_generation.budget import ApiBudget, BudgetExceeded, request_reservation


def test_unsupported_budget_platform_fails_without_creating_ledger(monkeypatch, tmp_path):
    import twinrouterbench.data_generation.budget as module

    monkeypatch.setattr(module, "fcntl", None)
    path = tmp_path / "cost.jsonl"
    with pytest.raises(RuntimeError, match="POSIX file locking"):
        ApiBudget(path, 1)
    assert not path.exists()


def test_cap_is_shared_across_concurrent_clients(tmp_path):
    path = tmp_path / "cost.jsonl"

    def attempt(_):
        try:
            return ApiBudget(path, 1).reserve("model", 0.25)
        except BudgetExceeded:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        requests = list(pool.map(attempt, range(12)))
    assert sum(value is not None for value in requests) == 4
    assert ApiBudget(path, 1).summary()["accounted_usd"] == 1


def test_missing_usage_does_not_refund_request(tmp_path):
    budget = ApiBudget(tmp_path / "cost.jsonl", 1)
    request = budget.reserve("model", 0.75)
    with pytest.raises(RuntimeError, match="full reservation retained"):
        budget.finish(request, usage={}, pricing={"prompt": 0.1, "completion": 0.1})
    with pytest.raises(BudgetExceeded):
        budget.reserve("model", 0.3)
    assert budget.summary()["pending_or_failed"] == 1


def test_settlement_records_reported_and_conservative_cost_separately(tmp_path):
    budget = ApiBudget(tmp_path / "cost.jsonl", 1)
    request = budget.reserve("model", 0.9)
    budget.finish(request, usage={"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.04},
                  pricing={"prompt": 0.001, "completion": 0.002}, provider="unit")
    assert budget.summary()["accounted_usd"] == pytest.approx(0.04)
    assert budget.summary()["reported_usd"] == pytest.approx(0.04)
    assert budget.summary()["pending_or_failed"] == 0
    with pytest.raises(RuntimeError, match="already settled"):
        budget.finish(request, usage={"prompt_tokens": 10, "completion_tokens": 2},
                      pricing={"prompt": 0.001, "completion": 0.002})


def test_overrun_is_recorded_and_blocks_followup_calls(tmp_path):
    budget = ApiBudget(tmp_path / "cost.jsonl", 1)
    request = budget.reserve("model", 0.2)
    with pytest.raises(BudgetExceeded):
        budget.finish(request, usage={"prompt_tokens": 100, "completion_tokens": 100},
                      pricing={"prompt": 0.01, "completion": 0.01})
    with pytest.raises(BudgetExceeded):
        budget.reserve("model", 0.01)


@pytest.mark.parametrize("pricing", [
    {"prompt": -1, "completion": 1}, {"prompt": 1, "completion": float("nan")},
    {"prompt": 1, "completion": 1, "currency": "CNY"},
])
def test_bad_prices_fail_before_request(pricing):
    with pytest.raises(ValueError):
        request_reservation({"messages": [{"content": "hello"}], "max_tokens": 100}, pricing)


def test_unbounded_output_and_images_need_explicit_pricing_bounds():
    for body in [{"messages": []}, {"messages": [{"image_url": "x"}], "max_tokens": 10}]:
        with pytest.raises(ValueError):
            request_reservation(body, {"prompt": 0.001, "completion": 0.001})
