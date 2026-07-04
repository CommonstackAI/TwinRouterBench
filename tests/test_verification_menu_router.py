from __future__ import annotations

import json

from swerouter.router import CacheStateSnapshot, RouterContext, RunConfig
from swerouter.routers.verification_menu import (
    VerificationMenuRouter,
    score_prefix,
)


MODEL_BY_TIER = {
    "low": "deepseek/deepseek-v3.2",
    "mid": "minimax/minimax-m2.7",
    "mid_high": "google/gemini-3-flash-preview",
    "high": "anthropic/claude-opus-4.6",
}


def _ctx(messages, *, step_index=1, tools=()):
    return RouterContext(
        instance_id="demo__case-1",
        step_index=step_index,
        messages=tuple(messages),
        tools=tuple(tools),
        available_models=tuple(MODEL_BY_TIER.values()),
        cache_state=CacheStateSnapshot(wallclock_ttl_sec=0, now_ts=0.0, by_model={}),
        budget_so_far_usd=0.0,
        run_config=RunConfig(
            max_steps=20,
            budget_usd=3.0,
            wallclock_ttl_sec=0,
            select_timeout_sec=30.0,
        ),
    )


def test_score_prefix_keeps_read_only_steps_cheap():
    tier, menu, features = score_prefix(
        messages=[
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "Fix the issue and submit a patch."},
            {"role": "assistant", "content": "I will inspect files.\nrg model_to_dict ."},
            {"role": "tool", "content": "django/forms/models.py:def model_to_dict(...)"},
        ],
        step_index=1,
        max_steps=20,
    )

    assert tier == "low"
    assert menu == "cheap_execute"
    assert features["read"] is True
    assert features["write"] is False


def test_score_prefix_escalates_patch_after_failure():
    tier, menu, features = score_prefix(
        messages=[
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "Fix the issue."},
            {"role": "assistant", "content": "apply_patch <<'PATCH'\n*** Begin Patch"},
            {"role": "tool", "content": "pytest tests/test_models.py\nFAILED AssertionError"},
        ],
        step_index=5,
        max_steps=20,
    )

    assert tier in {"mid_high", "high"}
    assert menu in {"verify_sensitive", "escalate_frontier"}
    assert features["write"] is True
    assert features["fail"] is True


def test_score_prefix_final_failure_goes_high():
    tier, menu, _ = score_prefix(
        messages=[
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "Fix the issue."},
            {"role": "assistant", "content": "run tests before final submit"},
            {"role": "tool", "content": "Traceback\nAssertionError\nFAILED"},
        ],
        step_index=9,
        max_steps=10,
    )

    assert tier == "high"
    assert menu == "escalate_frontier"


def test_score_prefix_budget_pressure_caps_non_final_high():
    tier, menu, features = score_prefix(
        messages=[
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "<pr_description>Fix the Django bug.</pr_description>"},
            {"role": "assistant", "content": "I will inspect compiler.py"},
            {"role": "tool", "content": "/testbed/django/db/models/sql/compiler.py\nFAILED AssertionError"},
            {"role": "assistant", "content": "grep -n test_ordering tests/queries/test_qs_combinators.py"},
        ],
        step_index=60,
        max_steps=80,
        budget_so_far_usd=1.3,
        budget_usd=1.5,
    )

    assert tier == "mid_high"
    assert menu == "verify_sensitive"
    assert features["budget_ratio"] > 0.8
    assert "budget_pressure_ceiling" in str(features["hard_floor_reason"])


def test_router_maps_tier_to_available_model_and_preserves_messages(tmp_path):
    tier_map = tmp_path / "tier_to_model.json"
    tier_map.write_text(json.dumps({"map": MODEL_BY_TIER}), encoding="utf-8")
    router = VerificationMenuRouter(tier_to_model_path=tier_map)
    messages = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "Use this tool schema."},
    ]
    original = tuple(dict(m) for m in messages)

    decision = router.select(
        _ctx(
            messages,
            tools=(
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "parameters": {"type": "object", "required": ["query"]},
                    },
                },
            ),
        )
    )

    assert decision.model_id == MODEL_BY_TIER["mid"]
    assert decision.rationale is not None
    assert "menu=standard_execute" in decision.rationale
    assert "features=" in decision.rationale
    assert '"hard_floor_reason": "schema"' in decision.rationale
    assert '"risk_profile": "balanced"' in decision.rationale
    assert tuple(messages) == original


def test_router_risk_profile_can_be_configured(tmp_path):
    tier_map = tmp_path / "tier_to_model.json"
    tier_map.write_text(json.dumps({"map": MODEL_BY_TIER}), encoding="utf-8")
    router = VerificationMenuRouter(
        tier_to_model_path=tier_map,
        risk_profile="conservative",
    )

    decision = router.select(
        _ctx(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Delete all files from Drafts."},
            ],
            tools=(
                {
                    "name": "delete_message",
                    "parameters": {"type": "object", "required": ["message_id"]},
                },
                {"name": "list_messages", "parameters": {"type": "object"}},
            ),
        )
    )

    assert decision.model_id == MODEL_BY_TIER["mid_high"]
    assert decision.rationale is not None
    assert '"risk_profile": "conservative"' in decision.rationale
    assert "destructive_schema" in decision.rationale
