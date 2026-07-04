"""Verification-aware service-menu router baseline.

This router maps router-visible prefixes to a deterministic service-menu
decision, then maps the selected public tier to an official dynamic-track
``model_id`` through ``data/dynamic/tier_to_model.json``.

It does not issue extra verifier calls. Verification awareness is encoded as
local risk features over the current prefix: write/edit intent, failing tests
or tracebacks, strict tool/schema surfaces, long context, late trajectory
steps, and budget pressure.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from main.tiers import PUBLIC_TIERS, TIER_TO_ID

from swerouter.router import RouterContext, RouterDecision

SERVICE_MENU_BY_TIER: dict[str, str] = {
    "low": "cheap_execute",
    "mid": "standard_execute",
    "mid_high": "verify_sensitive",
    "high": "escalate_frontier",
}

READ_PAT = re.compile(
    r"\b(read|grep|rg|ls|dir|cat|type|find|search|inspect|open file|view)\b",
    re.IGNORECASE,
)
WRITE_PAT = re.compile(
    r"(\bapply_patch\b|\bstr_replace\b|\bgit diff\b|\bdiff --git\b|"
    r"\bpatch\b|\bedit\b|\bwrite\b|\bdelete\b|\brename\b|\bsubmit\b|"
    r"\bsed\s+-i\b|\brm\s+-|\bmv\s+|\bcp\s+|"
    r"\b(?:cat|tee|python|printf|echo)\b[^\n]{0,120}>\s*[A-Za-z0-9_./\\-]+)",
    re.IGNORECASE,
)
FAIL_PAT = re.compile(
    r"(\bFAIL(?:ED|URES?)?\b|\bAssertionError\b|\bTraceback\b|"
    r"\bException\b|\bpytest\b|\bmismatch\b|\breturncode>\s*[1-9]\b|"
    r"\berror:|\bfailed\b)",
    re.IGNORECASE,
)
SCHEMA_PAT = re.compile(
    r"(\bfunction(?:s|_call)?\b|\btool_call\b|\barguments\b|"
    r"\bjson schema\b|\bparameters\b|\brequired\b)",
    re.IGNORECASE,
)
FINAL_PAT = re.compile(
    r"(\bfinal\b|\bsubmit\b|COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT|"
    r"\bpatch is ready\b|\brun tests?\b|\banswer now\b)",
    re.IGNORECASE,
)
CODE_PATH_PAT = re.compile(
    r"(/testbed/|(?:^|[\s\"'`])[\w./\\-]+\.(?:py|js|ts|java|go|rs|cpp|c|h)\b|"
    r"\bdiff --git\b|\bupdated [\w./\\-]+\b)",
    re.IGNORECASE,
)
DESTRUCTIVE_REQUEST_PAT = re.compile(
    r"\b(delete|remove|rename|archive|send|post|comment|follow|order|"
    r"transaction|authenticate|login|write|modify|update|create)\b",
    re.IGNORECASE,
)
RISK_PROFILES = ("cheap", "balanced", "conservative")


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, Mapping):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
                else:
                    parts.append(json.dumps(block, ensure_ascii=False, sort_keys=True))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    if isinstance(content, Mapping):
        return json.dumps(content, ensure_ascii=False, sort_keys=True)
    return str(content)


def _message_to_text(message: Mapping[str, Any]) -> str:
    chunks = [_content_to_text(message.get("content"))]
    tool_calls = message.get("tool_calls")
    if tool_calls:
        chunks.append(json.dumps(tool_calls, ensure_ascii=False, sort_keys=True))
    function_call = message.get("function_call")
    if function_call:
        chunks.append(json.dumps(function_call, ensure_ascii=False, sort_keys=True))
    return "\n".join(chunk for chunk in chunks if chunk)


def flatten_messages(messages: Sequence[Mapping[str, Any]]) -> str:
    """Flatten OpenAI-style messages into the router-prompt text shape."""

    blocks: list[str] = []
    for msg in messages:
        role = str(msg.get("role") or "user")
        text = _message_to_text(msg)
        blocks.append(f"{role}:\n{text}")
    return "\n\n".join(blocks)


def _tools_to_text(tools: Sequence[Mapping[str, Any]] | Sequence[Any]) -> str:
    if not tools:
        return ""
    return json.dumps(list(tools), ensure_ascii=False, sort_keys=True)


def _history_messages(messages: Sequence[Mapping[str, Any]]) -> Sequence[Mapping[str, Any]]:
    """Return non-initial agent history.

    Agent task prompts often contain words such as "patch" and "submit" before
    any action has happened. For write/final risk we therefore focus on the
    assistant/tool history after the initial system+user task prompt.
    """

    if len(messages) <= 2:
        return ()
    return messages[2:]


def _non_user_text(messages: Sequence[Mapping[str, Any]]) -> str:
    blocks: list[str] = []
    for msg in messages:
        role = str(msg.get("role") or "")
        if role in {"system", "user"}:
            continue
        blocks.append(_message_to_text(msg))
    return "\n\n".join(blocks)


def _tier_at_least(tier: str, floor: str) -> str:
    return tier if TIER_TO_ID[tier] >= TIER_TO_ID[floor] else floor


def _tier_at_most(tier: str, ceiling: str) -> str:
    return tier if TIER_TO_ID[tier] <= TIER_TO_ID[ceiling] else ceiling


def _tier_from_risk(risk: float) -> str:
    if risk <= 20:
        return "low"
    if risk <= 42:
        return "mid"
    if risk <= 65:
        return "mid_high"
    return "high"


def _normalize_risk_profile(risk_profile: str) -> str:
    normalized = risk_profile.strip().lower()
    if normalized not in RISK_PROFILES:
        raise ValueError(
            f"unknown risk_profile={risk_profile!r}; expected one of {RISK_PROFILES}"
        )
    return normalized


def _profile_delta(risk_profile: str) -> float:
    if risk_profile == "cheap":
        return -6.0
    if risk_profile == "conservative":
        return 6.0
    return 0.0


def score_prefix(
    *,
    messages: Sequence[Mapping[str, Any]],
    step_index: int,
    max_steps: int | None = None,
    budget_so_far_usd: float = 0.0,
    budget_usd: float | None = None,
    tools: Sequence[Mapping[str, Any]] | Sequence[Any] = (),
    risk_profile: str = "balanced",
) -> tuple[str, str, dict[str, float | int | str | bool]]:
    """Score a router prefix and return ``(tier, service_menu_item, features)``.

    The function is deterministic and dependency-free so it can be reused by
    the static-track helper script and by unit tests.
    """

    profile = _normalize_risk_profile(risk_profile)
    safe_step = max(int(step_index), 0)
    full_text = flatten_messages(messages)
    initial_text = flatten_messages(messages[:2])
    latest_text = _message_to_text(messages[-1]) if messages else ""
    history = _history_messages(messages)
    recent_history = history[-4:] if len(history) > 4 else history
    history_text = flatten_messages(history)
    recent_action_text = _non_user_text(recent_history)
    tools_text = _tools_to_text(tools)

    approx_tokens = len(full_text) / 4.0
    tool_count = len(tools)
    tool_schema_chars = len(tools_text)
    fail_count = len(FAIL_PAT.findall(history_text))
    write = bool(WRITE_PAT.search(recent_action_text))
    read = bool(READ_PAT.search(recent_action_text or latest_text))
    fail = fail_count > 0
    schema = bool(tools) or bool(SCHEMA_PAT.search(tools_text))
    if not schema:
        schema = bool(SCHEMA_PAT.search(full_text)) and (
            "tool_calls" in full_text or "functions" in full_text
        )
    final = bool(FINAL_PAT.search(recent_action_text))
    code_repair = "<pr_description>" in initial_text or "/testbed" in initial_text
    code_file_context = bool(CODE_PATH_PAT.search(history_text))
    destructive_request = bool(DESTRUCTIVE_REQUEST_PAT.search(latest_text))

    risk = _profile_delta(profile)
    if approx_tokens > 32_000:
        risk += 20
    elif approx_tokens > 16_000:
        risk += 14
    elif approx_tokens > 8_000:
        risk += 8
    elif approx_tokens > 3_000:
        risk += 4

    if schema:
        risk += 8
        if tool_count >= 8 or tool_schema_chars > 6_000:
            risk += 6
        elif tool_count >= 3 or tool_schema_chars > 2_000:
            risk += 4
        if destructive_request:
            risk += 8
    if code_repair:
        risk += 10 if safe_step == 0 else 18
    if code_file_context:
        risk += 10
    if write:
        risk += 18
    if fail:
        risk += 18 + min(12, fail_count * 3)
    if final:
        risk += 16
    if safe_step >= 15:
        risk += 8
    elif safe_step >= 8:
        risk += 5
    if max_steps and max_steps > 0 and safe_step / max_steps > 0.70:
        risk += 8

    # Simple read-only inspection steps are valuable places to save cost.
    if read and not write and not fail and not final:
        risk -= 10

    budget_ratio = 0.0
    if budget_usd and budget_usd > 0:
        budget_ratio = min(max(float(budget_so_far_usd) / float(budget_usd), 0.0), 1.0)
        if budget_ratio > 0.90 and not (write or fail):
            risk -= 8
        elif budget_ratio > 0.70 and not (write or fail or final):
            risk -= 6

    tier = _tier_from_risk(risk)
    hard_floor_reasons: list[str] = []

    # Hard floors: cost pressure should not undercut structurally risky steps.
    if schema:
        before = tier
        tier = _tier_at_least(tier, "mid")
        if tier != before:
            hard_floor_reasons.append("schema")
    if schema and destructive_request:
        tier = _tier_at_least(tier, "mid_high")
        hard_floor_reasons.append("destructive_schema")
    if profile == "conservative" and schema and (tool_count >= 8 or tool_schema_chars > 6_000):
        before = tier
        tier = _tier_at_least(tier, "mid_high")
        if tier != before:
            hard_floor_reasons.append("conservative_complex_schema")
    if code_repair and safe_step == 0:
        before = tier
        tier = _tier_at_least(tier, "mid_high")
        if tier != before:
            hard_floor_reasons.append("code_repair_initial")
    if code_repair and safe_step >= 1:
        before = tier
        tier = _tier_at_least(tier, "mid_high")
        if tier != before:
            hard_floor_reasons.append("code_repair_history")
        if code_file_context or write or fail or safe_step >= 2:
            tier = "high"
            hard_floor_reasons.append("code_repair_critical")
    if write or fail:
        before = tier
        tier = _tier_at_least(tier, "mid_high")
        if tier != before:
            hard_floor_reasons.append("write_or_failure")
    if (final and (write or fail)) or (write and fail and approx_tokens > 3_000):
        tier = "high"
        hard_floor_reasons.append("final_or_failed_write")

    # Once a run is close to its cost cap, high-tier calls are reserved for
    # actual final/submit decisions. This keeps long debugging loops from
    # spending the last budget slice on repeated file inspection.
    if budget_ratio >= 0.80 and not final:
        before = tier
        tier = _tier_at_most(tier, "mid_high")
        if tier != before:
            hard_floor_reasons.append("budget_pressure_ceiling")
    elif budget_ratio >= 0.55 and read and not write and safe_step >= 20 and not final:
        before = tier
        tier = _tier_at_most(tier, "mid_high")
        if tier != before:
            hard_floor_reasons.append("read_loop_budget_ceiling")

    menu = SERVICE_MENU_BY_TIER[tier]
    features: dict[str, float | int | str | bool] = {
        "risk": round(risk, 2),
        "risk_profile": profile,
        "approx_tokens": int(approx_tokens),
        "tool_count": tool_count,
        "tool_schema_chars": tool_schema_chars,
        "write": write,
        "read": read,
        "fail": fail,
        "fail_count": fail_count,
        "schema": schema,
        "destructive_request": destructive_request,
        "code_repair": code_repair,
        "code_file_context": code_file_context,
        "final": final,
        "step_index": safe_step,
        "budget_ratio": round(budget_ratio, 3),
        "tier": tier,
        "menu": menu,
        "hard_floor_reason": "|".join(hard_floor_reasons) or "none",
    }
    return tier, menu, features


@dataclass
class VerificationMenuRouter:
    """Deterministic verification-aware service-menu router."""

    tier_to_model_path: Path
    label: str = "verification_menu"
    risk_profile: str = "balanced"

    def __post_init__(self) -> None:
        self.risk_profile = _normalize_risk_profile(self.risk_profile)
        path = Path(self.tier_to_model_path)
        if not path.is_file():
            raise FileNotFoundError(f"tier_to_model not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            doc = json.load(fh)
        raw = doc.get("map") if isinstance(doc, dict) else None
        if not isinstance(raw, dict):
            raise ValueError(f"{path} missing .map object")
        missing = [tier for tier in PUBLIC_TIERS if tier not in raw]
        if missing:
            raise ValueError(f"tier_to_model.map missing tiers: {missing}")

        tier_to_model: dict[str, str] = {}
        for tier in PUBLIC_TIERS:
            model_id = raw[tier]
            if not isinstance(model_id, str) or not model_id:
                raise ValueError(f"tier_to_model.map[{tier!r}] must be a non-empty string")
            tier_to_model[tier] = model_id
        self._tier_to_model = tier_to_model

    def select(self, ctx: RouterContext) -> RouterDecision:
        tier, menu, features = score_prefix(
            messages=ctx.messages,
            tools=ctx.tools,
            step_index=ctx.step_index,
            max_steps=ctx.run_config.max_steps,
            budget_so_far_usd=ctx.budget_so_far_usd,
            budget_usd=ctx.run_config.budget_usd,
            risk_profile=self.risk_profile,
        )
        model_id = self._tier_to_model[tier]
        if model_id not in ctx.available_models:
            raise ValueError(
                f"VerificationMenuRouter[{self.label!r}]: tier={tier!r} maps to "
                f"{model_id!r}, not in available pool {list(ctx.available_models)}"
            )
        rationale = (
            f"{self.label} menu={menu} tier={tier} "
            f"features={json.dumps(features, sort_keys=True)}"
        )
        return RouterDecision(model_id=model_id, rationale=rationale)

    @classmethod
    def from_cli_args(
        cls,
        *,
        tier_to_model_path: str,
        label: str = "verification_menu",
        risk_profile: str = "balanced",
    ) -> "VerificationMenuRouter":
        """CLI-friendly factory; ``--router-arg`` values arrive as strings."""

        return cls(
            tier_to_model_path=Path(tier_to_model_path),
            label=label,
            risk_profile=risk_profile,
        )
