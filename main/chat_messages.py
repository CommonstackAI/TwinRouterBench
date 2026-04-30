"""Normalize agent trajectories to OpenAI-style chat messages (string content, tool turns folded)."""

from __future__ import annotations

import json
from typing import Any


def linearize_messages_for_openai_compat(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Build chat payloads for OpenAI-style HTTP APIs.

    Gateways often reject native ``tool`` role messages; tool outputs are folded into ``user``
    lines so every message is ``system`` / ``user`` / ``assistant`` with string ``content``.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            raise ValueError(f"Unsupported message role: {role!r}")
        content = m.get("content")
        if content is None:
            content = ""
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        if role == "assistant":
            if m.get("tool_calls"):
                extra = json.dumps(m["tool_calls"], ensure_ascii=False)
                if len(extra) > 12000:
                    extra = extra[:12000] + "...(truncated)"
                content = f"{content}\n[tool_calls JSON]\n{extra}"
            out.append({"role": "assistant", "content": content})
            continue
        if role == "tool":
            tid = m.get("tool_call_id")
            if not tid:
                raise ValueError("tool message missing tool_call_id")
            folded = f"[tool result tool_call_id={tid}]\n{content}"
            out.append({"role": "user", "content": folded})
            continue
        out.append({"role": role, "content": content})
    return out


def format_linear_messages_as_user_prompt(linear: list[dict[str, Any]]) -> str:
    """Turn linearized messages into one user-visible transcript for a classifier prompt."""
    blocks: list[str] = []
    for m in linear:
        role = m["role"]
        content = m["content"]
        if not isinstance(content, str):
            raise ValueError("linearized message content must be str")
        blocks.append(f"### {role}\n{content}")
    return "\n\n".join(blocks)


def question_bank_messages_to_classifier_prompt(
    messages: list[dict[str, Any]],
    functions: list[dict[str, Any]] | None = None,
) -> str:
    """Convert a question_bank row ``messages`` and optional ``functions`` into a single user prompt string."""
    linear = linearize_messages_for_openai_compat(messages)
    prompt = format_linear_messages_as_user_prompt(linear)
    if functions:
        tool_json = json.dumps(functions, indent=2, ensure_ascii=False)
        prompt = f"### available_tools\n{tool_json}\n\n{prompt}"
    return prompt
