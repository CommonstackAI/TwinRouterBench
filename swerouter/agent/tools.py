"""Tool definitions for the SWERouterBench agent loop.

Each tool ships with two things:

* An OpenAI-compatible ``tool`` JSON schema exposed to the model.
* A pure Python callable that executes the tool inside a running Docker
  container via ``docker exec``.

The intent is to match the tool surface used by SWE-bench reference scaffolds
(bash, Anthropic-style ``str_replace_editor`` subcommands, ``finish``) closely
enough that a model trained on SWE-bench traces can drive the loop without
retraining.

File edit subcommands (``view`` / ``create`` / ``str_replace`` / ``insert`` /
``undo_edit``) operate on files inside the container by streaming content over
``exec_run`` using base64 encoding to avoid shell-escaping issues.
"""

from __future__ import annotations

import base64
import json
import os
import shlex
from dataclasses import dataclass
from typing import Any, Callable, Mapping

# Docker SDK is imported lazily so unit tests that don't touch a container can
# still import this module.
try:  # pragma: no cover - import guard
    from docker.models.containers import Container  # type: ignore
except Exception:  # pragma: no cover - import guard
    Container = Any  # type: ignore[assignment,misc]


FINISH_SENTINEL = "__SWEROUTER_FINISH__"


def _safe_env_int(name: str, default: int) -> int:
    """Parse a non-negative int from ``os.environ``; bad or empty values fall back."""
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        v = int(str(raw).strip(), 10)
    except ValueError:
        return default
    return v if v > 0 else 0


# Wall-clock caps for ``docker exec`` (the SDK has no reliable per-exec timeout).
# Override with env vars if your tasks need longer shells. Zero disables wrapping
# for that category (not recommended for ``bash``).
_BASH_TOOL_TIMEOUT_SEC = _safe_env_int("SWEROUTER_BASH_TOOL_TIMEOUT_SEC", 1800)
_TOOL_SHORT_TIMEOUT_SEC = _safe_env_int("SWEROUTER_TOOL_SHORT_TIMEOUT_SEC", 300)
_TOOL_FILE_IO_TIMEOUT_SEC = _safe_env_int("SWEROUTER_TOOL_FILE_IO_TIMEOUT_SEC", 900)


@dataclass(frozen=True)
class ToolResult:
    """Uniform return type for every tool.

    ``content`` is the string appended to the conversation as a ``role=tool``
    message. ``ok`` is True iff the tool considered the operation successful;
    the agent loop still passes the content to the model either way so the
    model can learn from errors.
    """

    tool_name: str
    content: str
    ok: bool
    metadata: Mapping[str, Any]


ToolExecutor = Callable[["Container", Mapping[str, Any]], ToolResult]


def _exec_in_container(
    container: "Container",
    cmd: list[str],
    *,
    workdir: str | None = None,
    timeout_sec: int | None = None,
) -> tuple[int, str]:
    """Run ``cmd`` in the container and return ``(exit_code, combined_output)``.

    When ``timeout_sec`` is set, the inner command is wrapped with GNU
    ``timeout(1)`` inside the container so a hung model command (e.g. an
    unbounded test run) cannot block the host worker thread forever.

    ``timeout_sec`` is only supported when ``cmd`` is exactly
    ``["/bin/bash", "-lc", "<script>"]`` (all current call sites). Passing a
    positive timeout with any other shape raises :class:`ValueError` so the
    cap cannot be mistaken as applied when it is not.
    """
    if timeout_sec is not None and int(timeout_sec) > 0:
        t = int(timeout_sec)
        if len(cmd) == 3 and cmd[0] == "/bin/bash" and cmd[1] == "-lc":
            inner = cmd[2]
            wrapped = f"timeout -k 10 {t} bash -lc {shlex.quote(inner)}"
            cmd = ["/bin/bash", "-lc", wrapped]
            timeout_sec = None
        else:
            preview = f"{cmd[0]!r} {cmd[1]!r} ..." if len(cmd) >= 2 else repr(cmd)
            raise ValueError(
                "_exec_in_container: timeout_sec requires "
                "cmd == ['/bin/bash', '-lc', <script>], got "
                f"{preview}"
            )

    exec_kwargs: dict[str, Any] = {
        "cmd": cmd,
        "demux": False,
        "tty": False,
        "stream": False,
    }
    if workdir is not None:
        exec_kwargs["workdir"] = workdir
    result = container.exec_run(**exec_kwargs)
    exit_code = int(result.exit_code) if result.exit_code is not None else -1
    raw = result.output
    if isinstance(raw, bytes):
        output = raw.decode("utf-8", errors="replace")
    elif raw is None:
        output = ""
    else:
        output = str(raw)
    return exit_code, output


# ---------------------------------------------------------------------------
# bash
# ---------------------------------------------------------------------------

BASH_SCHEMA: Mapping[str, Any] = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": (
            "Run a bash command inside the SWE-bench container. The command is "
            "executed with /bin/bash -lc. Use this to inspect the repo, run "
            "tests, or apply scripted edits. Output is captured and returned."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The full shell command to run.",
                },
                "workdir": {
                    "type": "string",
                    "description": "Absolute path working directory for the command (default /testbed).",
                    "default": "/testbed",
                },
            },
            "required": ["command"],
        },
    },
}


def tool_bash(container: "Container", args: Mapping[str, Any]) -> ToolResult:
    command = args.get("command")
    if not isinstance(command, str) or not command:
        raise ValueError("bash.command must be a non-empty string")
    workdir = args.get("workdir", "/testbed")
    if not isinstance(workdir, str):
        raise TypeError("bash.workdir must be a string")
    exit_code, output = _exec_in_container(
        container,
        ["/bin/bash", "-lc", command],
        workdir=workdir,
        timeout_sec=_BASH_TOOL_TIMEOUT_SEC,
    )
    content = f"Exit code: {exit_code}\n{output}"
    return ToolResult(
        tool_name="bash",
        content=content,
        ok=exit_code == 0,
        metadata={"exit_code": exit_code, "workdir": workdir},
    )


# ---------------------------------------------------------------------------
# str_replace_editor: Anthropic-style file editing subcommands
# ---------------------------------------------------------------------------

EDITOR_SCHEMA: Mapping[str, Any] = {
    "type": "function",
    "function": {
        "name": "str_replace_editor",
        "description": (
            "Anthropic-style file editor. Subcommands: view, create, "
            "str_replace, insert, undo_edit. Operates on files inside the "
            "SWE-bench container."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["view", "create", "str_replace", "insert", "undo_edit"],
                },
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file or directory.",
                },
                "file_text": {
                    "type": "string",
                    "description": "Contents for 'create'.",
                },
                "old_str": {
                    "type": "string",
                    "description": "Exact literal substring to replace (for 'str_replace').",
                },
                "new_str": {
                    "type": "string",
                    "description": "Replacement string (for 'str_replace').",
                },
                "insert_line": {
                    "type": "integer",
                    "description": "1-based line number AFTER which to insert (for 'insert').",
                },
                "new_line": {
                    "type": "string",
                    "description": "Line content for 'insert' (trailing newline optional).",
                },
                "view_range": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Optional [start, end] 1-based inclusive line range for 'view'.",
                },
            },
            "required": ["command", "path"],
        },
    },
}


# Per-container, per-path edit history for 'undo_edit'. Simple in-memory
# snapshot: { container_id: { abspath: [prior_contents, ...] } }
_EDIT_HISTORY: dict[str, dict[str, list[str]]] = {}


def _history_for(container: "Container") -> dict[str, list[str]]:
    cid = getattr(container, "id", None) or str(id(container))
    return _EDIT_HISTORY.setdefault(cid, {})


def _push_history(container: "Container", path: str, contents: str) -> None:
    _history_for(container).setdefault(path, []).append(contents)


def _pop_history(container: "Container", path: str) -> str | None:
    bucket = _history_for(container).get(path)
    if not bucket:
        return None
    return bucket.pop()


def _read_file(container: "Container", path: str) -> tuple[bool, str]:
    """Return ``(exists, contents)`` for a file inside the container."""
    rc, out = _exec_in_container(
        container,
        ["/bin/bash", "-lc", f"test -f {shlex.quote(path)} && base64 -w0 {shlex.quote(path)}"],
        timeout_sec=_TOOL_FILE_IO_TIMEOUT_SEC,
    )
    if rc != 0:
        return False, ""
    try:
        return True, base64.b64decode(out.strip()).decode("utf-8", errors="replace")
    except Exception as ex:
        raise RuntimeError(f"failed to decode {path!r} from container: {ex}") from ex


def _write_file(container: "Container", path: str, contents: str) -> None:
    b64 = base64.b64encode(contents.encode("utf-8")).decode("ascii")
    rc, out = _exec_in_container(
        container,
        [
            "/bin/bash",
            "-lc",
            f"mkdir -p $(dirname {shlex.quote(path)}) && "
            f"printf %s {shlex.quote(b64)} | base64 -d > {shlex.quote(path)}",
        ],
        timeout_sec=_TOOL_FILE_IO_TIMEOUT_SEC,
    )
    if rc != 0:
        raise RuntimeError(f"failed to write {path!r}: {out}")


def _view(container: "Container", path: str, view_range: list[int] | None) -> ToolResult:
    rc, out = _exec_in_container(
        container,
        ["/bin/bash", "-lc", f"test -e {shlex.quote(path)}"],
        timeout_sec=_TOOL_SHORT_TIMEOUT_SEC,
    )
    if rc != 0:
        return ToolResult(
            tool_name="str_replace_editor",
            content=f"Path does not exist: {path}",
            ok=False,
            metadata={"command": "view", "path": path},
        )
    # If directory, list; if file, dump with line numbers.
    rc, is_dir = _exec_in_container(
        container,
        ["/bin/bash", "-lc", f"test -d {shlex.quote(path)} && echo yes || echo no"],
        timeout_sec=_TOOL_SHORT_TIMEOUT_SEC,
    )
    if is_dir.strip() == "yes":
        rc, listing = _exec_in_container(
            container,
            ["/bin/bash", "-lc", f"ls -la --time-style=long-iso {shlex.quote(path)}"],
            timeout_sec=_TOOL_SHORT_TIMEOUT_SEC,
        )
        return ToolResult(
            tool_name="str_replace_editor",
            content=f"Directory {path}:\n{listing}",
            ok=True,
            metadata={"command": "view", "path": path, "kind": "dir"},
        )

    ok, contents = _read_file(container, path)
    if not ok:
        return ToolResult(
            tool_name="str_replace_editor",
            content=f"Cannot read {path}",
            ok=False,
            metadata={"command": "view", "path": path},
        )
    lines = contents.splitlines()
    if view_range is not None:
        if len(view_range) != 2 or not all(isinstance(x, int) for x in view_range):
            raise ValueError("view_range must be [start, end] with integer values")
        start, end = view_range
        if start < 1 or end < start or end > len(lines):
            return ToolResult(
                tool_name="str_replace_editor",
                content=f"view_range out of bounds for {path} (length {len(lines)})",
                ok=False,
                metadata={"command": "view", "path": path},
            )
        selected = lines[start - 1 : end]
        offset = start
    else:
        selected = lines
        offset = 1
    numbered = "\n".join(f"{offset + i:6d}\t{line}" for i, line in enumerate(selected))
    return ToolResult(
        tool_name="str_replace_editor",
        content=numbered or f"(empty file {path})",
        ok=True,
        metadata={"command": "view", "path": path, "line_count": len(lines)},
    )


def _create(container: "Container", path: str, file_text: str) -> ToolResult:
    rc, _ = _exec_in_container(
        container,
        ["/bin/bash", "-lc", f"test -e {shlex.quote(path)}"],
        timeout_sec=_TOOL_SHORT_TIMEOUT_SEC,
    )
    if rc == 0:
        return ToolResult(
            tool_name="str_replace_editor",
            content=f"Refusing to overwrite existing path: {path}. Use str_replace or insert instead.",
            ok=False,
            metadata={"command": "create", "path": path},
        )
    _write_file(container, path, file_text)
    return ToolResult(
        tool_name="str_replace_editor",
        content=f"Created {path} ({len(file_text)} bytes).",
        ok=True,
        metadata={"command": "create", "path": path, "bytes": len(file_text)},
    )


def _str_replace(container: "Container", path: str, old_str: str, new_str: str) -> ToolResult:
    ok, contents = _read_file(container, path)
    if not ok:
        return ToolResult(
            tool_name="str_replace_editor",
            content=f"Cannot read {path}",
            ok=False,
            metadata={"command": "str_replace", "path": path},
        )
    occurrences = contents.count(old_str)
    if occurrences == 0:
        return ToolResult(
            tool_name="str_replace_editor",
            content=f"old_str not found in {path}. No replacement performed.",
            ok=False,
            metadata={"command": "str_replace", "path": path, "occurrences": 0},
        )
    if occurrences > 1:
        return ToolResult(
            tool_name="str_replace_editor",
            content=(
                f"old_str matches {occurrences} times in {path}; required exactly 1. "
                "Add surrounding context to make it unique."
            ),
            ok=False,
            metadata={
                "command": "str_replace",
                "path": path,
                "occurrences": occurrences,
            },
        )
    _push_history(container, path, contents)
    new_contents = contents.replace(old_str, new_str, 1)
    _write_file(container, path, new_contents)
    return ToolResult(
        tool_name="str_replace_editor",
        content=f"Replaced 1 occurrence in {path}.",
        ok=True,
        metadata={"command": "str_replace", "path": path},
    )


def _insert(container: "Container", path: str, insert_line: int, new_line: str) -> ToolResult:
    ok, contents = _read_file(container, path)
    if not ok:
        return ToolResult(
            tool_name="str_replace_editor",
            content=f"Cannot read {path}",
            ok=False,
            metadata={"command": "insert", "path": path},
        )
    lines = contents.splitlines(keepends=True)
    if insert_line < 0 or insert_line > len(lines):
        return ToolResult(
            tool_name="str_replace_editor",
            content=f"insert_line {insert_line} out of bounds (0..{len(lines)})",
            ok=False,
            metadata={"command": "insert", "path": path},
        )
    _push_history(container, path, contents)
    inserted = new_line if new_line.endswith("\n") else new_line + "\n"
    new_lines = lines[:insert_line] + [inserted] + lines[insert_line:]
    _write_file(container, path, "".join(new_lines))
    return ToolResult(
        tool_name="str_replace_editor",
        content=f"Inserted 1 line at position {insert_line} in {path}.",
        ok=True,
        metadata={"command": "insert", "path": path},
    )


def _undo_edit(container: "Container", path: str) -> ToolResult:
    prev = _pop_history(container, path)
    if prev is None:
        return ToolResult(
            tool_name="str_replace_editor",
            content=f"No edit history for {path}.",
            ok=False,
            metadata={"command": "undo_edit", "path": path},
        )
    _write_file(container, path, prev)
    return ToolResult(
        tool_name="str_replace_editor",
        content=f"Restored previous contents of {path}.",
        ok=True,
        metadata={"command": "undo_edit", "path": path},
    )


def tool_str_replace_editor(container: "Container", args: Mapping[str, Any]) -> ToolResult:
    command = args.get("command")
    path = args.get("path")
    if command not in {"view", "create", "str_replace", "insert", "undo_edit"}:
        raise ValueError(f"str_replace_editor.command invalid: {command!r}")
    if not isinstance(path, str) or not path.startswith("/"):
        raise ValueError(f"str_replace_editor.path must be an absolute path, got {path!r}")

    if command == "view":
        return _view(container, path, args.get("view_range"))
    if command == "create":
        file_text = args.get("file_text")
        if not isinstance(file_text, str):
            raise ValueError("str_replace_editor.create requires file_text (string)")
        return _create(container, path, file_text)
    if command == "str_replace":
        old_str = args.get("old_str")
        new_str = args.get("new_str", "")
        if not isinstance(old_str, str) or not old_str:
            raise ValueError("str_replace requires non-empty old_str")
        if not isinstance(new_str, str):
            raise TypeError("str_replace.new_str must be a string")
        return _str_replace(container, path, old_str, new_str)
    if command == "insert":
        insert_line = args.get("insert_line")
        new_line = args.get("new_line")
        if not isinstance(insert_line, int) or isinstance(insert_line, bool):
            raise ValueError("insert requires integer insert_line")
        if not isinstance(new_line, str):
            raise ValueError("insert requires string new_line")
        return _insert(container, path, insert_line, new_line)
    return _undo_edit(container, path)


# ---------------------------------------------------------------------------
# finish
# ---------------------------------------------------------------------------

FINISH_SCHEMA: Mapping[str, Any] = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": (
            "Signal that the patch is ready. The agent loop terminates after "
            "this tool is called; the final patch is whatever git diff shows "
            "in the container."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Optional short summary of the fix for the trace.",
                }
            },
            "required": [],
        },
    },
}


def tool_finish(container: "Container", args: Mapping[str, Any]) -> ToolResult:
    summary = args.get("summary", "")
    if summary is not None and not isinstance(summary, str):
        raise TypeError("finish.summary must be a string")
    return ToolResult(
        tool_name="finish",
        content=FINISH_SENTINEL,
        ok=True,
        metadata={"summary": summary or ""},
    )


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

TOOL_REGISTRY: Mapping[str, tuple[Mapping[str, Any], ToolExecutor]] = {
    "bash": (BASH_SCHEMA, tool_bash),
    "str_replace_editor": (EDITOR_SCHEMA, tool_str_replace_editor),
    "finish": (FINISH_SCHEMA, tool_finish),
}


def default_tool_schemas() -> list[Mapping[str, Any]]:
    """Return the OpenAI tool schemas in the canonical order."""
    return [schema for schema, _ in TOOL_REGISTRY.values()]


def execute_tool_call(
    container: "Container",
    tool_name: str,
    arguments_json: str,
) -> ToolResult:
    """Dispatch a ``tool_calls[i].function`` entry to the matching executor."""
    if tool_name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: {tool_name!r}")
    try:
        args = json.loads(arguments_json) if arguments_json else {}
    except json.JSONDecodeError as ex:
        raise ValueError(
            f"tool {tool_name!r} arguments are not JSON: {ex}. Payload={arguments_json!r}"
        ) from ex
    if not isinstance(args, dict):
        raise ValueError(
            f"tool {tool_name!r} arguments must decode to an object, got {type(args).__name__}"
        )
    _, executor = TOOL_REGISTRY[tool_name]
    return executor(container, args)
