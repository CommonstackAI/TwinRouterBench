"""Run an unmodified agent harness through a sequential per-call model proxy."""

from __future__ import annotations

import copy
import hashlib
import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from contextlib import contextmanager
from typing import Any, Callable, Sequence

from .interactive import InteractiveBackend, OfficialVerdict
from .types import Assignment, ExecutionResult, TaskSpec


@contextmanager
def _serial(lock: Any):
    if not lock.acquire(blocking=False):
        raise RuntimeError("concurrent model calls require a separate non-sequential construction protocol")
    try:
        yield
    finally:
        lock.release()


def completion_envelope(message: dict[str, Any], model: str) -> dict[str, Any]:
    return {"id": "replayed-" + uuid.uuid4().hex, "object": "chat.completion", "created": 0,
            "model": model, "choices": [{"index": 0, "message": message,
            "finish_reason": "tool_calls" if message.get("tool_calls") else "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}


def encode_completion(envelope: dict[str, Any], stream: bool) -> bytes:
    if not stream:
        return json.dumps(envelope).encode()
    message = copy.deepcopy(envelope["choices"][0]["message"])
    for index, call in enumerate(message.get("tool_calls", [])):
        call["index"] = index
    base = {key: envelope[key] for key in ("id", "created", "model")}
    frames = [
        {**base, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": message, "finish_reason": None}]},
        {**base, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": envelope["choices"][0].get("finish_reason", "stop")}]},
        {**base, "object": "chat.completion.chunk", "choices": [], "usage": envelope.get("usage", {})},
    ]
    return ("".join("data: " + json.dumps(frame) + "\n\n" for frame in frames) + "data: [DONE]\n\n").encode()


class HarnessProxyBackend(InteractiveBackend):
    """The harness executes its native tools; its official grader supplies pass/fail.

    Initial messages can explicitly be frozen to the successful seed's input.
    Subsequent observations and tool schemas must still match byte-for-byte.
    Every billed request, including judge calls, uses the same capped client.
    """

    def __init__(self, client: Any, runner: Callable[[TaskSpec, str, str], OfficialVerdict]) -> None:
        super().__init__(client, lambda task: None)
        self.runner = runner

    def run_seed(self, task: TaskSpec, strong_model: str) -> ExecutionResult:
        maximum = int(task.metadata.get("max_steps", len(task.steps)))
        return self._run(task, [Assignment("high", strong_model)] * maximum)

    def run_mixed(self, task: TaskSpec, assignments: Sequence[Assignment], *, target_step: int,
                  generation_parameters: dict[str, Any], locked_execution: ExecutionResult | None = None) -> ExecutionResult:
        return self._run(task, assignments, target_step=target_step, locked=locked_execution)

    def _run(self, task: TaskSpec, assignments: Sequence[Assignment], *, target_step: int = 1,
             locked: ExecutionResult | None = None) -> ExecutionResult:
        prefixes: list[tuple[dict[str, Any], ...]] = []
        responses: list[dict[str, Any]] = []
        schemas: list[list[dict[str, Any]]] = []
        parameters: list[dict[str, Any]] = []
        initial = copy.deepcopy(locked.prefixes[0]) if locked is not None else None
        native_initial = None
        lock = threading.Lock()
        errors: list[BaseException] = []
        stage = "agent"
        judge_calls = 0
        nonce = uuid.uuid4().hex
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def do_POST(self) -> None:
                nonlocal initial, native_initial, stage, judge_calls
                stream = False
                body = {}
                try:
                    body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                    if self.path == "/__stage":
                        if self.headers.get("X-TwinRouter-Nonce") != nonce:
                            raise ValueError("invalid harness control nonce")
                        stage = body["stage"]
                        if stage not in {"agent", "judge"}:
                            raise ValueError("invalid harness stage")
                        self.send_response(200);self.end_headers();return
                    if not self.path.rstrip("/").endswith("/chat/completions"):
                        raise ValueError("proxy supports chat completions only")
                    with _serial(lock):
                        if errors:
                            raise RuntimeError("previous harness request failed; no automatic paid retry")
                        stream = bool(body.get("stream"))
                        messages = copy.deepcopy(body["messages"])
                        functions = copy.deepcopy(body.get("tools") or [])
                        generation = {key: value for key, value in body.items()
                                      if key not in {"model", "messages", "tools", "stream", "stream_options"}}
                        if stage == "judge":
                            owner.client._chat(model=owner.client.judge_model, messages=messages,
                                functions=functions, generation_parameters=generation)
                            envelope = owner.client.last_completion
                            judge_calls += 1
                        else:
                            index = len(responses)
                            if index >= len(assignments):
                                raise RuntimeError("harness exceeded the fixed trajectory step limit")
                            if native_initial is None:
                                native_initial = copy.deepcopy(messages)
                                if any(message.get("role") not in {"system", "user", "developer"}
                                       for message in native_initial):
                                    raise RuntimeError("native harness initial input contains prior assistant/tool state")
                            if initial is None:
                                initial = copy.deepcopy(messages)
                            elif locked is not None and task.metadata.get("freeze_initial_prefix"):
                                # Freeze the captured seed input (e.g. scaffold
                                # clock metadata), never any new tool feedback.
                                if index == 0 and (len(native_initial) != len(initial)
                                        or [message.get("role") for message in native_initial] != [message.get("role") for message in initial]):
                                    raise RuntimeError("native harness changed its initial scaffold structure")
                                # Auxiliary model calls and compaction can use
                                # different inputs. Keep those actual inputs;
                                # locked-call comparisons below still apply.
                                if messages[:len(native_initial)] == native_initial:
                                    messages[:len(initial)] = copy.deepcopy(initial)
                            if locked is not None and index < target_step:
                                old_schemas = locked.verification["functions_by_step"]
                                if tuple(messages) != locked.prefixes[index] or functions != old_schemas[index]:
                                    raise RuntimeError("harness replay changed locked input or tool schemas")
                            if index < target_step - 1:
                                message = copy.deepcopy(locked.responses[index])
                                envelope = completion_envelope(message, assignments[index].model)
                            else:
                                message = owner.client._chat(model=assignments[index].model,
                                    messages=messages, functions=functions, generation_parameters=generation)
                                envelope = copy.deepcopy(owner.client.last_completion)
                            prefixes.append(tuple(messages))
                            responses.append(copy.deepcopy(message))
                            schemas.append(functions)
                            parameters.append(generation)
                    payload = encode_completion(envelope, stream)
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream" if stream else "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers();self.wfile.write(payload)
                except Exception as exc:
                    owner.last_failed_request = copy.deepcopy(body)
                    errors.append(exc)
                    payload = json.dumps({"error": {"message": str(exc), "type": type(exc).__name__}}).encode()
                    self.send_response(502);self.send_header("Content-Length", str(len(payload)))
                    self.end_headers();self.wfile.write(payload)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            verdict = self.runner(task, f"http://127.0.0.1:{server.server_port}", nonce)
            if errors:
                raise RuntimeError(f"native harness model request failed: {errors[0]}") from errors[0]
            if not responses:
                raise RuntimeError("native harness made no recorded model calls")
            return ExecutionResult(verdict.passed, len(responses), tuple(prefixes), tuple(responses),
                verdict.reason, verification={**verdict.evidence, "functions_by_step": schemas,
                    "generation_parameters_by_step": parameters, "judge_calls": judge_calls,
                    "native_initial_sha256": hashlib.sha256(json.dumps(native_initial, sort_keys=True).encode()).hexdigest(),
                    "initial_prefix_frozen": bool(task.metadata.get("freeze_initial_prefix"))})
        finally:
            server.shutdown();server.server_close();thread.join(timeout=5)
