from __future__ import annotations

import io
import sys
import tarfile
import threading
import types
from typing import Any, Callable

from swerouter.harness import container_runner


def _install_eval_modules(
    monkeypatch: Any,
    tmp_path: Any,
    run_instance: Callable[..., None],
    *,
    copy_to_container: object | None,
) -> types.ModuleType:
    swebench = types.ModuleType("swebench")
    harness = types.ModuleType("swebench.harness")
    constants = types.ModuleType("swebench.harness.constants")
    constants.LOG_REPORT = "report.json"
    constants.RUN_EVALUATION_LOG_DIR = str(tmp_path)
    run_evaluation = types.ModuleType("swebench.harness.run_evaluation")
    run_evaluation.run_instance = run_instance
    if copy_to_container is not None:
        run_evaluation.copy_to_container = copy_to_container

    swebench.harness = harness
    harness.constants = constants
    harness.run_evaluation = run_evaluation

    monkeypatch.setitem(sys.modules, "swebench", swebench)
    monkeypatch.setitem(sys.modules, "swebench.harness", harness)
    monkeypatch.setitem(sys.modules, "swebench.harness.constants", constants)
    monkeypatch.setitem(
        sys.modules, "swebench.harness.run_evaluation", run_evaluation
    )
    return run_evaluation


def _install_test_spec_modules(
    monkeypatch: Any,
    make_test_spec: Callable[..., object],
    *,
    nested_module: bool,
) -> None:
    swebench = types.ModuleType("swebench")
    harness = types.ModuleType("swebench.harness")
    test_spec_pkg = types.ModuleType("swebench.harness.test_spec")
    test_spec_pkg.__path__ = []

    swebench.harness = harness
    harness.test_spec = test_spec_pkg
    monkeypatch.setitem(sys.modules, "swebench", swebench)
    monkeypatch.setitem(sys.modules, "swebench.harness", harness)
    monkeypatch.setitem(sys.modules, "swebench.harness.test_spec", test_spec_pkg)

    if nested_module:
        test_spec_mod = types.ModuleType("swebench.harness.test_spec.test_spec")
        test_spec_mod.make_test_spec = make_test_spec
        test_spec_pkg.test_spec = test_spec_mod
        monkeypatch.setitem(
            sys.modules, "swebench.harness.test_spec.test_spec", test_spec_mod
        )
    else:
        test_spec_pkg.make_test_spec = make_test_spec
        monkeypatch.delitem(
            sys.modules, "swebench.harness.test_spec.test_spec", raising=False
        )


def test_default_eval_path_does_not_patch_copy_to_container(monkeypatch: Any, tmp_path: Any) -> None:
    calls: list[dict[str, Any]] = []
    original_copy = object()
    module_box: dict[str, types.ModuleType] = {}

    def run_instance(**kwargs: Any) -> None:
        assert module_box["module"].copy_to_container is original_copy
        calls.append(kwargs)

    module_box["module"] = _install_eval_modules(
        monkeypatch,
        tmp_path,
        run_instance,
        copy_to_container=original_copy,
    )

    report = container_runner.run_upstream_eval(
        test_spec=object(),
        instance_id="example__repo-1",
        patch_text="diff --git a/file.py b/file.py\n",
        run_id="default_eval",
        client=object(),
    )

    assert calls
    assert calls[0]["rewrite_reports"] is False
    assert module_box["module"].copy_to_container is original_copy
    assert report.error and "eval report not produced" in report.error


def test_windows_eval_path_patches_and_restores_copy_to_container(monkeypatch: Any, tmp_path: Any) -> None:
    calls: list[dict[str, Any]] = []
    original_copy = object()
    module_box: dict[str, types.ModuleType] = {}

    def run_instance(**kwargs: Any) -> None:
        assert (
            module_box["module"].copy_to_container
            is container_runner._copy_to_container_posix
        )
        calls.append(kwargs)

    module_box["module"] = _install_eval_modules(
        monkeypatch,
        tmp_path,
        run_instance,
        copy_to_container=original_copy,
    )

    report = container_runner.run_upstream_eval(
        test_spec=object(),
        instance_id="example__repo-1",
        patch_text="diff --git a/file.py b/file.py\n",
        run_id="windows_eval",
        client=object(),
        windows_compat=True,
    )

    assert calls
    assert calls[0]["rewrite_reports"] is False
    assert module_box["module"].copy_to_container is original_copy
    assert report.error and "eval report not produced" in report.error


def test_windows_eval_path_serializes_temporary_copy_patch(
    monkeypatch: Any,
    tmp_path: Any,
) -> None:
    calls: list[dict[str, Any]] = []
    entered = 0
    lock = threading.Lock()
    first_inside = threading.Event()
    release_first = threading.Event()
    second_thread_started = threading.Event()
    second_started = threading.Event()
    thread_errors: list[BaseException] = []
    original_copy = object()
    module_box: dict[str, types.ModuleType] = {}

    def run_instance(**kwargs: Any) -> None:
        nonlocal entered
        with lock:
            entered += 1
            call_index = entered
        calls.append(kwargs)
        assert (
            module_box["module"].copy_to_container
            is container_runner._copy_to_container_posix
        )
        if call_index == 1:
            first_inside.set()
            assert release_first.wait(timeout=5)
        else:
            second_started.set()

    module_box["module"] = _install_eval_modules(
        monkeypatch,
        tmp_path,
        run_instance,
        copy_to_container=original_copy,
    )

    def run_eval_thread(*, started: threading.Event | None = None) -> None:
        if started is not None:
            started.set()
        try:
            container_runner.run_upstream_eval(
                test_spec=object(),
                instance_id="example__repo-1",
                patch_text="diff --git a/file.py b/file.py\n",
                run_id="windows_eval",
                client=object(),
                windows_compat=True,
            )
        except BaseException as ex:  # noqa: BLE001 - re-raised in main test thread
            thread_errors.append(ex)

    t1 = threading.Thread(target=run_eval_thread)
    t2 = threading.Thread(
        target=run_eval_thread,
        kwargs={"started": second_thread_started},
    )
    t1.start()
    assert first_inside.wait(timeout=5)
    t2.start()
    assert second_thread_started.wait(timeout=5)

    # If the monkeypatch were not serialized, the second call could enter
    # while the first still owns the temporary module mutation.
    assert not second_started.wait(timeout=0.2)
    release_first.set()
    t1.join(timeout=5)
    t2.join(timeout=5)

    if thread_errors:
        raise thread_errors[0]
    assert not t1.is_alive()
    assert not t2.is_alive()
    assert len(calls) == 2
    assert second_started.is_set()
    assert module_box["module"].copy_to_container is original_copy


def test_windows_copy_shim_uses_posix_container_paths(tmp_path: Any) -> None:
    class ExecResult:
        exit_code = 0
        output = b""

    class FakeContainer:
        def __init__(self) -> None:
            self.exec_calls: list[list[str]] = []
            self.archives: list[tuple[str, bytes]] = []

        def exec_run(self, cmd: list[str]) -> ExecResult:
            self.exec_calls.append(cmd)
            return ExecResult()

        def put_archive(self, path: str, data: bytes) -> bool:
            self.archives.append((path, data))
            return True

    src = tmp_path / "eval.sh"
    src.write_bytes(b"#!/bin/sh\r\necho ok\r\n")
    container = FakeContainer()

    container_runner._copy_to_container_posix(container, src, "\\tmp\\eval.sh")

    assert container.exec_calls == [["/bin/sh", "-lc", "mkdir -p /tmp"]]
    assert len(container.archives) == 1
    parent, data = container.archives[0]
    assert parent == "/tmp"
    with tarfile.open(fileobj=io.BytesIO(data), mode="r") as tar:
        member = tar.getmember("eval.sh")
        assert tar.extractfile(member).read() == b"#!/bin/sh\necho ok\n"


def test_make_test_spec_default_uses_nested_module(monkeypatch: Any) -> None:
    calls: list[tuple[dict[str, str], str | None]] = []

    def make_test_spec(instance: dict[str, str], namespace: str | None = None) -> str:
        calls.append((instance, namespace))
        return "spec"

    _install_test_spec_modules(monkeypatch, make_test_spec, nested_module=True)

    spec = container_runner.make_test_spec_for_instance(
        {"instance_id": "i"}, image_namespace="swebench"
    )

    assert spec == "spec"
    assert calls == [({"instance_id": "i"}, "swebench")]


def test_make_test_spec_windows_compat_supports_old_import_and_signature(
    monkeypatch: Any,
) -> None:
    calls: list[dict[str, str]] = []

    def make_test_spec(instance: dict[str, str]) -> str:
        calls.append(instance)
        return "old-spec"

    _install_test_spec_modules(monkeypatch, make_test_spec, nested_module=False)

    spec = container_runner.make_test_spec_for_instance(
        {"instance_id": "i"},
        image_namespace="swebench",
        windows_compat=True,
    )

    assert spec == "old-spec"
    assert calls == [{"instance_id": "i"}]


def test_make_test_spec_windows_compat_keeps_namespace_when_supported(
    monkeypatch: Any,
) -> None:
    calls: list[tuple[dict[str, str], str | None]] = []

    def make_test_spec(instance: dict[str, str], namespace: str | None = None) -> str:
        calls.append((instance, namespace))
        return "new-spec"

    _install_test_spec_modules(monkeypatch, make_test_spec, nested_module=True)

    spec = container_runner.make_test_spec_for_instance(
        {"instance_id": "i"},
        image_namespace="swebench",
        windows_compat=True,
    )

    assert spec == "new-spec"
    assert calls == [({"instance_id": "i"}, "swebench")]


def test_run_cli_windows_compat_flag_defaults_false_and_can_be_enabled() -> None:
    from miniswerouter.cli import _build_parser as build_mini_parser
    from swerouter.cli import _build_parser as build_swe_parser

    required_args = [
        "run",
        "--router-import",
        "pkg.mod:Router.from_cli_args",
        "--router-label",
        "example",
        "--output-dir",
        "runs/example",
    ]

    assert build_mini_parser().parse_args(required_args).windows_compat is False
    assert build_swe_parser().parse_args(required_args).windows_compat is False
    assert (
        build_mini_parser().parse_args(required_args + ["--windows-compat"]).windows_compat
        is True
    )
    assert (
        build_swe_parser().parse_args(required_args + ["--windows-compat"]).windows_compat
        is True
    )
