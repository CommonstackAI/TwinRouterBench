"""Scaffold-agnostic helpers for driving a SWE-bench Verified instance.

Both scaffolds that live in / depend on SWERouterBench share the same
per-instance mechanics: load the dataset row, build a Docker work container
from the upstream image, extract the final patch via ``git diff``, and hand
the patch to SWE-bench's official evaluator for the canonical ``resolved``
flag. Only the agent loop that drives the model in between differs.

This module keeps those shared mechanics in one place so downstream scaffolds
(e.g. the editor-flavoured loop in :mod:`swerouter.harness.run_instance` and
the mini-swe-agent bridge in ``MiniSWERouterBench``) import the same code
path. No routing, pricing, or trace logic belongs here.

All heavy imports (``docker``, ``swebench``) are deferred to the function
bodies so unit tests that don't exercise a real cluster can still import the
module.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_DATASET_NAME = "SWE-bench/SWE-bench_Verified"
DEFAULT_DATASET_SPLIT = "test"

# Upstream's pre-built instance images live under this Docker Hub namespace.
# Passing it to ``make_test_spec`` tells swebench to pull rather than rebuild.
DEFAULT_IMAGE_NAMESPACE: str | None = "swebench"

# Model identifier embedded in the predictions dict handed to
# ``swebench.harness.run_evaluation``. Upstream uses it as the sub-directory
# under ``logs/run_evaluation/<run_id>/<model_name>/<instance_id>/``.
DEFAULT_EVAL_MODEL_NAME = "swerouterbench"

# The Windows compatibility path temporarily monkeypatches upstream
# swebench.harness.run_evaluation.copy_to_container. Keep that mutation
# process-local and serialized so concurrent workers cannot restore each
# other's patch while an eval is still running.
_WINDOWS_COMPAT_EVAL_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_dataset_instance(
    instance_id: str,
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    dataset_split: str = DEFAULT_DATASET_SPLIT,
) -> dict[str, Any]:
    """Return the single SWE-bench row for ``instance_id`` or raise ValueError.

    This function pays the full HF-dataset download/cache cost once per
    ``(dataset_name, dataset_split)``; callers that need many rows should
    cache the result themselves.
    """

    from swebench.harness.utils import load_swebench_dataset

    dataset = list(load_swebench_dataset(dataset_name, dataset_split))
    for row in dataset:
        if row.get("instance_id") == instance_id:
            return row
    raise ValueError(
        f"instance_id {instance_id!r} not found in dataset "
        f"(size={len(dataset)}). Double check dataset_name / split."
    )


def make_test_spec_for_instance(
    instance: dict[str, Any],
    *,
    image_namespace: str | None = DEFAULT_IMAGE_NAMESPACE,
    windows_compat: bool = False,
) -> Any:
    """Build the upstream ``TestSpec`` used by the container builder and the
    official evaluator. ``image_namespace`` pulls pre-built images from Docker
    Hub when set (saves the multi-hour local rebuild path)."""

    if not windows_compat:
        from swebench.harness.test_spec.test_spec import make_test_spec

        return make_test_spec(instance, namespace=image_namespace)

    import inspect

    try:
        from swebench.harness.test_spec.test_spec import make_test_spec
    except ModuleNotFoundError as exc:
        if exc.name != "swebench.harness.test_spec.test_spec":
            raise
        from swebench.harness.test_spec import make_test_spec

    make_test_spec_params = inspect.signature(make_test_spec).parameters
    if (
        "namespace" in make_test_spec_params
        or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in make_test_spec_params.values())
    ):
        return make_test_spec(instance, namespace=image_namespace)
    return make_test_spec(instance)


# ---------------------------------------------------------------------------
# Container lifecycle
# ---------------------------------------------------------------------------


def _is_docker_container_name_conflict_error(exc: BaseException) -> bool:
    """Return True when ``exc`` reflects Docker create failing on a taken name.

    SWE-bench uses a fixed container name per ``(instance_id, run_id)``. If a
    previous process died after creating the container but before
    :meth:`SwebenchContainerHandle.stop`, the next run hits HTTP 409. We treat
    that as recoverable by removing the orphaned container and retrying.
    """

    import docker.errors

    cur: BaseException | None = exc
    while cur is not None:
        if isinstance(cur, docker.errors.APIError):
            code = getattr(cur, "status_code", None)
            if code == 409:
                return True
        cur = cur.__cause__ or cur.__context__
    text = str(exc)
    return "409" in text and ("already in use" in text or "Conflict" in text)


def _remove_container_if_exists(client: Any, name: str) -> None:
    """``docker rm -f`` semantics for ``name``; no-op when absent."""

    import docker.errors

    try:
        existing = client.containers.get(name)
    except docker.errors.NotFound:
        return
    existing.remove(force=True)


@dataclass
class SwebenchContainerHandle:
    """Owning handle around one running SWE-bench work container.

    Lifecycle:

    1. Construct with the instance's ``test_spec`` and a ``run_id``.
    2. Call :meth:`start` (or use the context-manager form) to build + start
       the container and get the ``docker`` ``Container`` object.
    3. Use ``self.container`` however your scaffold needs (``exec_run`` /
       ``subprocess docker exec`` / etc.).
    4. Call :meth:`stop` (or exit the ``with`` block) to let swebench's
       cleanup routine remove the container.
    """

    test_spec: Any
    run_id: str
    log_path: Path
    force_rebuild: bool = False
    container: Any = None
    client: Any = None
    logger: Any = None

    def start(self) -> Any:
        """Build and start the container. Returns the docker container handle."""

        import docker
        from swebench.harness.docker_build import build_container, setup_logger

        if self.container is not None:
            raise RuntimeError(
                "SwebenchContainerHandle.start() called twice without stop()"
            )
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.client = docker.from_env()
        self.logger = setup_logger(self.test_spec.instance_id, self.log_path)
        container_name = self.test_spec.get_instance_container_name(self.run_id)
        max_attempts = 3
        for attempt in range(max_attempts):
            _remove_container_if_exists(self.client, container_name)
            try:
                self.container = build_container(
                    test_spec=self.test_spec,
                    client=self.client,
                    run_id=self.run_id,
                    logger=self.logger,
                    nocache=False,
                    force_rebuild=self.force_rebuild,
                )
                self.container.start()
                return self.container
            except Exception as ex:  # noqa: BLE001 — narrow retry below
                if attempt + 1 < max_attempts and _is_docker_container_name_conflict_error(
                    ex
                ):
                    continue
                raise

    def stop(self) -> None:
        """Cleanup the container (idempotent)."""

        from swebench.harness.docker_build import close_logger
        from swebench.harness.docker_utils import cleanup_container

        if self.container is not None and self.client is not None:
            try:
                cleanup_container(self.client, self.container, self.logger)
            finally:
                self.container = None
        if self.logger is not None:
            try:
                close_logger(self.logger)
            finally:
                self.logger = None

    def __enter__(self) -> "SwebenchContainerHandle":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


def extract_git_diff(container: Any, *, exclude_paths: tuple[str, ...] = ()) -> str:
    """Return the diff of /testbed vs its base commit.

    The container is expected to have ``/testbed`` checked out at the
    instance's ``base_commit`` (upstream's ``build_container`` guarantees
    this). We ``git add -A`` first so untracked files created by the agent
    are included in the diff.

    ``exclude_paths`` lists pathspecs (relative to ``/testbed``) that must not
    appear in the final diff -- used by scaffolds whose submission protocol
    writes a plumbing file inside ``/testbed`` (e.g. mini-swe-agent uses
    ``patch.txt`` as the intermediate artifact for its
    ``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`` handshake; when we fall back to
    ``git diff`` because the handshake didn't complete, that file would
    otherwise leak into the captured patch).
    """

    pathspec = ""
    if exclude_paths:
        import shlex

        # Quote the full git pathspec token; quoting only the trailing path leaves
        # ``:(exclude)`` unquoted and breaks under ``/bin/bash -lc`` (``(`` subshell).
        pathspec = " -- . " + " ".join(
            shlex.quote(f":(exclude){p}") for p in exclude_paths
        )
    res = container.exec_run(
        cmd=[
            "/bin/bash",
            "-lc",
            f"cd /testbed && git add -A && git diff --cached{pathspec}",
        ],
        demux=False,
        tty=False,
    )
    exit_code = int(res.exit_code) if res.exit_code is not None else -1
    out = res.output
    if isinstance(out, bytes):
        out = out.decode("utf-8", errors="replace")
    elif out is None:
        out = ""
    else:
        out = str(out)
    if exit_code != 0:
        raise RuntimeError(
            f"git diff failed in container: exit={exit_code} output={out[:500]!r}"
        )
    return out


# ---------------------------------------------------------------------------
# Upstream evaluation
# ---------------------------------------------------------------------------

@dataclass
class EvalReport:
    """Summary of upstream's official grading pass for one instance."""

    resolved: bool
    patch_applied: bool
    test_counts: dict[str, int] = field(default_factory=dict)
    report_path: Path | None = None
    error: str | None = None


def _ingest_report_json(report_path: Path, instance_id: str) -> EvalReport:
    if not report_path.is_file():
        return EvalReport(
            resolved=False,
            patch_applied=False,
            report_path=report_path,
            error=f"eval report not produced at {report_path}",
        )
    try:
        doc = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as ex:
        return EvalReport(
            resolved=False,
            patch_applied=False,
            report_path=report_path,
            error=f"eval report parse failed: {ex}",
        )
    entry = doc.get(instance_id)
    if not isinstance(entry, dict):
        return EvalReport(
            resolved=False,
            patch_applied=False,
            report_path=report_path,
            error=f"eval report has no entry for {instance_id!r}",
        )
    counts: dict[str, int] = {}
    tests = entry.get("tests_status", {}) or {}
    for k in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        bucket = tests.get(k) or {}
        if isinstance(bucket, dict):
            counts[f"{k}.success"] = len(bucket.get("success", []) or [])
            counts[f"{k}.failure"] = len(bucket.get("failure", []) or [])
    return EvalReport(
        resolved=bool(entry.get("resolved", False)),
        patch_applied=bool(entry.get("patch_successfully_applied", False)),
        test_counts=counts,
        report_path=report_path,
        error=None,
    )


def _copy_to_container_posix(container: Any, src: Path, dst: Any) -> None:
    """Copy ``src`` to a POSIX path inside ``container``.

    SWE-bench may pass ``Path("/eval.sh")`` to its copy helper. On Windows
    that can become ``WindowsPath("\\eval.sh")`` before reaching the Linux
    container. This shim is used only when Windows compatibility mode is
    explicitly enabled.
    """

    import io
    import posixpath
    import shlex
    import tarfile

    src_path = Path(src)
    dst_path = str(dst).replace("\\", "/")
    parent = posixpath.dirname(dst_path)
    basename = posixpath.basename(dst_path)
    if not dst_path.startswith("/") or not parent or not basename:
        raise ValueError(f"destination path must be an absolute file path: {dst!r}")

    if parent != "/":
        mkdir_cmd = f"mkdir -p {shlex.quote(parent)}"
        res = container.exec_run(["/bin/sh", "-lc", mkdir_cmd])
        if int(getattr(res, "exit_code", 1) or 0) != 0:
            out = getattr(res, "output", b"")
            if isinstance(out, bytes):
                out = out.decode("utf-8", errors="replace")
            raise RuntimeError(f"failed to create container directory {parent}: {out}")

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        if basename.endswith(".sh"):
            data = src_path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            info = tarfile.TarInfo(name=basename)
            stat_result = src_path.stat()
            info.size = len(data)
            info.mode = stat_result.st_mode
            info.mtime = stat_result.st_mtime
            tar.addfile(info, io.BytesIO(data))
        else:
            tar.add(src_path, arcname=basename)
    buf.seek(0)
    if not container.put_archive(parent, buf.read()):
        raise RuntimeError(f"failed to copy {src_path} to container path {dst_path}")


def run_upstream_eval(
    *,
    test_spec: Any,
    instance_id: str,
    patch_text: str,
    run_id: str,
    client: Any = None,
    timeout_sec: int = 1800,
    rm_image: bool = False,
    model_name: str = DEFAULT_EVAL_MODEL_NAME,
    windows_compat: bool = False,
) -> EvalReport:
    """Grade ``patch_text`` against ``instance_id`` using upstream's pipeline.

    An empty ``patch_text`` short-circuits to a failure report (no patch =
    no chance to resolve). Otherwise we drive ``swebench.harness.run_evaluation``
    with the ``(test_spec, pred)`` pair and parse the on-disk report.

    ``client`` defaults to a fresh ``docker.from_env()`` so callers that have
    already torn down their own work-container client don't need to keep one
    alive across the eval boundary.
    """

    from swebench.harness.constants import LOG_REPORT, RUN_EVALUATION_LOG_DIR

    if not patch_text.strip():
        return EvalReport(
            resolved=False,
            patch_applied=False,
            error="no patch produced (agent loop produced empty diff)",
        )

    if client is None:
        import docker

        client = docker.from_env()

    pred = {
        "instance_id": instance_id,
        "model_name_or_path": model_name,
        "model_patch": patch_text,
    }
    report_path = (
        Path(RUN_EVALUATION_LOG_DIR) / run_id / model_name / instance_id / LOG_REPORT
    )
    try:
        if windows_compat:
            _run_upstream_eval_windows_compat(
                test_spec=test_spec,
                pred=pred,
                rm_image=rm_image,
                client=client,
                run_id=run_id,
                timeout_sec=timeout_sec,
            )
        else:
            from swebench.harness.run_evaluation import (
                run_instance as upstream_run_instance,
            )

            upstream_run_instance(
                test_spec=test_spec,
                pred=pred,
                rm_image=rm_image,
                force_rebuild=False,
                client=client,
                run_id=run_id,
                timeout=timeout_sec,
                rewrite_reports=False,
            )
    except Exception as ex:
        return EvalReport(
            resolved=False,
            patch_applied=False,
            report_path=report_path,
            error=f"{type(ex).__name__}: {ex}",
        )
    return _ingest_report_json(report_path, instance_id)


def _run_upstream_eval_windows_compat(
    *,
    test_spec: Any,
    pred: dict[str, str],
    rm_image: bool,
    client: Any,
    run_id: str,
    timeout_sec: int,
) -> None:
    """Run upstream eval with Windows-only compatibility shims enabled."""

    import inspect
    import swebench.harness.run_evaluation as run_evaluation_module

    upstream_run_instance = run_evaluation_module.run_instance
    kwargs: dict[str, Any] = {
        "test_spec": test_spec,
        "pred": pred,
        "rm_image": rm_image,
        "force_rebuild": False,
        "client": client,
        "run_id": run_id,
        "timeout": timeout_sec,
    }
    upstream_params = inspect.signature(upstream_run_instance).parameters
    if (
        "rewrite_reports" in upstream_params
        or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in upstream_params.values())
    ):
        kwargs["rewrite_reports"] = False

    with _WINDOWS_COMPAT_EVAL_LOCK:
        sentinel = object()
        original_copy_to_container = getattr(
            run_evaluation_module, "copy_to_container", sentinel
        )
        run_evaluation_module.copy_to_container = _copy_to_container_posix
        try:
            upstream_run_instance(**kwargs)
        finally:
            if original_copy_to_container is sentinel:
                delattr(run_evaluation_module, "copy_to_container")
            else:
                run_evaluation_module.copy_to_container = original_copy_to_container


__all__ = [
    "DEFAULT_DATASET_NAME",
    "DEFAULT_DATASET_SPLIT",
    "DEFAULT_IMAGE_NAMESPACE",
    "DEFAULT_EVAL_MODEL_NAME",
    "EvalReport",
    "SwebenchContainerHandle",
    "extract_git_diff",
    "load_dataset_instance",
    "make_test_spec_for_instance",
    "run_upstream_eval",
]
