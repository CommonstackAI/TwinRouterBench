"""mini-swe-agent ``Environment`` on top of a running SWE-bench container.

mini's own ``DockerEnvironment`` (see ``minisweagent/environments/docker.py``)
creates its own throwaway container with ``docker run -d ... sleep 2h`` and
does ``docker exec`` for every command. That's incompatible with SWE-bench's
``test_spec``-driven container lifecycle: SWE-bench containers are built by
``swebench.harness.docker_build.build_container`` with ``/testbed`` already
checked out at the instance's ``base_commit`` and all environment scripts
run.

So we implement mini's ``Environment`` protocol (a duck-typed Protocol: see
``minisweagent/__init__.py``) on top of an already-started
``swerouter.harness.container_runner.SwebenchContainerHandle``. We re-use
mini's exact command-execution contract so an unmodified ``DefaultAgent``
and ``LitellmModel`` observation template can drive our env without any
template tweaks. In particular we mirror:

* ``bash -lc <command>`` as the interpreter (mini's default), so ``$PATH``
  and conda activation scripts from the SWE-bench image are respected.
* The ``{"output", "returncode", "exception_info"}`` dict shape produced by
  both ``DockerEnvironment.execute`` and ``LocalEnvironment.execute``; this
  is what mini's ``observation_template`` renders with ``StrictUndefined``.
* The ``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`` sentinel protocol: if the
  agent prints that string as the first line of stdout with a zero exit
  code, mini raises ``minisweagent.exceptions.Submitted`` and the agent
  stops. Without this we'd spin until ``step_limit`` every time.

See also: ``minisweagent/environments/docker.py`` and
``minisweagent/environments/local.py`` for the contract we're mirroring.
"""

from __future__ import annotations

import logging
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from minisweagent.exceptions import Submitted

from swerouter.harness.container_runner import SwebenchContainerHandle


_SUBMIT_SENTINEL = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

logger = logging.getLogger("miniswerouter.env")


@dataclass
class SwebenchContainerEnvConfig:
    """Pydantic-free config mirroring ``mini``'s docker/local Environment shapes.

    mini uses a pydantic ``BaseModel`` on its built-in envs; we don't depend
    on pydantic here because the :class:`SwebenchContainerEnv` is not built
    from CLI / yaml and never needs validation. All fields are optional so
    our harness can drive mini's ``DefaultAgent`` without surprising it.
    """

    cwd: str = "/testbed"
    env: dict[str, str] = field(default_factory=dict)
    forward_env: list[str] = field(default_factory=list)
    timeout: int = 300
    interpreter: list[str] = field(default_factory=lambda: ["bash", "-lc"])
    executable: str = "docker"

    def model_dump(self, *, mode: str = "python") -> dict[str, Any]:
        """``mini`` uses ``config.model_dump()`` on its Pydantic configs; we
        fake the surface with a plain ``asdict``-like export."""

        _ = mode
        return {
            "cwd": self.cwd,
            "env": dict(self.env),
            "forward_env": list(self.forward_env),
            "timeout": self.timeout,
            "interpreter": list(self.interpreter),
            "executable": self.executable,
        }


class SwebenchContainerEnv:
    """Implements ``minisweagent.Environment`` on a SWE-bench work container.

    The caller is responsible for the container lifecycle — instantiate
    :class:`SwebenchContainerHandle`, ``start()`` it, pass it here, then
    ``stop()`` after the agent's ``run()`` returns. We don't own the handle
    because :mod:`miniswerouter.harness.run_instance` needs it alive after
    the agent exits (to run ``extract_git_diff``).
    """

    def __init__(
        self,
        handle: SwebenchContainerHandle,
        *,
        config: SwebenchContainerEnvConfig | None = None,
        logger_: logging.Logger | None = None,
    ) -> None:
        if handle.container is None or getattr(handle.container, "id", None) is None:
            raise RuntimeError(
                "SwebenchContainerEnv requires an already-started "
                "SwebenchContainerHandle (container.id must be set)"
            )
        self.handle = handle
        self.config = config or SwebenchContainerEnvConfig()
        self.logger = logger_ or logger

    def execute(
        self,
        action: dict,
        cwd: str = "",
        *,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Run ``action['command']`` inside the SWE-bench container and return
        ``{"output", "returncode", "exception_info"}``.

        Implementation intentionally mirrors
        ``DockerEnvironment.execute``: we invoke ``docker exec`` through
        ``subprocess.run`` (instead of docker-py's ``exec_run``) so we get
        the timeout behaviour mini's agent implicitly relies on.
        """

        command = action.get("command", "")
        effective_cwd = cwd or self.config.cwd
        cmd = [self.config.executable, "exec", "-w", effective_cwd]
        # mimic DockerEnvironment's env forwarding semantics.
        import os as _os

        for key in self.config.forward_env:
            value = _os.getenv(key)
            if value is not None:
                cmd.extend(["-e", f"{key}={value}"])
        for key, value in self.config.env.items():
            cmd.extend(["-e", f"{key}={value}"])
        cmd.extend([self.handle.container.id, *self.config.interpreter, command])

        try:
            result = subprocess.run(
                cmd,
                text=True,
                timeout=timeout or self.config.timeout,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            output = {
                "output": result.stdout,
                "returncode": result.returncode,
                "exception_info": "",
            }
        except Exception as e:
            raw_output = getattr(e, "output", None)
            if isinstance(raw_output, bytes):
                raw_output = raw_output.decode("utf-8", errors="replace")
            elif raw_output is None:
                raw_output = ""
            output = {
                "output": raw_output,
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {e}",
                "extra": {"exception_type": type(e).__name__, "exception": str(e)},
            }

        self._check_finished(output)
        return output

    def _check_finished(self, output: dict) -> None:
        """Mirror of ``DockerEnvironment._check_finished``: if stdout's first
        stripped line is the sentinel and returncode==0, raise ``Submitted``
        with the rest of stdout as the submission payload.
        """

        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if (
            lines
            and lines[0].strip() == _SUBMIT_SENTINEL
            and output["returncode"] == 0
        ):
            submission = "".join(lines[1:])
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        """Return variables for mini's Jinja2 prompt/observation templates.

        Mirrors ``DockerEnvironment.get_template_vars``'s merge of config +
        ``platform.uname()`` dict. Caller-supplied ``**kwargs`` take
        precedence (matches mini's convention via ``recursive_merge``).
        """

        base = {
            **self.config.model_dump(),
            **platform.uname()._asdict(),
        }
        base.update(kwargs)
        return base

    def serialize(self) -> dict:
        """Payload stored under ``info.config.environment`` in mini's trajectory."""

        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(mode="json"),
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                    "container_id": self.handle.container.id,
                    "swebench_test_spec_id": getattr(
                        self.handle.test_spec, "instance_id", None
                    ),
                }
            }
        }


__all__ = [
    "SwebenchContainerEnv",
    "SwebenchContainerEnvConfig",
]
