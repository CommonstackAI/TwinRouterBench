"""Versioned source registry parsing.

Remote sources are metadata-only by design.  The pipeline never downloads a
benchmark implicitly; callers materialize it themselves and pass a local path.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path


_REMOTE_RE = re.compile(
    r"^(?P<owner>[^/@]+)/(?P<repo>[^/@]+)@(?P<revision>[^/]+)(?:/(?P<path>.*))?$"
)
_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{7,64}$")


@dataclass(frozen=True)
class SourceSpec:
    uri: str
    kind: str
    locator: str
    revision: str = ""
    path: str = ""

    @classmethod
    def parse(cls, uri: str) -> "SourceSpec":
        if uri.startswith("fixture://"):
            locator = uri.removeprefix("fixture://")
            if not locator:
                raise ValueError("fixture source requires a fixture name")
            return cls(uri=uri, kind="fixture", locator=locator)

        if uri.startswith("local://"):
            locator = uri.removeprefix("local://")
            if not locator:
                raise ValueError("local source requires a path")
            return cls(uri=uri, kind="local", locator=locator)

        for kind in ("hf", "github"):
            prefix = f"{kind}://"
            if uri.startswith(prefix):
                remainder = uri.removeprefix(prefix)
                match = _REMOTE_RE.match(remainder)
                if not match:
                    raise ValueError(
                        f"{kind} source must look like "
                        f"{kind}://owner/repo@revision/optional/path"
                    )
                locator = f"{match.group('owner')}/{match.group('repo')}"
                return cls(
                    uri=uri,
                    kind=kind,
                    locator=locator,
                    revision=match.group("revision"),
                    path=match.group("path") or "",
                )

        path = Path(uri)
        return cls(uri=f"local://{uri}", kind="local", locator=str(path))

    @property
    def is_pinned(self) -> bool:
        if self.kind in {"fixture", "local"}:
            return True
        return bool(_COMMIT_RE.fullmatch(self.revision))

    def require_materialized_path(self) -> Path:
        if self.kind != "local":
            raise ValueError(
                f"{self.kind} source {self.uri!r} is registry metadata only; "
                "materialize it first and pass local:///absolute/path"
            )
        path = Path(self.locator).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"source path does not exist: {path}")
        return path

    def to_dict(self) -> dict[str, str | bool]:
        return {**asdict(self), "is_pinned": self.is_pinned}
