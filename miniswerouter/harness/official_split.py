"""The dynamic track runs only the locked leaderboard instance list."""

from __future__ import annotations

from pathlib import Path

import swerouter

_TRB_ROOT = Path(swerouter.__file__).resolve().parent.parent
OFFICIAL_DYNAMIC_IDS_PATH = _TRB_ROOT / "data" / "dynamic" / "dynamic_heldout100_ids.txt"


def load_official_dynamic_instance_ids(path: Path | None = None) -> list[str]:
    """Return the locked dynamic instance ids, in file order."""

    path = Path(path) if path is not None else OFFICIAL_DYNAMIC_IDS_PATH
    if not path.is_file():
        raise FileNotFoundError(f"official dynamic instance list missing: {path}")
    ids: list[str] = []
    seen: set[str] = set()
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line in seen:
            raise ValueError(f"duplicate instance id {line!r} in {path} at line {lineno}")
        seen.add(line)
        ids.append(line)
    if not ids:
        raise ValueError(f"official dynamic instance list is empty: {path}")
    return ids


def select_official_dynamic_ids(
    selected: list[str] | tuple[str, ...] | None,
    *,
    official: list[str],
    limit: int | None = None,
) -> list[str]:
    """Choose ids to run, in official-list order.

    ``selected is None`` means the whole official list. An id outside that
    list is an error. ``limit`` truncates the chosen list and cannot add ids.
    """

    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
        raise ValueError(f"limit must be a positive integer, got {limit!r}")
    official_set = set(official)
    if selected is None:
        chosen = list(official)
    else:
        requested = [str(item).strip() for item in selected if str(item).strip()]
        if not requested:
            raise ValueError("instance list is empty")
        outside = [item for item in requested if item not in official_set]
        if outside:
            preview = ", ".join(outside[:8])
            suffix = "" if len(outside) <= 8 else f" (+{len(outside) - 8} more)"
            raise ValueError(
                "dynamic track only runs the official "
                f"{len(official)}-case list; refusing {len(outside)} id(s) "
                f"outside that list: {preview}{suffix}"
            )
        wanted = set(requested)
        chosen = [item for item in official if item in wanted]
    if limit is not None:
        chosen = chosen[:limit]
    if not chosen:
        raise ValueError("no official dynamic instances left to run")
    return chosen


def instances_outside_official(instance_ids, *, official: list[str]) -> list[str]:
    """Return sorted ids that are not on the official dynamic list."""

    official_set = set(official)
    return sorted({str(item) for item in instance_ids} - official_set)


def assert_no_instances_outside_official(instance_ids, *, official: list[str]) -> None:
    """Fail when a run contains instances the dynamic track does not keep."""

    outside = instances_outside_official(instance_ids, official=official)
    if not outside:
        return
    preview = ", ".join(outside[:8])
    suffix = "" if len(outside) <= 8 else f" (+{len(outside) - 8} more)"
    raise ValueError(
        "dynamic track only keeps the official "
        f"{len(official)}-case list; remove {len(outside)} other result(s): "
        f"{preview}{suffix}"
    )
