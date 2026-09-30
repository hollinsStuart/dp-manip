"""Per-task lock file of the failure-aware study (plan §12.1).

``configs/failure_aware/<task>.toml`` records every value the study derives
for one task, stage by stage: baseline cell and checkpoints, collection
results, the pilot's lr/steps, fine-tuned models, the offline gate, the dry
run and the alpha selections. A section is written exactly once and never
edited afterwards; later stages and every evaluation read their inputs from
here, so nothing downstream can silently use a different checkpoint or alpha.

The standard library reads TOML but cannot write it, so this module carries
a small writer for the value types the lock file uses.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOCK_DIR = ROOT / "configs" / "failure_aware"
HEADER = (
    "# Failure-aware study lock file (docs/failure_aware_finetuning_plan.md §12).\n"
    "# Written stage by stage by scripts/failure_study.py; sections are never edited\n"
    "# after the stage that fills them. Commit it before the test evaluation.\n"
)


def lock_path(task: str) -> Path:
    return LOCK_DIR / f"{task}.toml"


def _key(key: str) -> str:
    if key and all(char.isalnum() or char in "_-" for char in key):
        return key
    return json.dumps(key)


def _value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"TOML cannot store {value!r}")
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_value(item) for item in value) + "]"
    raise TypeError(f"unsupported lock value {value!r} ({type(value).__name__})")


def dumps(data: Mapping[str, Any]) -> str:
    """Serialize nested tables of scalars and lists (no arrays of tables)."""
    lines: list[str] = []

    def table(prefix: list[str], values: Mapping[str, Any]) -> None:
        scalars = {key: value for key, value in values.items() if not isinstance(value, Mapping)}
        tables = {key: value for key, value in values.items() if isinstance(value, Mapping)}
        if prefix and (scalars or not tables):
            lines.append("")
            lines.append("[" + ".".join(_key(part) for part in prefix) + "]")
        for key, value in scalars.items():
            if value is None:
                continue
            lines.append(f"{_key(key)} = {_value(value)}")
        for key, value in tables.items():
            table([*prefix, key], value)

    table([], data)
    return "\n".join(lines).lstrip("\n") + "\n"


def _normalize(values: Mapping[str, Any]) -> dict[str, Any]:
    """JSON-shaped copy (tuples -> lists) without ``None``, which TOML cannot hold."""

    def clean(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): clean(item) for key, item in value.items() if item is not None}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        return value

    return clean(json.loads(json.dumps(values)))


class LockFile:
    """Append-only view of one task's lock file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data: dict[str, Any] = (
            tomllib.loads(self.path.read_text(encoding="utf-8")) if self.path.is_file() else {}
        )

    @property
    def sha256(self) -> str | None:
        if not self.path.is_file():
            return None
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    def get(self, section: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in section.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def has(self, section: str) -> bool:
        return self.get(section) is not None

    def require(self, section: str, stage: str) -> Any:
        value = self.get(section)
        if value is None:
            raise RuntimeError(f"{self.path}: [{section}] is missing; run the '{stage}' stage first")
        return value

    def write(self, section: str, values: Mapping[str, Any]) -> None:
        """Add ``[section]`` once; an existing section is never replaced."""
        if self.has(section):
            raise FileExistsError(
                f"{self.path}: [{section}] is already locked; a locked stage is not re-run. "
                "Record any justified change by hand in the plan's §11 change log."
            )
        parts = section.split(".")
        node = self.data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"{section}: {part!r} is a value, not a table")
        node[parts[-1]] = {
            **_normalize(values),
            "recorded_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".partial")
        temporary.write_text(HEADER + "\n" + dumps(self.data), encoding="utf-8")
        # Round-trip before replacing, so a writer bug can never corrupt the lock.
        if tomllib.loads(temporary.read_text(encoding="utf-8")) != self.data:
            temporary.unlink()
            raise RuntimeError("lock file serialization does not round-trip")
        os.replace(temporary, self.path)
