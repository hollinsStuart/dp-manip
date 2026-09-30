"""Run metadata helpers kept independent of the training loop (plan §19)."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # typing only: importing dp_manip.data would pull in torch
    from .data import DatasetInfo


def git_revision(cwd: str | Path) -> dict[str, str | bool | None]:
    """Record the exact code revision; missing git is reported as ``None``."""

    def capture(*arguments: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *arguments], cwd=cwd, capture_output=True, text=True, timeout=10
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    # Only modified tracked files change the code that ran; an untracked note
    # or scratch file must not mark every run as dirty.
    status = capture("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": capture("rev-parse", "HEAD"),
        "branch": capture("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def file_sha256(path: str | Path) -> str:
    """Hash a file in 1 MiB blocks (checkpoints are a few hundred MB)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_fingerprint(info: "DatasetInfo") -> dict[str, Any]:
    """Identify a dataset file without re-reading its compressed RGB.

    Hashing the whole HDF5 at every job start would re-read gigabytes on the
    shared filesystem, so the fingerprint covers the exporter's sidecar metadata
    (episode ids/seeds and environment info) plus the file size.
    ``data_selection.demo_seeds`` and the dataset path complete the audit trail.
    """
    sidecar = info.path.with_suffix(".json")
    return {
        "algorithm": "sha256(sidecar.json)+h5_size",
        "sidecar_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
        "h5_size_bytes": info.path.stat().st_size,
    }
