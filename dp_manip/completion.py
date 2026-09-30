"""Completion state of a run directory, shared by the trainer and schedulers.

``dp_manip.trainer.run_training`` stops at an existing
``checkpoints/final.pt`` only when the recorded ``run.json`` describes the same
configuration, so a different arm can never silently reuse another arm's
checkpoint. Any component that decides which runs still need training (the
dry-run planner, later the dual-GPU queue) must make exactly the same decision;
this module is that single decision, and the trainer delegates to it.

This module is deliberately stdlib-only: planning a 96-run sweep must not
import the training stack.
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass
from pathlib import Path

from .config import Config, from_recorded, same_run


class RunState(enum.Enum):
    """What would happen to a declared run under a given output root."""

    # ``checkpoints/final.pt`` exists for this exact configuration.
    COMPLETED = "completed"
    # No usable ``final.pt``; the trainer would run (resuming from ``resume.pt``
    # when present, via ``--resume auto``).
    PENDING = "pending"
    # ``final.pt`` exists but belongs to another configuration (or its
    # ``run.json`` cannot be read). The trainer refuses to reuse it; a scheduler
    # must surface it instead of counting it as completed or silently retrying.
    CONFLICT = "conflict"


@dataclass(frozen=True)
class Completion:
    """Trainer-consistent completion decision for one run directory."""

    state: RunState
    # Only meaningful for PENDING: a checkpoint the trainer can resume from.
    resumable: bool = False
    # Only set for CONFLICT: why the finished run cannot be reused.
    reason: str | None = None


def completion_state(
    cfg: Config, run_dir: str | Path, finetune: dict | None = None
) -> Completion:
    """Return what the trainer would do for ``cfg`` in ``run_dir``.

    ``finetune`` is the trainer's fine-tuning record (init checkpoint and its
    hash, frozen modules, schedule); a finished run must have recorded the same
    one. Baseline runs pass ``None`` and record none.

    Mirrors the skip block of :func:`dp_manip.trainer.run_training`:

    * no ``checkpoints/final.pt`` -> ``PENDING`` (``resume.pt`` does not change
      the state; the trainer resumes it through ``--resume auto``);
    * ``final.pt`` with a missing ``run.json`` -> ``COMPLETED``, exactly like
      the trainer, which only checks the recorded config when it can read it;
    * ``final.pt`` whose recorded config differs, or whose ``run.json`` is
      unreadable, -> ``CONFLICT``, never a silent skip.
    """
    run_dir = Path(run_dir)
    final_path = run_dir / "checkpoints" / "final.pt"
    resume_path = run_dir / "checkpoints" / "resume.pt"
    if not final_path.is_file():
        return Completion(RunState.PENDING, resumable=resume_path.is_file())

    run_info_path = run_dir / "run.json"
    if run_info_path.is_file():
        try:
            run_info = json.loads(run_info_path.read_text(encoding="utf-8"))
            finished = from_recorded(run_info["config"])
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            return Completion(RunState.CONFLICT, reason=f"run.json is unreadable: {error}")
        if not same_run(finished, cfg):
            return Completion(RunState.CONFLICT, reason="run.json records a different config")
        if run_info.get("finetune") != finetune:
            return Completion(RunState.CONFLICT, reason="run.json records a different fine-tuning init")
    return Completion(RunState.COMPLETED)
