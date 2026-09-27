#!/usr/bin/env python3
"""Train the pending runs of one sweep with one GPU per queue worker.

The cluster QOS allows one Slurm job with two GPUs, so this entry point plans
the declared runs of an experiment, skips the runs whose ``final.pt`` already
matches the resolved config, and feeds the rest through the shared dynamic
queue (:mod:`dp_manip.scheduler`). Worker N only ever sees physical GPU N via
``CUDA_VISIBLE_DEVICES``; ``scripts/train_dp.py`` keeps its own ``--device
cuda`` default, so no GPU argument is added to the command.

``--data-root`` defaults to ``$DATA_ROOT`` and ``--output-root`` to
``$RUN_ROOT`` (falling back to ``<repository>/runs``), matching the environment
variables the Slurm scripts use:

```bash
DATA_ROOT=/scratch/$USER/dp-data/dataset RUN_ROOT=/scratch/$USER/dp-runs \
python scripts/train_queue.py --task peginsertionside
```

The exit status is 0 when every run finished, 1 when a required run failed and
no run is interrupted, and 75 when runs were interrupted after writing
``resume.pt`` (or never started). 75 is the Slurm requeue convention: the
Slurm entry point requeues the job, and the next launch skips completed runs
and resumes interrupted ones through ``--resume auto``.

``--set SECTION.KEY=VALUE`` (repeatable) adds an ordinary runtime config
override to every training command, for example a reduced-budget smoke:

```bash
python scripts/train_queue.py --task peginsertionside \\
  --set train.total_iters=200 --set train.validation_steps=[200] \\
  --set train.checkpoint_steps=[200]
```
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dp_manip.completion import RunState  # noqa: E402
from dp_manip.runlist import (  # noqa: E402
    DEFAULT_EXPERIMENT,
    TASKS,
    PlannedRun,
    Run,
    plan_runs,
    train_command,
)
from dp_manip.scheduler import (  # noqa: E402
    MAX_WORKERS,
    Job,
    QueueSummary,
    gpu_environment,
    run_queue,
)


def queue_training(
    planned: Sequence[PlannedRun],
    *,
    output_root: str | Path,
    data_root: str | Path | None = None,
    logs_dir: str | Path | None = None,
    workers: int = MAX_WORKERS,
    overrides: Sequence[str] = (),
    python: str | None = None,
    command_builder: Callable[[Run], Sequence[str]] | None = None,
) -> QueueSummary:
    """Run every non-completed run of the plan through the shared queue.

    ``command_builder`` exists so tests can inject a fake training command; the
    default builder is the same :func:`dp_manip.runlist.train_command` the sweep
    CLI uses, so no task/value/seed mapping is duplicated here. ``overrides``
    are passed through as ordinary ``--set`` config overrides. Conflicting runs
    are queued deliberately: the trainer refuses them with a clear error and the
    queue reports a failure instead of skipping them silently.
    """
    if command_builder is None:
        def command_builder(run: Run) -> Sequence[str]:
            return train_command(
                run,
                python=python,
                output_root=output_root,
                data_root=data_root,
                overrides=overrides,
            )

    jobs = [
        Job(name=item.run.name, command=command_builder(item.run))
        for item in planned
        if item.completion.state is not RunState.COMPLETED
    ]
    skipped = tuple(item.run.name for item in planned if item.completion.state is RunState.COMPLETED)
    if logs_dir is None:
        logs_dir = Path(output_root).expanduser().resolve() / "logs"
    return run_queue(
        jobs,
        workers=workers,
        logs_dir=logs_dir,
        skipped=skipped,
        worker_env=gpu_environment(workers),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--task", choices=TASKS, help="restrict the sweep to one declared task")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=os.environ.get("DATA_ROOT") or None,
        help="override data.root (default: $DATA_ROOT)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=os.environ.get("RUN_ROOT") or ROOT / "runs",
        help="run directory root (default: $RUN_ROOT, else <repository>/runs)",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override a config value for every run (repeatable; for example smoke budgets)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_WORKERS,
        choices=tuple(range(1, MAX_WORKERS + 1)),
        help="parallel trainers; a single Slurm job has at most 2 GPUs",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    planned = plan_runs(
        args.experiment,
        args.task,
        output_root=args.output_root,
        overrides=args.overrides,
        data_root=args.data_root,
    )
    summary = queue_training(
        planned,
        output_root=args.output_root,
        data_root=args.data_root,
        overrides=args.overrides,
        workers=args.workers,
    )
    return summary.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
