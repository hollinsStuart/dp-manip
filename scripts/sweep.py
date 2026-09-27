#!/usr/bin/env python3
"""Show, plan and launch the declared experiment grid.

``show`` lists the stable index -> run mapping used by the Slurm arrays,
``plan`` prints a read-only completed/pending/conflict report for an output
root, and ``train``/``eval`` execute one indexed run.
"""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# The run list lives in the package so schedulers reuse the exact sweep order,
# config resolution and run names instead of re-deriving them.
from dp_manip.completion import RunState  # noqa: E402
from dp_manip.runlist import (  # noqa: E402
    DEFAULT_EXPERIMENT,
    TASKS,
    PlannedRun,
    Run,
    eval_command,
    plan_runs,
    runs,
    train_command,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("show", "plan", "train", "eval"))
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--task", choices=TASKS, help="restrict the grid to one declared task")
    parser.add_argument("--index", type=int)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs")
    parser.add_argument("--checkpoint", default="final.pt")
    parser.add_argument("--split", choices=("test", "val", "train"), default="test")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--render-backend")
    return parser.parse_args()


def selected_run(args: argparse.Namespace) -> Run:
    grid = runs(args.experiment, args.task)
    if args.index is None or not 0 <= args.index < len(grid):
        raise ValueError(f"--index must be in [0, {len(grid) - 1}]")
    return grid[args.index]


def print_plan(planned: list[PlannedRun]) -> int:
    """Print every run's state and the summary; return the plan's exit code."""
    counts = {state: 0 for state in RunState}
    for index, item in enumerate(planned):
        state = item.completion.state
        counts[state] += 1
        note = ""
        if state is RunState.PENDING and item.completion.resumable:
            note = " (resume.pt)"
        elif item.completion.reason is not None:
            note = f" ({item.completion.reason})"
        print(f"{index:03d} {item.run.name} {state.value}{note}")
    print(
        f"plan: {len(planned)} runs, {counts[RunState.COMPLETED]} completed (skipped), "
        f"{counts[RunState.PENDING]} pending, {counts[RunState.CONFLICT]} conflict"
    )
    # A config conflict would make the trainer refuse this run, so a plan with a
    # conflict must not report success.
    return 1 if counts[RunState.CONFLICT] else 0


def build_command(args: argparse.Namespace, run: Run) -> list[str]:
    if args.action == "train":
        return train_command(run, output_root=args.output_root, data_root=args.data_root)
    return eval_command(
        run,
        output_root=args.output_root,
        checkpoint=args.checkpoint,
        split=args.split,
        episodes=args.episodes,
        num_envs=args.num_envs,
        render_backend=args.render_backend,
    )


def main() -> None:
    args = parse_args()
    if args.action == "plan":
        planned = plan_runs(
            args.experiment,
            args.task,
            output_root=args.output_root,
            data_root=args.data_root,
        )
        raise SystemExit(print_plan(planned))
    grid = runs(args.experiment, args.task)
    if args.action == "show":
        for index, run in enumerate(grid):
            print(f"{index:03d} {run.name} {run.config.relative_to(ROOT)}")
        print(f"{len(grid)} runs")
        return

    run = selected_run(args)
    command = build_command(args, run)
    print(f"[{args.index}] {run.name}", flush=True)
    process = subprocess.Popen(command, cwd=ROOT)

    def forward_signal(signum, _frame) -> None:
        if process.poll() is None:
            process.send_signal(signum)

    signal.signal(signal.SIGTERM, forward_signal)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, forward_signal)
    raise SystemExit(process.wait())


if __name__ == "__main__":
    main()
