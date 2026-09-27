#!/usr/bin/env python3
"""Gate B checker: every cell of an experiment matrix must share one control.

Resolve all declared ``(value, replicate seed)`` cells of an experiment for one
or more tasks and compare their resolved configs. The declared experiment
variable, the replicate seed, runtime data paths and the backbone architecture
definitions may differ; anything else (optimizer, LR, batch size, budget, EMA,
diffusion, horizons, vision, evaluation) is reported as experiment drift.

```bash
python scripts/check_experiment.py --experiment backbone
python scripts/check_experiment.py --experiment data_size --task pickcube --seed 1
python scripts/check_experiment.py --experiment backbone --run-root /scratch/$USER/dp-runs
```

Without ``--run-root`` the check resolves the declared cells from the current
config files, which is the pre-submission check (REFACTOR_PLAN.md §3.5). With
``--run-root`` it reads the config each finished or running cell actually
trained with from ``<run-root>/<run name>/run.json``. That is where real drift
lives: runs submitted under an older ``baseline.toml``, or with a runtime
``--set``/``--num-demos``. Recorded configs are compared with each other and
with the current declaration; cells without ``run.json`` are listed as not run.
Pass the same ``--set`` values a smoke submitted so its recorded config is
compared against the intended declaration.

The exit status is non-zero when unexpected drift is found. Every cell also
gets a ``control_hash`` over its non-experimental values (§19); all cells of a
task matrix must share it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dp_manip import config as config_lib  # noqa: E402
from dp_manip.config import Config, ExperimentSpec  # noqa: E402
from dp_manip.invariants import (  # noqa: E402
    RUNTIME_KEYS,
    allowed_keys,
    config_differences,
    control_hash,
)


TASKS_DIR = ROOT / "configs" / "tasks"
EXPERIMENTS_DIR = ROOT / "configs" / "experiments"


@dataclass(frozen=True)
class Cell:
    """One resolved matrix cell: a declared value at a replicate seed."""

    label: str
    config: Config


@dataclass(frozen=True)
class Drift:
    """One unexpected config difference against the matrix reference cell."""

    key: str
    reference_label: str
    reference_value: Any
    cell_label: str
    cell_value: Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True, help="experiment name in configs/experiments or a spec path")
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="task to check (repeatable; default: every configs/tasks entry)",
    )
    parser.add_argument("--seed", type=int, help="restrict to one declared replicate seed")
    parser.add_argument("--data-root", type=Path, help="runtime override for data.root")
    parser.add_argument("--num-demos", type=int, help="runtime override for data.num_demos")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="runtime override applied to the declared cells (repeatable; must match the runs)",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        help="check the configs recorded in <run-root>/<run name>/run.json instead of the declaration",
    )
    return parser.parse_args(argv)


def matrix_cells(
    task_path: str | Path,
    experiment_path: str | Path,
    spec: ExperimentSpec,
    *,
    seed: int | None = None,
    overrides: Sequence[str] = (),
) -> list[Cell]:
    """Resolve every declared cell of the experiment for one task."""
    cells: list[Cell] = []
    for value in spec.values:
        for replicate in spec.seeds_for(value):
            if seed is not None and replicate != seed:
                continue
            cfg = config_lib.load(
                task_path,
                [*overrides, f"train.seed={replicate}"],
                experiment=experiment_path,
                experiment_value=value,
            )
            cells.append(Cell(f"{value} s{replicate}", cfg))
    return cells


def recorded_cells(
    declared: Sequence[Cell], run_root: Path
) -> tuple[list[Cell], list[Drift], list[str]]:
    """Load the config each declared cell actually trained with.

    Returns the recorded cells, every difference between a recorded config and
    its current declaration (runtime keys excepted), and the labels of cells
    whose run directory has no ``run.json`` yet.
    """
    recorded: list[Cell] = []
    mismatches: list[Drift] = []
    missing: list[str] = []
    for cell in declared:
        path = run_root / config_lib.default_run_name(cell.config) / "run.json"
        if not path.is_file():
            missing.append(cell.label)
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))["config"]
        config = config_lib.from_recorded(raw)
        recorded.append(Cell(cell.label, config))
        differences = config_differences(cell.config.to_dict(), config.to_dict())
        for key, (declared_value, recorded_value) in differences.items():
            if key not in RUNTIME_KEYS:
                mismatches.append(
                    Drift(key, "declared", declared_value, f"{cell.label} run.json", recorded_value)
                )
    return recorded, mismatches, missing


def check_cells(cells: Sequence[Cell], allowed: set[str]) -> list[Drift]:
    """Compare every cell against the first; return unexpected differences."""
    if not cells:
        return []
    reference = cells[0]
    reference_config = reference.config.to_dict()
    drifts: list[Drift] = []
    for cell in cells[1:]:
        differences = config_differences(reference_config, cell.config.to_dict())
        for key, (reference_value, cell_value) in differences.items():
            if key in allowed:
                continue
            drifts.append(
                Drift(key, reference.label, reference_value, cell.label, cell_value)
            )
    return drifts


def format_drifts(
    task_name: str, drifts: Sequence[Drift], title: str = "Unexpected experiment config drift"
) -> str:
    """Render the per-key matrix from the plan's example."""
    grouped: dict[str, list[Drift]] = {}
    for drift in drifts:
        grouped.setdefault(drift.key, []).append(drift)
    lines = [f"ERROR: {title} in {task_name}", ""]
    for key, entries in sorted(grouped.items()):
        lines.append(f"{key}:")
        reference = entries[0]
        lines.append(f"    reference {reference.reference_label} = {reference.reference_value!r}")
        for entry in entries:
            lines.append(f"    {entry.cell_label} = {entry.cell_value!r}")
        lines.append("")
    return "\n".join(lines).rstrip()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    experiment_path = config_lib.resolve_config_path(args.experiment, EXPERIMENTS_DIR, "experiment")
    spec = config_lib.load_experiment(experiment_path)
    if args.task:
        task_paths = [
            config_lib.resolve_config_path(name, TASKS_DIR, "task") for name in args.task
        ]
    else:
        task_paths = sorted(TASKS_DIR.glob("*.toml"))
    overrides: list[str] = list(args.overrides)
    if args.data_root is not None:
        overrides.append(f"data.root={args.data_root}")
    if args.num_demos is not None:
        overrides.append(f"data.num_demos={args.num_demos}")

    allowed = allowed_keys(spec)
    failures = 0
    total_cells = 0
    for task_path in task_paths:
        cells = matrix_cells(task_path, experiment_path, spec, seed=args.seed, overrides=overrides)
        mismatches: list[Drift] = []
        if args.run_root is not None:
            declared_count = len(cells)
            cells, mismatches, missing = recorded_cells(cells, args.run_root.expanduser())
            if missing:
                print(
                    f"{task_path.stem}: {len(cells)}/{declared_count} cells have run.json; "
                    f"not run: {', '.join(missing)}"
                )
        if mismatches:
            print(format_drifts(task_path.stem, mismatches, "run.json differs from the declared config"))
        if not cells:
            failures += bool(mismatches)
            continue
        total_cells += len(cells)
        drifts = check_cells(cells, allowed)
        hashes = {control_hash(cell.config, allowed) for cell in cells}
        failures += bool(mismatches or drifts or len(hashes) != 1)
        if drifts or len(hashes) != 1:
            if drifts:
                print(format_drifts(task_path.stem, drifts))
            else:
                print(f"ERROR: control_hash differs within {task_path.stem}: {sorted(hashes)}")
        elif not mismatches:
            print(f"{task_path.stem}: {len(cells)} cells ok, control_hash={next(iter(hashes))[:12]}")
    if failures:
        print(f"Gate B FAILED: {failures} task(s) drift", file=sys.stderr)
        return 1
    if total_cells == 0:
        print("ERROR: no experiment cells were selected", file=sys.stderr)
        return 1
    print(f"Gate B ok: {spec.name} matrix is controlled across {total_cells} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
