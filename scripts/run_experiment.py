#!/usr/bin/env python3
"""Unified experiment entry point: task + experiment + value + seed.

```bash
python scripts/run_experiment.py --task pickcube --experiment data_size --value 50 --seed 0
python scripts/run_experiment.py --task pickcube --experiment backbone --value transformer --seed 0
```

``--task`` and ``--experiment`` accept the short names declared in
``configs/tasks`` and ``configs/experiments`` (or explicit paths). The entry
point only resolves the canonical config layering and delegates to the single
trainer in ``dp_manip.trainer``; there is deliberately no per-experiment
branch, so a new experiment is a config file, never a new pipeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dp_manip import config as config_lib  # noqa: E402
from dp_manip import invariants  # noqa: E402
from dp_manip.config import Config  # noqa: E402


TASKS_DIR = ROOT / "configs" / "tasks"
EXPERIMENTS_DIR = ROOT / "configs" / "experiments"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, help="task name in configs/tasks or a task config path")
    parser.add_argument("--experiment", required=True, help="experiment name in configs/experiments or a spec path")
    parser.add_argument("--value", required=True, help="one declared value of the experiment grid")
    parser.add_argument("--seed", type=int, help="training seed; default: baseline train.seed")
    parser.add_argument("--num-demos", type=int, help="runtime override for data.num_demos")
    parser.add_argument("--data-root", type=Path, help="override data.root (for example, a scratch dataset directory)")
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs")
    parser.add_argument("--exp", help="run directory name; default: <task>_rgb_<backbone>_n<N>_s<seed>")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", choices=("auto", "never"), default="auto")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override a config value (repeatable)",
    )
    return parser.parse_args(argv)


def resolve_config(args: argparse.Namespace) -> Config:
    """Resolve baseline -> task -> experiment(value) -> runtime seed/data."""
    return config_lib.load_run(
        config_lib.resolve_config_path(args.task, TASKS_DIR, "task"),
        args.overrides,
        experiment=config_lib.resolve_config_path(args.experiment, EXPERIMENTS_DIR, "experiment"),
        experiment_value=args.value,
        num_demos=args.num_demos,
        seed=args.seed,
        data_root=args.data_root,
    )


def experiment_context(args: argparse.Namespace, cfg: Config) -> dict:
    """Describe the declared cell this invocation resolves to (plan §19)."""
    experiment_path = config_lib.resolve_config_path(args.experiment, EXPERIMENTS_DIR, "experiment")
    spec = config_lib.load_experiment(experiment_path)
    return invariants.experiment_context(cfg, spec, args.value, spec_path=experiment_path)


def main() -> int:
    args = parse_args()
    cfg = resolve_config(args)
    print(
        f"{args.experiment}={args.value} seed={cfg.train.seed} backbone={cfg.policy.backbone} "
        f"num_demos={cfg.data.num_demos} device={args.device}",
        flush=True,
    )
    # Imported late so config resolution stays importable without the heavy
    # torch training stack (the same reason both entry points share it).
    from dp_manip.trainer import run_training

    return run_training(
        cfg,
        output_root=args.output_root,
        run_name=args.exp,
        device=args.device,
        resume=args.resume,
        experiment_context=experiment_context(args, cfg),
    )


if __name__ == "__main__":
    raise SystemExit(main())
