#!/usr/bin/env python3
"""Fine-tune a baseline checkpoint on its own rollouts (failure-aware plan §4).

Trains the failure model (``--label failure``) or the success-control model
(``--label success``) of one baseline checkpoint::

    finetune_dp.py <final.pt> --label failure --lr 1e-5 --steps 20000 --checkpoint-steps 5000 10000

The config is the checkpoint's own recorded config with only the data paths,
``train.lr``, ``train.total_iters`` and the intermediate checkpoint steps
changed, so model, diffusion and evaluation settings cannot drift from the
baseline. Training goes through the single trainer
(``dp_manip.trainer.run_training``) with a :class:`FinetuneSpec`: baseline
weights and normalization, frozen observation encoder, constant lr after warmup.
Runs land in ``<rollout dir>/<label>_model_lr<lr>_it<steps>/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dp_manip import config as config_lib  # noqa: E402
from dp_manip.failure_protocol import DEFAULT_PROTOCOL, load_protocol  # noqa: E402
from dp_manip.failure_rollout import DATASET_DIR, rollout_dir_for  # noqa: E402
from dp_manip.finetune import FinetuneSpec, rollout_overrides  # noqa: E402

FROZEN_MODULES = ("observation_encoder",)  # plan §2.2
LR_SCHEDULE = "constant_with_warmup"  # plan §4.4


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path, help="baseline final.pt the rollouts were collected from")
    parser.add_argument("--label", choices=("failure", "success"), required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--steps", type=int, required=True, help="total optimizer steps")
    parser.add_argument(
        "--checkpoint-steps", type=int, nargs="*", default=[], help="extra EMA checkpoints (pilot)"
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--run-root", type=Path, help="as for collect_rollouts.py")
    parser.add_argument("--rollout-dir", type=Path, help="default: <run root>/failure_aware/<task>/s<seed>")
    parser.add_argument("--output-root", type=Path, help="default: the rollout directory")
    parser.add_argument("--exp", help="run directory name; default: <label>_model_lr<lr>_it<steps>")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", choices=("auto", "never"), default="auto")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="machine overrides such as train.num_workers (locked sections are rejected)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import torch

    from dp_manip.trainer import run_training

    checkpoint_path = args.checkpoint.resolve()
    baseline = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    recorded = config_lib.from_recorded(baseline["config"])
    rollout_dir = args.rollout_dir or rollout_dir_for(
        checkpoint_path, recorded.task.name, recorded.train.seed, args.run_root
    )
    overrides = rollout_overrides(
        rollout_dir / DATASET_DIR,
        args.label,
        lr=args.lr,
        total_iters=args.steps,
        checkpoint_steps=args.checkpoint_steps,
    )
    cfg = config_lib.from_recorded(baseline["config"], [*overrides, *args.overrides])
    del baseline
    protocol = load_protocol(args.protocol)
    protocol.check_against(cfg)
    spec = FinetuneSpec(
        init_checkpoint=str(checkpoint_path),
        frozen_modules=FROZEN_MODULES,
        lr_schedule=LR_SCHEDULE,
        train_seed_range=protocol.seeds.collection_train,
        val_seed_range=protocol.seeds.collection_holdout,
    )
    return run_training(
        cfg,
        output_root=args.output_root or rollout_dir,
        run_name=args.exp or f"{args.label}_model_lr{args.lr:g}_it{args.steps}",
        device=args.device,
        resume=args.resume,
        finetune=spec,
    )


if __name__ == "__main__":
    raise SystemExit(main())
