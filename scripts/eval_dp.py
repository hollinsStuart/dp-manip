#!/usr/bin/env python3
"""Evaluate a trained checkpoint on the config's held-out test seeds.

Uses the config stored in the checkpoint, so control mode, horizons and episode
length always match training. Writes results/<exp>/test_<ckpt>.json.

Example (wsl):
  .venv/bin/python scripts/eval_dp.py checkpoints/pickcube_smoke/final.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dp_manip.envs import make_eval_envs  # noqa: E402
from dp_manip.evaluate import evaluate  # noqa: E402
from dp_manip.policy import load_checkpoint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episodes", type=int, help="evaluate only the first N test seeds")
    parser.add_argument("--video", action="store_true", help="record videos of the first env")
    parser.add_argument("--seed", type=int, default=0, help="torch seed for diffusion sampling noise")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("warning: CUDA not available, using cpu")
        device = torch.device("cpu")
    torch.manual_seed(args.seed)

    policy, cfg, ckpt = load_checkpoint(args.checkpoint, device)
    seeds = cfg.test_seeds()[: args.episodes] if args.episodes else cfg.test_seeds()
    exp_dir = Path(__file__).resolve().parents[1] / "results" / args.checkpoint.parent.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    tag = args.checkpoint.stem

    envs = make_eval_envs(cfg, min(cfg.eval.num_envs, len(seeds)),
                          video_dir=str(exp_dir / f"videos_test_{tag}") if args.video else None)
    try:
        result = evaluate(policy, envs, seeds, device)
    finally:
        envs.close()
    result.update(checkpoint=str(args.checkpoint), iteration=ckpt["iteration"], sampling_seed=args.seed)
    out = exp_dir / f"test_{tag}.json"
    out.write_text(json.dumps(result, indent=2))
    s = result["summary"]
    print(f"{cfg.task.env_id} {tag} (iter {ckpt['iteration']}): success_once={s.get('success_once'):.3f} "
          f"success_at_end={s.get('success_at_end'):.3f} over {s['num_episodes']} test episodes "
          f"(seeds {seeds[0]}..{seeds[-1]}), {s['mean_inference_ms']:.1f} ms/inference -> {out}")


if __name__ == "__main__":
    main()
