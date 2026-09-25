#!/usr/bin/env python3
"""Quality numbers of generated demos (final-plan §2.3), per task and split.

For every ``<record dir>/<env>/motionplanning/<name>.h5`` raw demo file that has
conversions next to it (``<name>.rgb.<mode>.physx_cpu.h5`` / ``<name>.state...``) it reports:
expert successes / seeds tried, conversion successes per obs mode and the seeds they
dropped, episodes with a mid-episode env reset (dropped again by export_demos.py),
demo lengths and the number of cameras. Standard library, h5py and numpy only.

    scripts/smoke/replay_stats.py demos-smoke0925b [--name smoke_train smoke_val]
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

MAX_QPOS_STEP = 0.2  # same rule as scripts/export_demos.py


def resets(path: Path) -> list[int]:
    seeds = []
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    with h5py.File(path, "r") as file:
        for episode in meta["episodes"]:
            articulations = file[f"traj_{episode['episode_id']}/env_states/articulations"]
            for articulation in articulations.values():
                state = articulation[()]
                dof = (state.shape[1] - 13) // 2
                if np.abs(np.diff(state[:, 13:13 + dof], axis=0)).max() > MAX_QPOS_STEP:
                    seeds.append(episode["episode_seed"])
                    break
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("record_dir", type=Path)
    parser.add_argument("--name", nargs="+", default=["smoke_train", "smoke_val"], help="raw demo file stems")
    args = parser.parse_args()

    for task_dir in sorted(args.record_dir.glob("*/motionplanning")):
        env = task_dir.parent.name
        for name in args.name:
            raw_path = task_dir / f"{name}.h5"
            if not raw_path.exists():
                continue
            raw = json.loads(raw_path.with_suffix(".json").read_text(encoding="utf-8"))["episodes"]
            raw_seeds = [episode["episode_seed"] for episode in raw]
            line = [f"{env:20s} {name:12s} expert {len(raw_seeds)}/{max(raw_seeds) - min(raw_seeds) + 1} seeds tried"]
            for mode in ("rgb", "state"):
                converted = sorted(task_dir.glob(f"{name}.{mode}.*.physx_cpu.h5"))
                converted = [path for path in converted if not path.name.endswith(".first_obs.h5")]
                if not converted:
                    continue
                meta = json.loads(converted[0].with_suffix(".json").read_text(encoding="utf-8"))
                seeds = [episode["episode_seed"] for episode in meta["episodes"]]
                lengths = [episode["elapsed_steps"] for episode in meta["episodes"]]
                line.append(f"{mode} {len(seeds)}/{len(raw_seeds)} (dropped {sorted(set(raw_seeds) - set(seeds))})")
                if mode == "rgb":
                    with h5py.File(converted[0], "r") as file:
                        cameras = list(file[f"traj_{meta['episodes'][0]['episode_id']}/obs/sensor_data"].keys())
                    reset = resets(converted[0])
                    usable = len(seeds) - len(reset)
                    line.append(f"mid-episode resets {reset}, usable {usable}/{len(raw_seeds)} = {usable / len(raw_seeds):.0%}, "
                                f"len {min(lengths)}-{int(np.median(lengths))}-{max(lengths)}, cameras {cameras}")
            print(", ".join(line))


if __name__ == "__main__":
    main()
