#!/usr/bin/env python3
"""Compare the first frame of exported demos with a fresh env reset on this machine.

The env is made the way the teammates' evaluation makes it: plain ``gym.make`` with
the file's env id and control mode and no camera arguments, plus
``FlattenRGBDObservationWrapper(rgb=True, depth=False)`` for the rgb side. For each of
the first ``--num`` demos it resets with the demo's seed and compares

- ``traj_i/obs[0]`` with an ``obs_mode="state"`` reset (must match on physx_cpu),
- ``traj_i/obs_rgb/state[0]`` with the wrapper's ``obs["state"]`` (must match on physx_cpu),
- ``traj_i/obs_rgb/rgb[0]`` with the wrapper's ``obs["rgb"]`` (reported: images rendered by
  another GPU / Vulkan driver differ slightly; a shader or camera mismatch shows as a large mean).

On physx_cuda initial states differ from CPU for the same seed (final-plan §2.1), so
there the numbers are only reported. Needs mani-skill 3.0.1; run with our wsl .venv.
WSL2 has no NVIDIA Vulkan device for ManiSkill's default renderer: pass
``--render-backend cpu`` there (lavapipe). The state env never renders.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import gymnasium as gym
import h5py
import mani_skill.envs  # noqa: F401  registers the envs
import numpy as np
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

STATE_ATOL = 1e-4


def to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", type=Path, help="scripts/export_demos.py outputs")
    parser.add_argument("--num", type=int, default=3, help="demos per file")
    parser.add_argument("--sim-backend", default="physx_cpu")
    parser.add_argument("--render-backend", help="default: ManiSkill's (a CUDA Vulkan device); 'cpu' on WSL")
    args = parser.parse_args()

    # Same rule as dp_manip/envs.py::ensure_render_icd: lavapipe where there is no NVIDIA ICD.
    if not os.environ.get("VK_ICD_FILENAMES") and not os.path.isfile("/usr/share/vulkan/icd.d/nvidia_icd.json"):
        os.environ["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/lvp_icd.json"
    strict = args.sim_backend == "physx_cpu"
    failures = 0
    for path in args.files:
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        env_id = meta["env_info"]["env_id"]
        control_mode = meta["dp_manip"]["control_mode"]
        kwargs = dict(control_mode=control_mode, sim_backend=args.sim_backend, num_envs=1)
        render = {"render_backend": args.render_backend} if args.render_backend else {}
        state_env = gym.make(env_id, obs_mode="state", render_backend="none", **kwargs)
        rgb_env = FlattenRGBDObservationWrapper(gym.make(env_id, obs_mode="rgb", **render, **kwargs),
                                                rgb=True, depth=False, state=True)
        with h5py.File(path, "r") as file:
            for index, episode in enumerate(meta["episodes"][: args.num]):
                seed = episode["episode_seed"]
                group = file[f"traj_{index}"]
                state_obs, _ = state_env.reset(seed=seed)
                rgb_obs, _ = rgb_env.reset(seed=seed)
                state_diff = np.abs(to_numpy(state_obs).reshape(-1) - group["obs"][0]).max()
                proprio_diff = np.abs(to_numpy(rgb_obs["state"]).reshape(-1) - group["obs_rgb/state"][0]).max()
                image = to_numpy(rgb_obs["rgb"])[0].astype(np.int16)
                stored = group["obs_rgb/rgb"][0].astype(np.int16)
                assert image.shape == stored.shape, f"{path.name} seed {seed}: rgb {image.shape} vs stored {stored.shape}"
                pixel = np.abs(image - stored)
                ok = state_diff <= STATE_ATOL and proprio_diff <= STATE_ATOL
                failures += strict and not ok
                print(f"{path.name} seed {seed}: obs max diff {state_diff:.1e}, obs_rgb/state max diff {proprio_diff:.1e}, "
                      f"rgb mean abs {pixel.mean():.2f}, max {pixel.max()}, >8 in {(pixel > 8).mean():.2%} of values"
                      f"{'' if ok or not strict else '  MISMATCH'}")
        state_env.close()
        rgb_env.close()
    print(f"{args.sim_backend}: {'all state checks passed' if not failures else f'{failures} demos mismatched'}"
          f"{'' if strict else ' (report only)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
