#!/usr/bin/env python3
"""Re-render saved evaluation episodes offline, at any resolution and shader.

Input is a states directory written by `eval_dp.py --save-states`
(env<i>.{h5,json}: reset seed + env state at every step). Each chosen episode
is reset with its seed, then every recorded state is restored with
set_state_dict and rendered, so the video shows exactly the evaluated episode;
no policy or physics is run. Rendering is decoupled from evaluation, so slow
shaders never affect success rates or timings.

Shaders: default (rasterized, ~0.1 s/frame at 1080p on wsl lavapipe),
rt (ray traced, ~25 s/frame there), rt-med, rt-fast (need the OptiX denoiser,
which lavapipe lacks: very noisy on wsl).

Example (wsl):
  .venv/bin/python scripts/render_episodes.py results/<exp>/states_test_final --seeds 10000 10010
  .venv/bin/python scripts/render_episodes.py results/<exp>/states_test_final --seeds 10010 \\
      --shader rt --width 1920 --height 1080 --eye 0.6 0.7 0.6 --target 0 0 0.35
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dp_manip.envs import ensure_render_icd  # noqa: E402


def load_episodes(states_dir: Path) -> dict[int, tuple[Path, dict, dict]]:
    """seed -> (h5 path, episode record, env_info) over all env<i> files."""
    found = {}
    for json_path in sorted(states_dir.glob("env*.json")):
        meta = json.loads(json_path.read_text())
        for ep in meta["episodes"]:
            seed = int(ep["reset_kwargs"].get("seed", ep["episode_seed"]))
            found[seed] = (json_path.with_suffix(".h5"), ep, meta["env_info"])
    if not found:
        raise SystemExit(f"no env*.json episodes in {states_dir}")
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("states_dir", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", help="episodes to render (default: all)")
    parser.add_argument("--shader", default="default", choices=("default", "rt", "rt-med", "rt-fast", "minimal"))
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fov", type=float, help="vertical field of view in radians (env default: 1.0)")
    parser.add_argument("--eye", type=float, nargs=3, help="camera position (default: the env's render camera)")
    parser.add_argument("--target", type=float, nargs=3, help="point the camera looks at (with --eye)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--crf", type=int, default=16, help="x264 quality, lower is better (default 16)")
    parser.add_argument("--out", type=Path, help="output directory (default: <states_dir>/../renders_<shader>_<W>x<H>)")
    args = parser.parse_args()
    if (args.eye is None) != (args.target is None):
        parser.error("--eye and --target go together")

    ensure_render_icd()
    import gymnasium as gym
    import h5py
    import imageio.v2 as imageio
    import mani_skill.envs  # noqa: F401
    from mani_skill.trajectory import utils as trajectory_utils
    from mani_skill.utils import sapien_utils

    episodes = load_episodes(args.states_dir)
    seeds = args.seeds or sorted(episodes)
    missing = [s for s in seeds if s not in episodes]
    if missing:
        raise SystemExit(f"seeds not in {args.states_dir}: {missing}")
    out_dir = args.out or args.states_dir.parent / f"renders_{args.shader}_{args.width}x{args.height}"
    out_dir.mkdir(parents=True, exist_ok=True)

    camera = dict(shader_pack=args.shader, width=args.width, height=args.height)
    if args.fov is not None:
        camera["fov"] = args.fov
    if args.eye is not None:
        camera["pose"] = sapien_utils.look_at(eye=args.eye, target=args.target)

    env_info = episodes[seeds[0]][2]
    kwargs = dict(env_info["env_kwargs"])
    kwargs.update(render_mode="rgb_array", render_backend="cpu", human_render_camera_configs=camera)
    env = gym.make(env_info["env_id"], **kwargs)
    try:
        for seed in seeds:
            h5_path, ep, _ = episodes[seed]
            with h5py.File(h5_path, "r") as f:
                states = trajectory_utils.dict_to_list_of_dicts(f[f"traj_{ep['episode_id']}"]["env_states"])
            env.reset(seed=seed)
            path = out_dir / f"seed{seed}.mp4"
            start = time.time()
            # macro_block_size=8 keeps 1080 rows (not a multiple of 16) unscaled.
            with imageio.get_writer(path, fps=args.fps, codec="libx264", quality=None, macro_block_size=8,
                                    pixelformat="yuv420p", output_params=["-crf", str(args.crf)]) as writer:
                for state in states:
                    env.unwrapped.set_state_dict(state)
                    frame = env.render()
                    frame = np.asarray(frame.cpu() if hasattr(frame, "cpu") else frame)
                    writer.append_data(frame[0] if frame.ndim == 4 else frame)
            print(f"seed {seed}: {len(states)} frames, success={ep.get('success')}, "
                  f"{time.time() - start:.1f}s -> {path}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
