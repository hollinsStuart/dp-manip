#!/usr/bin/env python3
"""Convert ManiSkill state replays into the robomimic HDF5 layout read by Stanford DP.

Target reader: real-stanford/diffusion_policy
``diffusion_policy/dataset/robomimic_replay_lowdim_dataset.py``. It opens
``data/demo_0 .. data/demo_{n-1}`` in order, concatenates ``obs/<key>`` over
``obs_keys`` and reads ``actions``. ManiSkill's flat state vector is written as the
single key ``state``, so the task config uses ``obs_keys: [state]``.

Output layout::

    data                     attrs: total, env_args (JSON), split, seeds, ...
    data/demo_i/obs/state    (T, D) float32   = ManiSkill obs[:-1]
    data/demo_i/next_obs/state (T, D) float32 = ManiSkill obs[1:]
    data/demo_i/actions      (T, A) float32
    data/demo_i/dones        (T,) uint8, 1 on the last step
    data/demo_i/success      (T,) bool, copied from ManiSkill
    data/demo_i              attrs: num_samples, seed, source_file, source_episode_id

Demos are sorted by environment seed, so ``demo_0 .. demo_{N-1}`` is the "first N"
subset of docs/final-plan.md §2.2. Stanford's ``max_train_episodes`` draws a random
subset instead, so ``--subsets`` writes one file per N rather than relying on it.
Rewards are not written: the replays do not record them and DP does not use them.
"""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

# docs/final-plan.md §2.2: training pool 0-3999, validation demos 4000-4999.
# Validation rollouts (5000-5049) and test rollouts (10000-10099) never become demos.
SEED_RANGES = {"train": range(0, 4000), "val": range(4000, 5000)}
OBS_KEY = "state"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_episodes(path: Path) -> tuple[dict, list[dict]]:
    """Read one ManiSkill replay (H5 + JSON) and return its env_info and checked episodes."""
    with path.with_suffix(".json").open(encoding="utf-8") as stream:
        meta = json.load(stream)
    env_kwargs = meta["env_info"]["env_kwargs"]
    assert env_kwargs["obs_mode"] == "state", f"{path}: obs_mode is {env_kwargs['obs_mode']}, need state"
    # Older replays record the CPU backend as "cpu"; ManiSkill treats it as physx_cpu.
    assert env_kwargs.get("sim_backend") in ("cpu", "physx_cpu"), f"{path}: sim_backend is {env_kwargs.get('sim_backend')}"

    episodes = []
    with h5py.File(path, "r") as source:
        names = {f"traj_{item['episode_id']}" for item in meta["episodes"]}
        assert set(source.keys()) == names, f"{path}: H5 groups and JSON episodes differ"
        for item in meta["episodes"]:
            name = f"traj_{item['episode_id']}"
            group = source[name]
            assert isinstance(group["obs"], h5py.Dataset), f"{path}/{name}: obs is not a flat state array"
            obs = group["obs"][()]
            actions = group["actions"][()]
            success = group["success"][()]
            length = len(actions)
            seed = item["reset_kwargs"]["seed"]
            assert seed == item["episode_seed"], f"{path}/{name}: reset seed differs from episode_seed"
            assert item["control_mode"] == env_kwargs["control_mode"], f"{path}/{name}: control mode differs"
            assert item["elapsed_steps"] == length, f"{path}/{name}: elapsed_steps != len(actions)"
            assert obs.ndim == 2 and obs.shape[0] == length + 1, f"{path}/{name}: obs is not (T+1, D)"
            assert actions.ndim == 2 and success.shape == (length,), f"{path}/{name}: bad action/success shape"
            assert item["success"] is True and bool(success[-1]), f"{path}/{name}: episode did not succeed"
            assert np.isfinite(obs).all() and np.isfinite(actions).all(), f"{path}/{name}: non-finite values"
            episodes.append({
                "seed": int(seed),
                "obs": obs.astype(np.float32),
                "actions": actions.astype(np.float32),
                "success": success.astype(bool),
                "source_file": path.name,
                "source_episode_id": int(item["episode_id"]),
            })
    return meta["env_info"], episodes


def write_robomimic(path: Path, episodes: list[dict], data_attrs: dict) -> None:
    with h5py.File(path, "w") as out:
        data = out.create_group("data")
        for key, value in data_attrs.items():
            data.attrs[key] = value
        data.attrs["total"] = sum(len(ep["actions"]) for ep in episodes)
        data.attrs["seeds"] = np.array([ep["seed"] for ep in episodes], dtype=np.int64)
        for index, ep in enumerate(episodes):
            demo = data.create_group(f"demo_{index}")
            length = len(ep["actions"])
            demo.attrs["num_samples"] = length
            demo.attrs["seed"] = ep["seed"]
            demo.attrs["source_file"] = ep["source_file"]
            demo.attrs["source_episode_id"] = ep["source_episode_id"]
            demo.create_dataset(f"obs/{OBS_KEY}", data=ep["obs"][:-1])
            demo.create_dataset(f"next_obs/{OBS_KEY}", data=ep["obs"][1:])
            demo.create_dataset("actions", data=ep["actions"])
            dones = np.zeros(length, dtype=np.uint8)
            dones[-1] = 1
            demo.create_dataset("dones", data=dones)
            demo.create_dataset("success", data=ep["success"])


def verify(path: Path, episodes: list[dict]) -> None:
    """Re-read the file the way RobomimicReplayLowdimDataset does and compare with the source."""
    with h5py.File(path, "r") as file:
        demos = file["data"]
        assert len(demos) == len(episodes)
        for i, ep in enumerate(episodes):
            demo = demos[f"demo_{i}"]
            obs = np.concatenate([demo["obs"][key] for key in [OBS_KEY]], axis=-1).astype(np.float32)
            actions = demo["actions"][:].astype(np.float32)
            assert np.array_equal(obs, ep["obs"][:-1]), f"demo_{i}: obs mismatch"
            assert np.array_equal(actions, ep["actions"]), f"demo_{i}: actions mismatch"
            assert obs.shape[0] == actions.shape[0] == demo.attrs["num_samples"]
            assert demo.attrs["seed"] == ep["seed"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path,
                        help="ManiSkill state replay H5 files (JSON alongside); merged and sorted by seed")
    parser.add_argument("-o", "--output", type=Path, required=True, help="output .hdf5 path")
    parser.add_argument("--split", choices=sorted(SEED_RANGES), required=True,
                        help="train = pool seeds 0-3999, val = validation demo seeds 4000-4999")
    parser.add_argument("--num-demos", type=int,
                        help="keep the first K demos by seed (e.g. 400 for the pool, 50 for val); default all")
    parser.add_argument("--subsets", type=lambda s: [int(x) for x in s.split(",")], default=[],
                        help="also write <output stem>_n<N>.hdf5 for each N, e.g. 25,50,100,200")
    args = parser.parse_args()

    env_info, episodes = load_episodes(args.inputs[0])
    for path in args.inputs[1:]:
        info, loaded = load_episodes(path)
        assert info["env_id"] == env_info["env_id"], f"{path}: env_id differs from {args.inputs[0]}"
        assert info["env_kwargs"]["control_mode"] == env_info["env_kwargs"]["control_mode"], \
            f"{path}: control mode differs from {args.inputs[0]}"
        episodes.extend(loaded)

    episodes.sort(key=lambda ep: ep["seed"])
    seeds = [ep["seed"] for ep in episodes]
    assert len(set(seeds)) == len(seeds), "duplicate seeds across inputs"
    allowed = SEED_RANGES[args.split]
    outside = [s for s in seeds if s not in allowed]
    assert not outside, f"seeds outside the {args.split} range {allowed.start}-{allowed.stop - 1}: {outside[:10]}"
    if args.num_demos is not None:
        assert len(episodes) >= args.num_demos, f"only {len(episodes)} successful demos, need {args.num_demos}"
        episodes = episodes[: args.num_demos]
    for n in args.subsets:
        assert n <= len(episodes), f"subset {n} is larger than the {len(episodes)} demos kept"
    obs_dims = {ep["obs"].shape[1] for ep in episodes}
    action_dims = {ep["actions"].shape[1] for ep in episodes}
    assert len(obs_dims) == 1 and len(action_dims) == 1, "observation or action dims differ between demos"

    env_kwargs = env_info["env_kwargs"]
    data_attrs = {
        "env_args": json.dumps({"env_name": env_info["env_id"], "type": "maniskill", "env_kwargs": env_kwargs}),
        "split": args.split,
        "obs_key": OBS_KEY,
        "control_mode": env_kwargs["control_mode"],
        "sources": json.dumps({path.name: sha256(path) for path in args.inputs}),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    targets = [(args.output, episodes)]
    targets += [(args.output.with_name(f"{args.output.stem}_n{n}{args.output.suffix}"), episodes[:n])
                for n in args.subsets]
    for path, subset in targets:
        write_robomimic(path, subset, data_attrs)
        verify(path, subset)
        lengths = [len(ep["actions"]) for ep in subset]
        summary = {
            "file": path.name,
            "sha256": sha256(path),
            "env_id": env_info["env_id"],
            "control_mode": env_kwargs["control_mode"],
            "split": args.split,
            "num_demos": len(subset),
            "total_steps": int(sum(lengths)),
            "obs_dim": next(iter(obs_dims)),
            "action_dim": next(iter(action_dims)),
            "length_min_median_max": [min(lengths), int(np.median(lengths)), max(lengths)],
            "seeds": [ep["seed"] for ep in subset],
            "sources": json.loads(data_attrs["sources"]),
        }
        path.with_suffix(".json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
        print(f"{path}: {len(subset)} demos, {sum(lengths)} steps, seeds {subset[0]['seed']}-{subset[-1]['seed']}, "
              f"obs {summary['obs_dim']}, action {summary['action_dim']}, sha256 {summary['sha256'][:16]}")
    print(f"env {env_info['env_id']}, control mode {env_kwargs['control_mode']}; "
          f"Stanford DP config: obs_keys [{OBS_KEY}], obs_dim {next(iter(obs_dims))}, "
          f"action_dim {next(iter(action_dims))}, val_ratio 0")


if __name__ == "__main__":
    main()
