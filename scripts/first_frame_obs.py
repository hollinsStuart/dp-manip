#!/usr/bin/env python3
"""Recompute frame 0 (observation and env state) of converted ManiSkill replays (sidecar file).

Two frame-0 defects of ``replay_trajectory --use-first-env-state -c <mode>`` in
mani-skill 3.0.1, which replays all episodes of a file in one env:

- the observation is taken right after setting the first state, with no physics step,
  so contact-based entries still describe the end of the previous episode: PickCube's
  ``extra/is_grasped`` is 1 in frame 0 of every episode but the first;
- ``env_states[0]`` is recorded after the source states were shifted by one, so it holds
  the source's state at t=1 (cube pose off by millimetres), not the state frame 0 shows.

For each episode this script resets an env that has never stepped (no cached contacts)
with the episode's seed, checks that the state equals the raw demo's ``env_states[0]``
(the source of the conversion), and records that state and ``get_obs()``. It writes
``<replay stem>.first_obs.h5`` next to each input, one group per episode with the
replay's layout and a time axis of length 1:

    traj_k/obs          state replay: (1, D) float32; rgb replay: agent/extra/sensor_* groups
    traj_k/env_states   actors/..., articulations/... (1, ...)
    traj_k attrs        seed, recorded_first_state_diff (replay's env_states[0] vs the reset state)

scripts/export_demos.py replaces frame 0 with it. Run on the data machine with the
expert env (mani-skill 3.0.1) and the same Vulkan setup as the replays:

    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \\
    .venv/bin/python scripts/first_frame_obs.py <dir>/smoke_train.rgb.pd_ee_delta_pos.physx_cpu.h5 ...

The raw demo is found by ManiSkill's naming (``<name>.<obs>.<mode>.<backend>.h5`` was
converted from ``<name>.h5``); pass ``--raw`` when it lives elsewhere.
"""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import h5py
import mani_skill.envs  # noqa: F401  registers the envs
import numpy as np
from mani_skill.utils import common

RESET_STATE_ATOL = 1e-6


def first_state(group: h5py.Group) -> dict:
    return {key: first_state(value) if isinstance(value, h5py.Group) else value[0] for key, value in group.items()}


def max_diff(a, b) -> float:
    if isinstance(a, dict):
        assert a.keys() == b.keys(), f"state keys differ: {sorted(a)} vs {sorted(b)}"
        return max((max_diff(a[key], b[key]) for key in a), default=0.0)
    a, b = common.to_numpy(a).reshape(-1), np.asarray(b).reshape(-1)
    return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max()) if a.size else 0.0


def write(group: h5py.Group, key: str, value) -> None:
    if isinstance(value, dict):
        child = group.create_group(key, track_order=True)
        for name, item in value.items():
            write(child, name, item)
    else:
        group.create_dataset(key, data=common.to_numpy(value))


def raw_first_states(raw: Path) -> dict[int, dict]:
    meta = json.loads(raw.with_suffix(".json").read_text(encoding="utf-8"))
    with h5py.File(raw, "r") as file:
        return {episode["episode_seed"]: first_state(file[f"traj_{episode['episode_id']}"]["env_states"])
                for episode in meta["episodes"]}


def process(path: Path, raw: Path, overwrite: bool) -> None:
    output = path.with_name(path.stem + ".first_obs.h5")
    assert overwrite or not output.exists(), f"{output} exists; pass --overwrite to replace it"
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    env_info = meta["env_info"]
    source_states = raw_first_states(raw)
    env = gym.make(env_info["env_id"], **env_info["env_kwargs"])
    base = env.unwrapped
    temporary = output.with_name(output.name + ".tmp")
    recorded = []
    try:
        with h5py.File(path, "r") as source, h5py.File(temporary, "w") as out:
            for episode in meta["episodes"]:
                name = f"traj_{episode['episode_id']}"
                seed = episode["episode_seed"]
                env.reset(seed=seed, options=episode["reset_kwargs"].get("options"))
                state = base.get_state_dict()
                diff = max_diff(state, source_states[seed])
                assert diff <= RESET_STATE_ATOL, f"{path.name}/{name}: reset(seed={seed}) is {diff:.2e} from {raw.name}"
                group = out.create_group(name, track_order=True)
                group.attrs["seed"] = seed
                recorded.append(max_diff(state, first_state(source[name]["env_states"])))
                group.attrs["recorded_first_state_diff"] = recorded[-1]
                write(group, "obs", base.get_obs())
                write(group, "env_states", state)
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        env.close()
    print(f"{output}: {len(recorded)} episodes; reset matches {raw.name}; "
          f"replay's recorded env_states[0] was off by up to {max(recorded):.1e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("replays", nargs="+", type=Path, help="converted replay H5 files (JSON alongside)")
    parser.add_argument("--raw", type=Path, help="raw demo H5 the replays were converted from")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for path in args.replays:
        raw = args.raw or path.with_name(path.name.split(".")[0] + ".h5")
        process(path, raw, args.overwrite)


if __name__ == "__main__":
    main()
