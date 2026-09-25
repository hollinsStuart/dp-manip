#!/usr/bin/env python3
"""Merge ManiSkill rgb and state replays into one demo file for the teammates' DP trainers.

Target readers, both of which only open ``traj_N/obs`` and ``traj_N/actions``:

- hryang1130/VariDP ``dp/dp_lib.py::load_trajectories`` (``train.py --h5``);
- Seanzrui/7606-train-template ``tasks/<id>/data.py::read_trajectories`` (``--dataset``).

Both sort ``traj_N`` by N and take the first N episodes (``--max-episodes`` /
``--num-demos``), so episodes are renumbered in seed order and "first N" is the
nested subset of docs/final-plan.md §2.2. The file is a superset of ManiSkill's
native state replay; readers that do not know about ``obs_rgb`` ignore it::

    traj_i/obs              (T+1, D)        float32  obs_mode=state vector (privileged)
    traj_i/obs_rgb/rgb      (T+1, H, W, 3C) uint8    camera images, C cameras
    traj_i/obs_rgb/state    (T+1, P)        float32  agent + extra of obs_mode=rgb
    traj_i/actions          (T, A)          float32
    traj_i/success, terminated, truncated   (T,) bool
    traj_i/env_states/...                            copied from the rgb replay

``obs_rgb/rgb`` and ``obs_rgb/state`` equal ``obs["rgb"]`` and ``obs["state"]`` of
``FlattenRGBDObservationWrapper(env, rgb=True, depth=False)`` on an ``obs_mode="rgb"``
env: ManiSkill writes obs groups with ``track_order=True``, and they are flattened
here in that order, as ``common.flatten_state_dict`` does. ``obs`` holds object poses
and is for state-only training; an RGB policy must condition on ``obs_rgb`` only.

Inputs: two conversions of the same raw pd_joint_pos demos, ``--use-first-env-state
-c <mode> -o rgb`` and ``... -o state``. CPU physics is deterministic, so episodes
paired by seed must agree in actions, env_states, success flags and the agent part of
the observation. A state replay with ``--use-env-states`` does not work here: in
mani-skill 3.0.1 it records the step taken from each set state, not the state itself.

Episodes with an env reset in the middle are dropped and listed in the JSON. In 3.0.1
a control-mode conversion can save failed attempts and the final successful one as a
single episode (seen on PlugCharger), with a random action on each reset step; such
an episode shows a joint jump no controller step can make.
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
# Dropped by FlattenRGBDObservationWrapper before flattening the rest.
SENSOR_KEYS = ("sensor_data", "sensor_param")
ENV_STATE_ATOL = 1e-5
# Largest joint change allowed between two recorded steps. Panda joints move at most
# ~0.13 rad per 20 Hz control step; converted demos stay below 0.03, resets jump ~0.48.
MAX_QPOS_STEP = 0.2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def index_replay(path: Path, obs_mode: str) -> tuple[dict, dict[int, dict]]:
    """Read a replay's JSON and return its env_info and {seed: episode record}."""
    with path.with_suffix(".json").open(encoding="utf-8") as stream:
        meta = json.load(stream)
    env_kwargs = meta["env_info"]["env_kwargs"]
    assert env_kwargs["obs_mode"] == obs_mode, f"{path}: obs_mode is {env_kwargs['obs_mode']}, need {obs_mode}"
    # Older replays record the CPU backend as "cpu"; ManiSkill treats it as physx_cpu.
    assert env_kwargs.get("sim_backend") in ("cpu", "physx_cpu"), f"{path}: sim_backend is {env_kwargs.get('sim_backend')}"

    episodes = {}
    with h5py.File(path, "r") as source:
        names = {f"traj_{item['episode_id']}" for item in meta["episodes"]}
        assert set(source.keys()) == names, f"{path}: H5 groups and JSON episodes differ"
        for item in meta["episodes"]:
            name = f"traj_{item['episode_id']}"
            seed = item["reset_kwargs"]["seed"]
            assert seed == item["episode_seed"], f"{path}/{name}: reset seed differs from episode_seed"
            assert item["control_mode"] == env_kwargs["control_mode"], f"{path}/{name}: control mode differs"
            assert item["elapsed_steps"] == len(source[name]["actions"]), f"{path}/{name}: elapsed_steps != len(actions)"
            assert item["success"] is True, f"{path}/{name}: episode did not succeed"
            assert seed not in episodes, f"{path}: seed {seed} appears twice"
            episodes[seed] = {"path": path, "name": name, "meta": item}
    return meta["env_info"], episodes


def merge_index(paths: list[Path], obs_mode: str) -> tuple[dict, dict[int, dict]]:
    env_info, episodes = index_replay(paths[0], obs_mode)
    for path in paths[1:]:
        info, more = index_replay(path, obs_mode)
        assert info["env_id"] == env_info["env_id"], f"{path}: env_id differs from {paths[0]}"
        assert info["env_kwargs"]["control_mode"] == env_info["env_kwargs"]["control_mode"], \
            f"{path}: control mode differs from {paths[0]}"
        duplicate = episodes.keys() & more.keys()
        assert not duplicate, f"{path}: seeds already seen in another {obs_mode} replay: {sorted(duplicate)[:10]}"
        episodes.update(more)
    return env_info, episodes


def flatten_obs(group: h5py.Group, skip=SENSOR_KEYS) -> tuple[np.ndarray, list]:
    """Flatten a (T+1)-batched obs group in stored order, like common.flatten_state_dict.

    Returns the (T+1, P) float32 array and its layout as [(key path, width), ...].
    """
    columns, layout = [], []

    def visit(node: h5py.Group, prefix: str) -> None:
        for key, child in node.items():
            if not prefix and key in skip:
                continue
            path = f"{prefix}{key}"
            if isinstance(child, h5py.Group):
                visit(child, path + "/")
                continue
            value = child[()]
            assert value.ndim in (1, 2), f"{child.name}: {value.ndim}-d obs cannot be flattened"
            if value.size == 0:
                continue
            value = value.reshape(len(value), -1)
            columns.append(value.astype(np.float32))
            layout.append((path, value.shape[1]))

    visit(group, "")
    return np.concatenate(columns, axis=1), layout


def camera_images(group: h5py.Group) -> tuple[np.ndarray, list[str]]:
    """Concatenate every camera's rgb along channels, like FlattenRGBDObservationWrapper."""
    cameras = list(group["sensor_data"].keys())
    assert cameras, f"{group.name}: no cameras in sensor_data"
    images = [group["sensor_data"][camera]["rgb"][()] for camera in cameras]
    for camera, image in zip(cameras, images):
        assert image.dtype == np.uint8 and image.ndim == 4 and image.shape[-1] == 3, \
            f"{group.name}/sensor_data/{camera}/rgb: {image.dtype} {image.shape}"
    return np.concatenate(images, axis=-1), cameras


def reset_step(group: h5py.Group) -> tuple[int, float] | None:
    """Return (step, jump) of the first joint jump above MAX_QPOS_STEP, i.e. an env reset."""
    for articulation in group["env_states/articulations"].values():
        state = articulation[()]
        dof = (state.shape[1] - 13) // 2  # root pose 7 + root velocity 6, then qpos and qvel
        jumps = np.abs(np.diff(state[:, 13:13 + dof], axis=0)).max(axis=1)
        above = np.flatnonzero(jumps > MAX_QPOS_STEP)
        if above.size:
            return int(above[0]) + 1, float(jumps[above[0]])
    return None


def compare_env_states(rgb: h5py.Group, state: h5py.Group, where: str) -> float:
    """Return the max abs difference between two env_states trees, which must share keys and shapes."""
    worst = 0.0
    names = []
    rgb.visit(names.append)
    other = []
    state.visit(other.append)
    assert names == other, f"{where}: env_states layout differs between the rgb and state replays"
    for name in names:
        if isinstance(rgb[name], h5py.Dataset):
            a, b = rgb[name][()], state[name][()]
            assert a.shape == b.shape, f"{where}: env_states/{name} shape {a.shape} vs {b.shape}"
            if a.size:
                worst = max(worst, float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max()))
    return worst


def write_episode(out: h5py.File, index: int, rgb: h5py.Group, state: h5py.Group, where: str) -> dict:
    actions = rgb["actions"][()]
    length = len(actions)
    assert np.array_equal(actions, state["actions"][()]), f"{where}: actions differ between the rgb and state replays"
    for key in ("success", "terminated", "truncated"):
        assert np.array_equal(rgb[key][()], state[key][()]), f"{where}: {key} differs between the rgb and state replays"
    assert bool(rgb["success"][-1]), f"{where}: not successful at the last step"
    state_diff = compare_env_states(rgb["env_states"], state["env_states"], where)
    assert state_diff <= ENV_STATE_ATOL, f"{where}: env_states differ by {state_diff:.2e}"

    assert isinstance(state["obs"], h5py.Dataset), f"{where}: state replay obs is not a flat array"
    obs = state["obs"][()].astype(np.float32)
    images, cameras = camera_images(rgb["obs"])
    proprio, layout = flatten_obs(rgb["obs"])
    assert obs.ndim == 2 and len(obs) == length + 1, f"{where}: state obs is {obs.shape}, need ({length + 1}, D)"
    assert len(images) == length + 1 and len(proprio) == length + 1, f"{where}: rgb obs is not T+1 long"
    assert np.isfinite(obs).all() and np.isfinite(proprio).all() and np.isfinite(actions).all(), \
        f"{where}: non-finite values"
    # obs_mode=state flattens the same agent dict first, so its leading columns are the
    # agent part of obs_rgb/state; equality ties the two replays together step by step.
    agent_width = sum(width for key, width in layout if key.startswith("agent/"))
    assert agent_width > 0, f"{where}: rgb obs has no agent entries"
    assert np.allclose(obs[:, :agent_width], proprio[:, :agent_width], atol=ENV_STATE_ATOL), \
        f"{where}: agent obs differ between the rgb and state replays"

    group = out.create_group(f"traj_{index}", track_order=True)
    group.create_dataset("obs", data=obs)
    rgb_group = group.create_group("obs_rgb", track_order=True)
    rgb_group.create_dataset("rgb", data=images, chunks=(1, *images.shape[1:]),
                             compression="gzip", compression_opts=5)
    rgb_group.create_dataset("state", data=proprio)
    group.create_dataset("actions", data=actions.astype(np.float32))
    for key in ("success", "terminated", "truncated"):
        group.create_dataset(key, data=rgb[key][()].astype(bool))
    rgb.file.copy(rgb["env_states"], group, name="env_states")
    return {
        "length": length,
        "obs_dim": obs.shape[1],
        "proprio_layout": layout,
        "image_shape": list(images.shape[1:]),
        "cameras": cameras,
        "action_dim": actions.shape[1],
        "env_state_max_diff": state_diff,
    }


def verify(path: Path, expected: list[dict], handles: dict[Path, h5py.File]) -> None:
    """Re-read the file the way both teammate loaders do and compare with the sources."""
    with h5py.File(path, "r") as file:
        keys = sorted(file.keys(), key=lambda key: int(key.split("_")[-1]))
        assert keys == [f"traj_{i}" for i in range(len(expected))], f"{path}: traj groups are not traj_0..traj_{len(expected) - 1}"
        for key, item in zip(keys, expected):
            group = file[key]
            obs = np.asarray(group["obs"], dtype=np.float32)
            actions = np.asarray(group["actions"], dtype=np.float32)
            assert obs.ndim == 2 and actions.ndim == 2 and len(obs) == len(actions) + 1, f"{key}: loader shapes"
            assert bool(group["success"][-1]), f"{key}: loader success filter would drop it"
            rgb = handles[item["rgb"]["path"]][item["rgb"]["name"]]
            state = handles[item["state"]["path"]][item["state"]["name"]]
            assert np.array_equal(obs, state["obs"][()].astype(np.float32)), f"{key}: obs mismatch"
            assert np.array_equal(actions, rgb["actions"][()].astype(np.float32)), f"{key}: actions mismatch"
            assert np.array_equal(group["obs_rgb/rgb"][()], camera_images(rgb["obs"])[0]), f"{key}: rgb mismatch"
            assert np.array_equal(group["obs_rgb/state"][()], flatten_obs(rgb["obs"])[0]), f"{key}: obs_rgb/state mismatch"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rgb", nargs="+", type=Path, required=True,
                        help="rgb replay H5 files (JSON alongside), e.g. *.rgb.pd_ee_delta_pos.physx_cpu.h5")
    parser.add_argument("--state", nargs="+", type=Path, required=True,
                        help="state conversions (-o state) of the same raw demos; paired by seed")
    parser.add_argument("-o", "--output", type=Path, required=True, help="output .h5 path; a .json is written beside it")
    parser.add_argument("--split", choices=sorted(SEED_RANGES), required=True,
                        help="train = pool seeds 0-3999, val = validation demo seeds 4000-4999")
    parser.add_argument("--num-demos", type=int,
                        help="keep the first K demos by seed (400 for the pool, 50 for val); default all")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    args = parser.parse_args()

    for path in (args.output, args.output.with_suffix(".json")):
        assert args.overwrite or not path.exists(), f"{path} exists; pass --overwrite to replace it"

    env_info, rgb_episodes = merge_index(args.rgb, "rgb")
    state_info, state_episodes = merge_index(args.state, "state")
    assert state_info["env_id"] == env_info["env_id"], "rgb and state replays are of different envs"
    control_mode = env_info["env_kwargs"]["control_mode"]
    assert state_info["env_kwargs"]["control_mode"] == control_mode, "rgb and state replays use different control modes"
    missing = rgb_episodes.keys() ^ state_episodes.keys()
    assert not missing, f"seeds present in only one of the rgb and state replays: {sorted(missing)[:10]}"

    seeds = sorted(rgb_episodes)
    allowed = SEED_RANGES[args.split]
    outside = [seed for seed in seeds if seed not in allowed]
    assert not outside, f"seeds outside the {args.split} range {allowed.start}-{allowed.stop - 1}: {outside[:10]}"

    handles = {path: h5py.File(path, "r") for path in {*args.rgb, *args.state}}
    temporary = args.output.with_name(args.output.name + ".tmp")
    try:
        # Rejected seeds count as conversion failures: "first N" skips them (final-plan §2.2).
        rejected = []
        for seed in seeds:
            episode = rgb_episodes[seed]
            found = reset_step(handles[episode["path"]][episode["name"]])
            if found is not None:
                step, jump = found
                rejected.append({"seed": seed, "reason": f"env reset mid-episode: joint jump {jump:.3f} rad at step {step}"})
                print(f"drop seed {seed} ({episode['path'].name}/{episode['name']}): {rejected[-1]['reason']}")
        seeds = [seed for seed in seeds if seed not in {item["seed"] for item in rejected}]
        if args.num_demos is not None:
            assert len(seeds) >= args.num_demos, f"only {len(seeds)} usable demos, need {args.num_demos}"
            seeds = seeds[: args.num_demos]
        assert seeds, "no usable demos"
        selected = [{"seed": seed, "rgb": rgb_episodes[seed], "state": state_episodes[seed]} for seed in seeds]

        args.output.parent.mkdir(parents=True, exist_ok=True)
        stats = []
        with h5py.File(temporary, "w") as out:
            for index, item in enumerate(selected):
                rgb = handles[item["rgb"]["path"]][item["rgb"]["name"]]
                state = handles[item["state"]["path"]][item["state"]["name"]]
                where = f"seed {item['seed']} ({item['rgb']['path'].name}/{item['rgb']['name']})"
                stats.append(write_episode(out, index, rgb, state, where))
        for key in ("obs_dim", "proprio_layout", "image_shape", "cameras", "action_dim"):
            assert all(s[key] == stats[0][key] for s in stats), f"{key} differs between demos"
        verify(temporary, selected, handles)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        for handle in handles.values():
            handle.close()
    temporary.replace(args.output)

    lengths = [s["length"] for s in stats]
    episodes = []
    for index, item in enumerate(selected):
        record = dict(item["rgb"]["meta"])
        record["episode_id"] = index
        record.update(source_rgb_file=item["rgb"]["path"].name, source_rgb_episode_id=item["rgb"]["meta"]["episode_id"],
                      source_state_file=item["state"]["path"].name, source_state_episode_id=item["state"]["meta"]["episode_id"])
        episodes.append(record)
    meta = {
        # env_info is the rgb conversion's, so re-creating the env reproduces obs_rgb; traj_i/obs
        # comes from the state conversion, whose env_info is kept under dp_manip.state_env_info.
        "env_info": env_info,
        "dp_manip": {
            "layout": "traj_i/obs = obs_mode state; traj_i/obs_rgb/{rgb,state} = FlattenRGBDObservationWrapper(rgb=True, depth=False) of obs_mode rgb",
            "split": args.split,
            "num_demos": len(selected),
            "total_steps": int(sum(lengths)),
            "length_min_median_max": [min(lengths), int(np.median(lengths)), max(lengths)],
            "control_mode": control_mode,
            "action_dim": stats[0]["action_dim"],
            "obs_dim": stats[0]["obs_dim"],
            "obs_rgb_state_dim": sum(width for _, width in stats[0]["proprio_layout"]),
            "obs_rgb_state_layout": stats[0]["proprio_layout"],
            "obs_rgb_image_shape": stats[0]["image_shape"],
            "cameras": stats[0]["cameras"],
            "env_state_max_diff": max(s["env_state_max_diff"] for s in stats),
            "seeds": seeds,
            "rejected": rejected,
            "sources": {path.name: sha256(path) for path in [*args.rgb, *args.state]},
            "state_env_info": state_info,
            "sha256": sha256(args.output),
        },
        "episodes": episodes,
    }
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")

    summary = meta["dp_manip"]
    print(f"{args.output}: {len(selected)} demos, {summary['total_steps']} steps, seeds {seeds[0]}-{seeds[-1]}, "
          f"sha256 {summary['sha256'][:16]}")
    print(f"env {env_info['env_id']}, control mode {control_mode}, action {summary['action_dim']}; "
          f"obs {summary['obs_dim']}; obs_rgb/state {summary['obs_rgb_state_dim']}; "
          f"obs_rgb/rgb {summary['obs_rgb_image_shape']} from {summary['cameras']}; "
          f"env_states max diff {summary['env_state_max_diff']:.1e}")


if __name__ == "__main__":
    main()
