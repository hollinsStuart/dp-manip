"""ManiSkill demo loading, action normalization and DP training windows.

Episode boundaries come only from the JSON `episodes[].episode_id` and the
matching H5 `traj_<id>` group; per-step terminated/truncated flags are ignored
because they can turn true before the group ends (see AGENT.md §4).

Two observation modes, read from the files of scripts/export_demos.py:
- "state": `traj_i/obs`, the privileged obs_mode=state vector (also plain
  ManiSkill state replays);
- "rgb": `traj_i/obs_rgb/state` (agent + extra, no object poses) and
  `traj_i/obs_rgb/rgb` (T+1, H, W, 3*C) uint8, matching the eval env's
  FlattenRGBDObservationWrapper(rgb=True, depth=False) output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch


@dataclass
class Episode:
    episode_id: int
    seed: int
    obs: np.ndarray  # (T+1, obs_dim) float32; the state part in rgb mode
    actions: np.ndarray  # (T, act_dim) float32
    images: np.ndarray | None = None  # (T+1, H, W, C) uint8, rgb mode only


@dataclass
class DemoSet:
    env_id: str
    control_mode: str
    obs_mode: str
    episodes: list[Episode]

    @property
    def seeds(self) -> list[int]:
        return [e.seed for e in self.episodes]

    @property
    def obs_dim(self) -> int:
        return self.episodes[0].obs.shape[1]

    @property
    def act_dim(self) -> int:
        return self.episodes[0].actions.shape[1]

    @property
    def image_shape(self) -> tuple[int, int, int] | None:
        """(H, W, C) of the camera images, None for state demos."""
        images = self.episodes[0].images
        return None if images is None else tuple(images.shape[1:])


def _entries(h5_path: Path, num_demos: int | None) -> tuple[dict, list[dict]]:
    meta = json.loads(h5_path.with_suffix(".json").read_text())
    entries = sorted(meta["episodes"], key=lambda e: e["episode_id"])
    if num_demos is not None:
        if not 0 < num_demos <= len(entries):
            raise ValueError(f"num_demos={num_demos} but file has {len(entries)} episodes")
        entries = entries[:num_demos]
    return meta, entries


def demo_seeds(h5_path: str | Path, num_demos: int | None = None) -> list[int]:
    """Reset seeds of the first num_demos episodes, from the JSON only (no arrays read)."""
    return [e["episode_seed"] for e in _entries(Path(h5_path), num_demos)[1]]


def load_demos(h5_path: str | Path, num_demos: int | None = None, obs_mode: str = "state") -> DemoSet:
    h5_path = Path(h5_path)
    meta, entries = _entries(h5_path, num_demos)
    env_kwargs = meta["env_info"]["env_kwargs"]
    if obs_mode == "state":
        # Plain state replays and exported demos both record obs_mode=state here.
        file_obs_mode = env_kwargs["obs_mode"]
    elif obs_mode == "rgb":
        # The JSON of an export describes traj_i/obs (state); rgb lives in obs_rgb.
        file_obs_mode = "rgb"
    else:
        raise ValueError(f"unsupported obs_mode {obs_mode!r}")

    episodes = []
    with h5py.File(h5_path, "r") as f:
        for entry in entries:
            if not entry.get("success", False):
                raise ValueError(f"episode {entry['episode_id']} is not marked successful")
            name = f"traj_{entry['episode_id']}"
            group = f[name]
            images = None
            if obs_mode == "rgb":
                if "obs_rgb" not in group:
                    raise ValueError(f"{h5_path}/{name} has no obs_rgb group; rgb training needs "
                                     "the output of scripts/export_demos.py")
                obs = np.asarray(group["obs_rgb/state"], dtype=np.float32)
                images = np.asarray(group["obs_rgb/rgb"])
                if images.dtype != np.uint8 or images.ndim != 4 or images.shape[0] != obs.shape[0]:
                    raise ValueError(f"{name}: expected (T+1, H, W, C) uint8 images, got {images.dtype} {images.shape}")
            else:
                obs = np.asarray(group["obs"], dtype=np.float32)
            actions = np.asarray(group["actions"], dtype=np.float32)
            if obs.ndim != 2:
                raise ValueError(f"expected flat state obs, got shape {obs.shape}")
            if obs.shape[0] != actions.shape[0] + 1:
                raise ValueError(f"{name}: obs {obs.shape} vs actions {actions.shape}")
            if not (np.isfinite(obs).all() and np.isfinite(actions).all()):
                raise ValueError(f"{name}: non-finite values")
            episodes.append(Episode(entry["episode_id"], entry["episode_seed"], obs, actions, images))

    return DemoSet(
        env_id=meta["env_info"]["env_id"],
        control_mode=env_kwargs["control_mode"],
        obs_mode=file_obs_mode,
        episodes=episodes,
    )


class ActionNormalizer:
    """Per-dimension min-max scaling of actions to [-1, 1].

    The DDPM scheduler clips samples to [-1, 1], so actions must live in that
    range. Absolute joint targets (pd_joint_pos) are in radians and do not.
    """

    def __init__(self, low: np.ndarray, high: np.ndarray, eps: float = 1e-4):
        low = np.asarray(low, dtype=np.float32)
        high = np.asarray(high, dtype=np.float32)
        # Constant dims map to 0 instead of dividing by ~0.
        flat = (high - low) < eps
        center = (high + low) / 2
        self.low = np.where(flat, center - 1, low).astype(np.float32)
        self.high = np.where(flat, center + 1, high).astype(np.float32)

    @classmethod
    def fit(cls, demos: DemoSet) -> "ActionNormalizer":
        actions = np.concatenate([e.actions for e in demos.episodes])
        return cls(actions.min(axis=0), actions.max(axis=0))

    def normalize(self, x):
        low, high = self._like(x)
        return 2 * (x - low) / (high - low) - 1

    def unnormalize(self, x):
        low, high = self._like(x)
        return (x + 1) / 2 * (high - low) + low

    def _like(self, x):
        if isinstance(x, torch.Tensor):
            return (torch.as_tensor(self.low, device=x.device, dtype=x.dtype),
                    torch.as_tensor(self.high, device=x.device, dtype=x.dtype))
        return self.low, self.high

    def state_dict(self) -> dict[str, list[float]]:
        return {"low": self.low.tolist(), "high": self.high.tolist()}

    @classmethod
    def from_state_dict(cls, d: dict[str, list[float]]) -> "ActionNormalizer":
        return cls(np.asarray(d["low"]), np.asarray(d["high"]), eps=0.0)


def is_delta_control(control_mode: str) -> bool:
    return "delta" in control_mode or control_mode == "base_pd_joint_vel_arm_pd_joint_vel"


def build_windows(demos: DemoSet, obs_horizon: int, pred_horizon: int):
    """All (obs history, action chunk) training windows, as in ManiSkill's DP baseline.

    For a window starting at s the policy sees obs[s : s+obs_horizon] and predicts
    actions[s : s+pred_horizon]; the current time is t = s + obs_horizon - 1.
    Before the episode start, obs and actions repeat index 0. After the end, the
    robot is told to stay still: absolute modes repeat the last action, delta
    modes use a zero arm delta with the last gripper command.

    Unlike the baseline (whose range excludes it), the window with t = T-1 is
    included so the final real action is also a training target.

    Observations are not copied per window: returns `frame_idx` (N, obs_horizon)
    int64 rows into the episodes' frames concatenated in order (episode i
    starts at sum of the earlier episodes' T+1), and the action windows
    (N, pred_horizon, act_dim) float32.
    """
    delta = is_delta_control(demos.control_mode)
    idx_windows, act_windows = [], []
    offset = 0
    for ep in demos.episodes:
        T = ep.actions.shape[0]
        pad_before = obs_horizon - 1
        pad_after = pred_horizon - obs_horizon
        still = ep.actions[-1:].copy()
        if delta:
            still[:, :-1] = 0
        acts = np.concatenate([
            np.repeat(ep.actions[:1], pad_before, axis=0),
            ep.actions,
            np.repeat(still, pad_after, axis=0),
        ])
        # Padded index i is original index i - pad_before, so the window whose
        # current time is t starts at padded index t.
        for t in range(T):
            idx_windows.append(offset + np.maximum(np.arange(t - pad_before, t + 1), 0))
            act_windows.append(acts[t : t + pred_horizon])
        offset += T + 1
    idx_arr = np.stack(idx_windows).astype(np.int64)
    act_arr = np.stack(act_windows).astype(np.float32)
    assert idx_arr.shape[1] == obs_horizon and act_arr.shape[1] == pred_horizon
    return idx_arr, act_arr


class WindowSampler:
    """Holds every frame and training window on `device` and samples batches with replacement.

    Frames are stored once (images as uint8); a batch gathers its observation
    histories through the window index. Batches are dicts in the layout of
    dp_manip.obs_encoder: {"state": (B, Th, P)[, "rgb": (B, Th, H, W, C)]}.

    The baseline used an epoch sampler with drop_last=True, which yields no
    batches at all when there are fewer windows than batch_size (10 PickCube
    demos give 726 windows < 1024).
    """

    def __init__(self, demos: DemoSet, normalizer: ActionNormalizer,
                 obs_horizon: int, pred_horizon: int, device: torch.device):
        idx, acts = build_windows(demos, obs_horizon, pred_horizon)
        self.frame_idx = torch.from_numpy(idx).to(device)
        self.actions = torch.from_numpy(normalizer.normalize(acts)).to(device)
        self.state = torch.from_numpy(np.concatenate([e.obs for e in demos.episodes])).to(device)
        self.images = None
        if demos.image_shape is not None:
            # Filled episode by episode so host memory holds no second full copy.
            num_frames = self.state.shape[0]
            self.images = torch.empty((num_frames, *demos.image_shape), dtype=torch.uint8, device=device)
            row = 0
            for ep in demos.episodes:
                self.images[row : row + len(ep.images)] = torch.from_numpy(ep.images)
                row += len(ep.images)
        self.device = device

    def __len__(self) -> int:
        return self.frame_idx.shape[0]

    @property
    def image_bytes(self) -> int:
        return 0 if self.images is None else self.images.numel()

    def sample(self, batch_size: int, generator: torch.Generator | None = None):
        idx = torch.randint(len(self), (batch_size,), generator=generator).to(self.device)
        return self.window(idx)

    def window(self, idx: torch.Tensor):
        """Observation dict and normalized action chunk of the windows `idx`."""
        frames = self.frame_idx[idx]
        obs = {"state": self.state[frames]}
        if self.images is not None:
            obs["rgb"] = self.images[frames]
        return obs, self.actions[idx]
