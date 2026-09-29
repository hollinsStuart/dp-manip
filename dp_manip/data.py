"""Lazy RGB trajectory loading for the ``maniskill-demogen`` export schema.

Only low-dimensional observations and actions are scanned eagerly. By default
compressed RGB frames remain in HDF5 and are read by DataLoader workers per
temporal window, keeping large demonstration sets from consuming several GB of
RAM. With ``preload=True`` the selected episodes are decoded into RAM once;
every chunk is then decompressed a single time instead of once per window.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from torch.utils.data import Dataset


@dataclass(frozen=True)
class EpisodeInfo:
    group: str
    episode_id: int
    seed: int
    length: int


@dataclass(frozen=True)
class DatasetInfo:
    path: Path
    env_id: str
    control_mode: str
    episodes: tuple[EpisodeInfo, ...]
    image_shape: tuple[int, int, int]
    proprio_dim: int
    action_dim: int
    cameras: tuple[str, ...]
    rgb_env_info: dict[str, Any] | None

    @property
    def seeds(self) -> list[int]:
        return [episode.seed for episode in self.episodes]

    @property
    def num_cameras(self) -> int:
        return self.image_shape[-1] // 3

    @property
    def num_transitions(self) -> int:
        return sum(episode.length for episode in self.episodes)


@dataclass(frozen=True)
class NormalizationStats:
    proprio_mean: np.ndarray
    proprio_std: np.ndarray
    action_low: np.ndarray
    action_high: np.ndarray

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "proprio_mean": self.proprio_mean.tolist(),
            "proprio_std": self.proprio_std.tolist(),
            "action_low": self.action_low.tolist(),
            "action_high": self.action_high.tolist(),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, list[float]]) -> "NormalizationStats":
        # Version-1 checkpoints used ``state_*`` for non-privileged
        # proprioception. Keep that vocabulary confined to this adapter.
        raw = dict(raw)
        for legacy, canonical in (
            ("state_mean", "proprio_mean"),
            ("state_std", "proprio_std"),
        ):
            if legacy in raw:
                if canonical in raw:
                    raise ValueError(f"normalization contains both {legacy!r} and {canonical!r}")
                raw[canonical] = raw.pop(legacy)
        expected = {"proprio_mean", "proprio_std", "action_low", "action_high"}
        if set(raw) != expected:
            raise ValueError(f"normalization keys must be {sorted(expected)}, got {sorted(raw)}")
        return cls(**{key: np.asarray(value, dtype=np.float32) for key, value in raw.items()})


def export_info_path(path: Path) -> Path:
    return path.with_name(path.stem + ".export_info.json")


def _episode_seed(entry: dict[str, Any], sidecar: Path) -> int:
    seed = int(entry.get("episode_seed", entry.get("reset_kwargs", {}).get("seed", -1)))
    if seed < 0:
        raise ValueError(f"{sidecar}: episode {entry.get('episode_id')} has no reset seed")
    return seed


def _select_episode_entries(
    entries: list[dict[str, Any]], sidecar: Path, num_demos: int | None
) -> list[dict[str, Any]]:
    """Return the canonical seed-ordered prefix used by data-size experiments.

    Export order and ``episode_id`` are not demonstration identities: the same
    seed pool may be re-exported with different trajectory numbering. Ordering
    by demonstration seed makes every N-demo subset a prefix of every larger
    subset, so ``N1 < N2`` implies ``seeds(N1)`` is a subset of ``seeds(N2)``
    regardless of how the exporter numbered or listed the trajectories.
    Demonstration seeds must be unique, otherwise the seed prefix is ambiguous.
    """
    episode_ids = [int(entry["episode_id"]) for entry in entries]
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError(f"{sidecar}: episode_id values must be unique")
    seeds = [_episode_seed(entry, sidecar) for entry in entries]
    duplicate_seeds = sorted(seed for seed, count in Counter(seeds).items() if count > 1)
    if duplicate_seeds:
        raise ValueError(f"{sidecar}: duplicate demonstration seeds: {duplicate_seeds[:10]}")
    ordered = sorted(entries, key=lambda entry: _episode_seed(entry, sidecar))
    if num_demos is None:
        return ordered
    if not 0 < num_demos <= len(ordered):
        raise ValueError(f"requested {num_demos} demos, but {sidecar} contains {len(ordered)}")
    return ordered[:num_demos]


def read_dataset_info(path: str | Path, num_demos: int | None = None) -> DatasetInfo:
    """Validate an exported dataset and return lightweight episode metadata."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    sidecar = path.with_suffix(".json")
    if not sidecar.is_file():
        raise FileNotFoundError(f"dataset metadata is missing: {sidecar}")
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    env_info = meta["env_info"]
    env_kwargs = env_info["env_kwargs"]
    entries = _select_episode_entries(meta["episodes"], sidecar, num_demos)

    episodes: list[EpisodeInfo] = []
    image_shape: tuple[int, int, int] | None = None
    proprio_dim: int | None = None
    action_dim: int | None = None
    with h5py.File(path, "r") as file:
        for entry in entries:
            episode_id = int(entry["episode_id"])
            group_name = f"traj_{episode_id}"
            if group_name not in file:
                raise ValueError(f"{path}: JSON references missing group {group_name}")
            group = file[group_name]
            for key in ("obs_rgb/rgb", "obs_rgb/state", "actions"):
                if key not in group:
                    raise ValueError(f"{path}/{group_name}: missing {key}; use maniskill-demogen export output")
            images = group["obs_rgb/rgb"]
            proprio = group["obs_rgb/state"]
            actions = group["actions"]
            if images.dtype != np.uint8 or images.ndim != 4 or images.shape[-1] % 3:
                raise ValueError(f"{images.name}: expected uint8 (T+1,H,W,3*C), got {images.dtype} {images.shape}")
            if (
                proprio.ndim != 2
                or actions.ndim != 2
                or len(proprio) != len(actions) + 1
                or len(images) != len(actions) + 1
            ):
                raise ValueError(f"{path}/{group_name}: RGB/state/actions are not aligned as T+1/T")
            current_image_shape = tuple(int(value) for value in images.shape[1:])
            current_proprio_dim = int(proprio.shape[1])
            current_action_dim = int(actions.shape[1])
            if image_shape is None:
                image_shape, proprio_dim, action_dim = current_image_shape, current_proprio_dim, current_action_dim
            elif (image_shape, proprio_dim, action_dim) != (
                current_image_shape,
                current_proprio_dim,
                current_action_dim,
            ):
                raise ValueError(f"{path}/{group_name}: observation or action dimensions changed between episodes")
            if "success" in group and not bool(group["success"][-1]):
                raise ValueError(f"{path}/{group_name}: exported episode is not successful")
            episodes.append(
                EpisodeInfo(group_name, episode_id, _episode_seed(entry, sidecar), len(actions))
            )

    if not episodes or image_shape is None or proprio_dim is None or action_dim is None:
        raise ValueError(f"{path}: no usable episodes")

    cameras: tuple[str, ...] = tuple(f"camera_{index}" for index in range(image_shape[-1] // 3))
    rgb_env_info = None
    info_path = export_info_path(path)
    if info_path.is_file():
        export = json.loads(info_path.read_text(encoding="utf-8"))
        cameras = tuple(export.get("cameras", cameras))
        rgb_env_info = export.get("rgb_env_info")
        if tuple(export.get("obs_rgb_image_shape", image_shape)) != image_shape:
            raise ValueError(f"{info_path}: image shape does not match HDF5")
    if len(cameras) != image_shape[-1] // 3:
        raise ValueError(f"{path}: {len(cameras)} camera names but image has {image_shape[-1]} channels")

    return DatasetInfo(
        path=path,
        env_id=env_info["env_id"],
        control_mode=env_kwargs["control_mode"],
        episodes=tuple(episodes),
        image_shape=image_shape,
        proprio_dim=proprio_dim,
        action_dim=action_dim,
        cameras=cameras,
        rgb_env_info=rgb_env_info,
    )


def compute_normalization(info: DatasetInfo, epsilon: float = 1e-3) -> NormalizationStats:
    """Compute proprioception z-score and action min/max from training demos only."""
    count = 0
    proprio_sum = np.zeros(info.proprio_dim, dtype=np.float64)
    proprio_sq_sum = np.zeros(info.proprio_dim, dtype=np.float64)
    action_low = np.full(info.action_dim, np.inf, dtype=np.float64)
    action_high = np.full(info.action_dim, -np.inf, dtype=np.float64)
    with h5py.File(info.path, "r") as file:
        for episode in info.episodes:
            group = file[episode.group]
            proprio = np.asarray(group["obs_rgb/state"][: episode.length], dtype=np.float64)
            actions = np.asarray(group["actions"], dtype=np.float64)
            if not (np.isfinite(proprio).all() and np.isfinite(actions).all()):
                raise ValueError(f"{info.path}/{episode.group}: non-finite proprio or action")
            count += len(proprio)
            proprio_sum += proprio.sum(axis=0)
            proprio_sq_sum += np.square(proprio).sum(axis=0)
            action_low = np.minimum(action_low, actions.min(axis=0))
            action_high = np.maximum(action_high, actions.max(axis=0))
    mean = proprio_sum / count
    variance = np.maximum(proprio_sq_sum / count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std[std < epsilon] = 1.0
    flat = action_high - action_low < 1e-4
    center = (action_high + action_low) / 2
    action_low[flat], action_high[flat] = center[flat] - 1.0, center[flat] + 1.0
    return NormalizationStats(
        proprio_mean=mean.astype(np.float32),
        proprio_std=std.astype(np.float32),
        action_low=action_low.astype(np.float32),
        action_high=action_high.astype(np.float32),
    )


# Per-episode arrays a window is read from, as HDF5 paths inside ``traj_<i>``.
EPISODE_KEYS = ("obs_rgb/rgb", "obs_rgb/state", "actions")


class RGBWindowDataset(Dataset):
    """Episode-local observation/action windows with repeated boundary padding.

    ``preload`` decodes every selected episode into RAM up front. Windows are
    bit-identical to lazy reads; build the dataset before DataLoader workers
    fork so they share the arrays copy-on-write instead of copying them.
    """

    def __init__(self, info: DatasetInfo, obs_horizon: int, pred_horizon: int, *, preload: bool = False):
        self.info = info
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.index = [
            (episode_index, timestep)
            for episode_index, episode in enumerate(info.episodes)
            for timestep in range(episode.length)
        ]
        self._file: h5py.File | None = None
        self._episodes: list[dict[str, np.ndarray]] | None = None
        if preload:
            # Whole-episode reads decompress each chunk once.
            with h5py.File(info.path, "r") as file:
                self._episodes = [
                    {key: file[episode.group][key][()] for key in EPISODE_KEYS}
                    for episode in info.episodes
                ]

    def __len__(self) -> int:
        return len(self.index)

    @property
    def preloaded_bytes(self) -> int:
        """Bytes held in RAM by ``preload`` (0 for lazy reads)."""
        if self._episodes is None:
            return 0
        return sum(array.nbytes for arrays in self._episodes for array in arrays.values())

    def _handle(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.info.path, "r")
        return self._file

    @staticmethod
    def _read_frames(dataset: h5py.Dataset | np.ndarray, indices: np.ndarray) -> np.ndarray:
        lower, upper = int(indices.min()), int(indices.max())
        block = dataset[lower : upper + 1]
        return np.asarray(block[indices - lower])

    def __getitem__(self, item: int) -> dict[str, np.ndarray]:
        episode_index, timestep = self.index[item]
        episode = self.info.episodes[episode_index]
        group = (
            self._episodes[episode_index]
            if self._episodes is not None
            else self._handle()[episode.group]
        )

        obs_indices = np.arange(timestep - self.obs_horizon + 1, timestep + 1)
        obs_indices = np.clip(obs_indices, 0, episode.length)
        images = self._read_frames(group["obs_rgb/rgb"], obs_indices)
        proprio = self._read_frames(group["obs_rgb/state"], obs_indices).astype(np.float32)
        images = np.transpose(images, (0, 3, 1, 2))

        action_indices = np.arange(
            timestep - self.obs_horizon + 1,
            timestep - self.obs_horizon + 1 + self.pred_horizon,
        )
        clipped = np.clip(action_indices, 0, episode.length - 1)
        actions = self._read_frames(group["actions"], clipped).astype(np.float32)
        return {"rgb": images, "proprio": proprio, "actions": actions}

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __getstate__(self) -> dict[str, Any]:
        attributes = self.__dict__.copy()
        attributes["_file"] = None
        return attributes

    def __del__(self) -> None:
        self.close()
