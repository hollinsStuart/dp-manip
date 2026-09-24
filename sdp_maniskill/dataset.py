"""Stanford DP low-dim dataset for files written by scripts/export_stanford_dp.py.

Extends upstream ``RobomimicReplayLowdimDataset`` (real-stanford/diffusion_policy)
where it differs from docs/final-plan.md:

- validation demos come from a separate file (§2.2 ②), not a ``val_ratio`` split;
- obs normalization is per-dim z-score, dims with std < 1e-3 are only centred (§3);
  upstream divides every dim by one global max-abs;
- action normalization is per-dim min-max to [-1, 1] (§3); upstream uses identity
  when ``abs_action`` is False;
- both are fitted on the training file only, which holds exactly the first N demos;
- the files' split, env, control mode, demo count and dims are checked against the
  task config, so a wrong path or a stale ``obs_dim`` fails at load time.
"""

import copy
import json
from typing import Optional, Sequence

import h5py
import numpy as np
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler
from diffusion_policy.dataset.robomimic_replay_lowdim_dataset import (
    RobomimicReplayLowdimDataset,
    _data_to_obs,
)
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer


def _check_file(path: str, split: str, env_id: str, control_mode: str, num_demos: Optional[int]) -> np.ndarray:
    with h5py.File(path, "r") as file:
        attrs = file["data"].attrs
        env_name = json.loads(attrs["env_args"])["env_name"]
        assert attrs["split"] == split, f"{path}: split is {attrs['split']}, expected {split}"
        assert env_name == env_id, f"{path}: env is {env_name}, expected {env_id}"
        assert attrs["control_mode"] == control_mode, \
            f"{path}: control mode is {attrs['control_mode']}, expected {control_mode}"
        if num_demos is not None:
            assert len(file["data"]) == num_demos, f"{path}: {len(file['data'])} demos, expected {num_demos}"
        return np.asarray(attrs["seeds"])


def _load_buffer(path: str, obs_keys: Sequence[str]) -> ReplayBuffer:
    buffer = ReplayBuffer.create_empty_numpy()
    with h5py.File(path, "r") as file:
        demos = file["data"]
        for i in range(len(demos)):
            demo = demos[f"demo_{i}"]
            buffer.add_episode(_data_to_obs(
                raw_obs=demo["obs"],
                raw_actions=demo["actions"][:].astype(np.float32),
                obs_keys=obs_keys,
                abs_action=False,
                rotation_transformer=None))
    return buffer


class ManiSkillLowdimDataset(RobomimicReplayLowdimDataset):
    def __init__(self,
            dataset_path: str,
            env_id: str,
            control_mode: str,
            obs_dim: int,
            action_dim: int,
            num_demos: Optional[int] = None,
            val_dataset_path: Optional[str] = None,
            horizon=1,
            pad_before=0,
            pad_after=0,
            obs_keys: Sequence[str] = ("state",),
            obs_std_eps: float = 1e-3,
        ):
        train_seeds = _check_file(dataset_path, "train", env_id, control_mode, num_demos)
        super().__init__(
            dataset_path=dataset_path,
            horizon=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            obs_keys=list(obs_keys),
            abs_action=False,
            val_ratio=0.0)
        assert self.replay_buffer["obs"].shape[-1] == obs_dim, \
            f"{dataset_path}: obs dim {self.replay_buffer['obs'].shape[-1]}, task config says {obs_dim}"
        assert self.replay_buffer["action"].shape[-1] == action_dim, \
            f"{dataset_path}: action dim {self.replay_buffer['action'].shape[-1]}, task config says {action_dim}"

        self.val_replay_buffer = None
        if val_dataset_path is not None:
            val_seeds = _check_file(val_dataset_path, "val", env_id, control_mode, None)
            overlap = np.intersect1d(train_seeds, val_seeds)
            assert overlap.size == 0, f"train and val share seeds {overlap[:10]}"
            self.val_replay_buffer = _load_buffer(val_dataset_path, list(obs_keys))
            assert self.val_replay_buffer["obs"].shape[-1] == obs_dim
            assert self.val_replay_buffer["action"].shape[-1] == action_dim
        self.obs_std_eps = obs_std_eps

    def get_validation_dataset(self):
        if self.val_replay_buffer is None:
            return super().get_validation_dataset()  # empty: val_ratio is 0
        val_set = copy.copy(self)
        val_set.replay_buffer = self.val_replay_buffer
        val_set.train_mask = np.ones(self.val_replay_buffer.n_episodes, dtype=bool)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.val_replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=val_set.train_mask)
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        normalizer["action"] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer["action"], mode="limits", output_min=-1.0, output_max=1.0)
        normalizer["obs"] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer["obs"], mode="gaussian", range_eps=self.obs_std_eps)
        return normalizer
