"""ManiSkill evaluation environments (CPU physics, vectorized with gymnasium).

Adapted from ManiSkill examples/baselines/diffusion_policy/diffusion_policy/make_env.py
(haosulab/ManiSkill@62ff3a5, Apache-2.0), CPU branch only, for mani-skill 3.0.1:
- rendering is disabled (render_backend="none") unless a video directory is given,
  so state-only evaluation does not need a working Vulkan device;
- RecordEpisode in 3.0.1 has no source_type/source_desc arguments.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from .config import Config


def env_kwargs(cfg: Config, render: bool = False) -> dict[str, Any]:
    kwargs: dict[str, Any] = dict(
        obs_mode=cfg.task.obs_mode,
        control_mode=cfg.task.control_mode,
        reward_mode="sparse",
        sim_backend=cfg.task.sim_backend,
        max_episode_steps=cfg.task.max_episode_steps,
    )
    if render:
        kwargs.update(render_mode="rgb_array", render_backend="cpu",
                      human_render_camera_configs=dict(shader_pack="default"))
    else:
        kwargs.update(render_backend="none")
    return kwargs


def make_eval_envs(cfg: Config, num_envs: int, video_dir: str | None = None):
    """Vector env whose observations are stacked to (obs_horizon, obs_dim) per env.

    Episodes never terminate early (ignore_terminations) and all run for
    max_episode_steps, so every sub-env truncates on the same step. Only the
    first sub-env records video.
    """
    if cfg.task.sim_backend != "physx_cpu":
        raise NotImplementedError("only physx_cpu evaluation is wired up")
    import mani_skill.envs  # noqa: F401  (registers env ids)
    from mani_skill.utils.wrappers import CPUGymWrapper, FrameStack, RecordEpisode

    obs_horizon = cfg.policy.obs_horizon

    def make(index: int):
        record = video_dir is not None and index == 0

        def thunk():
            env = gym.make(cfg.task.env_id, reconfiguration_freq=1, **env_kwargs(cfg, render=record))
            env = FrameStack(env, num_stack=obs_horizon)
            env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
            if record:
                env = RecordEpisode(env, output_dir=video_dir, save_trajectory=False, info_on_video=True)
            return env

        return thunk

    thunks = [make(i) for i in range(num_envs)]
    if num_envs == 1:
        return gym.vector.SyncVectorEnv(thunks)
    return gym.vector.AsyncVectorEnv(thunks, context="forkserver")
