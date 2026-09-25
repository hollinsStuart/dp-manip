"""Diffusion Policy agent (state or rgb observations, 1D conditional UNet, DDPM).

Adapted from ManiSkill examples/baselines/diffusion_policy/train.py and
train_rgbd.py (haosulab/ManiSkill@62ff3a5, Apache-2.0). Changes from the baseline:
- dimensions come from the dataset instead of an env object, so the agent can
  be built and checked without ManiSkill installed;
- the model works in normalized action space, and `ActionNormalizer` maps
  sampled actions back to the env's units in `get_action`;
- checkpoints carry the resolved config and normalizer statistics;
- state and rgb share one agent: observations go through `ObsEncoder`
  (identity flatten for state, PlainConv + state for rgb) into the UNet.

Observations are dicts, see dp_manip/obs_encoder.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from .conditional_unet1d import ConditionalUnet1D
from .config import PolicyConfig
from .data import ActionNormalizer
from .obs_encoder import ObsEncoder


class DiffusionPolicy(nn.Module):
    def __init__(self, cfg: PolicyConfig, obs_dim: int, act_dim: int, normalizer: ActionNormalizer,
                 image_shape: tuple[int, int, int] | None = None):
        """obs_dim is the state width; image_shape (H, W, C) adds the visual encoder."""
        super().__init__()
        self.obs_horizon = cfg.obs_horizon
        self.act_horizon = cfg.act_horizon
        self.pred_horizon = cfg.pred_horizon
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.image_shape = None if image_shape is None else tuple(image_shape)
        self.normalizer = normalizer

        self.obs_encoder = ObsEncoder(
            cfg.obs_horizon, obs_dim,
            image_channels=None if image_shape is None else image_shape[-1],
            visual_feature_dim=cfg.visual_feature_dim,
        )
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=act_dim,
            global_cond_dim=self.obs_encoder.out_dim,
            diffusion_step_embed_dim=cfg.diffusion_step_embed_dim,
            down_dims=cfg.unet_dims,
            n_groups=cfg.n_groups,
        )
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=cfg.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",  # baseline: large effect on performance
            clip_sample=True,  # samples are clipped to [-1, 1] -> actions must be normalized
            prediction_type="epsilon",
        )

    def compute_loss(self, obs_seq: dict[str, torch.Tensor], action_seq: torch.Tensor) -> torch.Tensor:
        """obs_seq values (B, obs_horizon, ...) raw; action_seq (B, pred_horizon, act_dim) normalized."""
        B = action_seq.shape[0]
        obs_cond = self.obs_encoder(obs_seq)
        noise = torch.randn((B, self.pred_horizon, self.act_dim), device=action_seq.device)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (B,), device=action_seq.device
        ).long()
        noisy = self.noise_scheduler.add_noise(action_seq, noise, timesteps)
        noise_pred = self.noise_pred_net(noisy, timesteps, global_cond=obs_cond)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def get_action(self, obs_seq: dict[str, torch.Tensor]) -> torch.Tensor:
        """obs_seq values (B, obs_horizon, ...) -> (B, act_horizon, act_dim) in env units."""
        state = obs_seq["state"]
        B = state.shape[0]
        obs_cond = self.obs_encoder(obs_seq)
        sample = torch.randn((B, self.pred_horizon, self.act_dim), device=state.device)
        # Inference uses all training diffusion steps, so set_timesteps is not needed.
        for k in self.noise_scheduler.timesteps:
            noise_pred = self.noise_pred_net(sample=sample, timestep=k, global_cond=obs_cond)
            sample = self.noise_scheduler.step(model_output=noise_pred, timestep=k, sample=sample).prev_sample
        start = self.obs_horizon - 1
        chunk = sample[:, start : start + self.act_horizon]
        return self.normalizer.unnormalize(chunk)


def num_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def save_checkpoint(path, *, policy: DiffusionPolicy, ema_policy: DiffusionPolicy,
                    config: dict, iteration: int, extra: dict | None = None) -> None:
    torch.save({
        "policy": policy.state_dict(),
        "ema_policy": ema_policy.state_dict(),
        "normalizer": policy.normalizer.state_dict(),
        "obs_dim": policy.obs_dim,
        "act_dim": policy.act_dim,
        "image_shape": policy.image_shape,
        "config": config,
        "iteration": iteration,
        "extra": extra or {},
    }, path)


def load_checkpoint(path, device: torch.device, use_ema: bool = True):
    """Return (policy, config dict, checkpoint dict). EMA weights by default, as the baseline evaluates."""
    from .config import from_dict

    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = from_dict(ckpt["config"])
    normalizer = ActionNormalizer.from_state_dict(ckpt["normalizer"])
    policy = DiffusionPolicy(cfg.policy, ckpt["obs_dim"], ckpt["act_dim"], normalizer,
                             image_shape=ckpt.get("image_shape")).to(device)
    policy.load_state_dict(ckpt["ema_policy" if use_ema else "policy"])
    policy.eval()
    return policy, cfg, ckpt
