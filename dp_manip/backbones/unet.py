"""The canonical conditional 1D U-Net preserved as the first backbone."""

from __future__ import annotations

import torch

from ..conditional_unet1d import ConditionalUnet1D
from ..config import PolicyConfig
from .base import NoisePredictor


class UNetBackbone(NoisePredictor):
    """Flatten ``(B, To, Dobs)`` into the U-Net's FiLM global conditioning.

    The wrapped network is the original ``ConditionalUnet1D`` with unchanged
    dimensions, kernel size, and group count, all read from ``baseline.toml``
    through ``PolicyConfig``. This class only adapts the backbone contract; it
    does not re-encode observations.
    """

    def __init__(self, policy_cfg: PolicyConfig, *, obs_dim: int, action_dim: int):
        super().__init__()
        self.obs_horizon = policy_cfg.obs_horizon
        self.unet = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=policy_cfg.obs_horizon * obs_dim,
            diffusion_step_embed_dim=policy_cfg.diffusion_step_embed_dim,
            down_dims=policy_cfg.unet_dims,
            kernel_size=policy_cfg.kernel_size,
            n_groups=policy_cfg.n_groups,
        )

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        obs_features: torch.Tensor,
    ) -> torch.Tensor:
        if obs_features.ndim != 3:
            raise ValueError(
                f"expected observation features (B, To, Dobs), got {tuple(obs_features.shape)}"
            )
        condition = obs_features.flatten(start_dim=1)
        return self.unet(noisy_actions, timestep, global_cond=condition)
