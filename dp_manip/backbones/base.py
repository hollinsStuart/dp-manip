"""Noise-prediction backbone interface of the shared diffusion policy."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class NoisePredictor(nn.Module, ABC):
    """Predict diffusion noise from noisy actions and shared observation features.

    Every backbone receives the same inputs:

    ```text
    noisy_actions: (B, Tp, action_dim)  noisy action sequence to denoise
    timestep:      (B,)                 integer diffusion step
    obs_features:  (B, To, Dobs)        ObservationEncoder output, unchanged
    returns:       (B, Tp, action_dim)  predicted noise
    ```

    Backbones decide themselves how to consume ``obs_features``: flatten it into
    a global conditioning vector, keep it as condition tokens, pool it, and so
    on. They must never re-encode RGB or proprioception, because every backbone
    experiment shares one visual encoder.
    """

    @abstractmethod
    def forward(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        obs_features: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError
