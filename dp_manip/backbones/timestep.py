"""Timestep embedding shared by the Transformer and MLP backbones.

The canonical UNet keeps its own vendored ``SinusoidalPosEmb`` inside
``conditional_unet1d.py`` (unmodified). The backbones migrated from VariDP use
the donor embedding from ``VariDP/dp/dp_lib.py``, which is numerically
equivalent: ``exp(-log(10000) * i / max(half - 1, 1))`` for ``i`` in
``[0, dim // 2)``, concatenating ``sin`` and ``cos``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    """Ported from ``VariDP/dp/dp_lib.py`` (official diffusion_policy embedding).

    Maps an integer diffusion step tensor ``(B,)`` to ``(B, dim)`` features.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=timesteps.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        arguments = timesteps.float()[:, None] * frequencies[None]
        return torch.cat([arguments.sin(), arguments.cos()], dim=-1)


def expand_timesteps(timestep, batch_size: int, device: torch.device) -> torch.Tensor:
    """Broadcast an int or 0-dim tensor diffusion step to ``(batch_size,)``.

    The DDPM trainer passes ``(B,)`` steps, while the sampling loop iterates
    over 0-dim scheduler steps; both reach the backbones through the same
    contract. The UNet and Transformer networks expand internally; the MLP uses
    this helper.
    """
    if not torch.is_tensor(timestep):
        timestep = torch.tensor([timestep], dtype=torch.long, device=device)
    elif timestep.ndim == 0:
        timestep = timestep[None].to(device)
    return timestep.expand(batch_size)
