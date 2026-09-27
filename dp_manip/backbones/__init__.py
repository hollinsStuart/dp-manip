"""Noise-prediction backbones selected by ``policy.backbone``."""

from __future__ import annotations

from ..config import PolicyConfig
from .base import NoisePredictor
from .mlp import MLPBackbone
from .transformer import TransformerBackbone
from .unet import UNetBackbone

_BACKBONES: dict[str, type[NoisePredictor]] = {
    "unet": UNetBackbone,
    "transformer": TransformerBackbone,
    "mlp": MLPBackbone,
}


def build_noise_predictor(
    name: str,
    policy_cfg: PolicyConfig,
    *,
    obs_dim: int,
    action_dim: int,
) -> NoisePredictor:
    """Build the declared backbone; every experiment shares this boundary."""
    try:
        backbone = _BACKBONES[name]
    except KeyError:
        raise ValueError(
            f"unknown policy.backbone {name!r}; available: {sorted(_BACKBONES)}"
        ) from None
    return backbone(policy_cfg, obs_dim=obs_dim, action_dim=action_dim)


__all__ = [
    "MLPBackbone",
    "NoisePredictor",
    "TransformerBackbone",
    "UNetBackbone",
    "build_noise_predictor",
]
