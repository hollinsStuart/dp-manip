"""Observation encoder: an observation-history dict -> one flat conditioning vector.

The denoising network only sees the output of `ObsEncoder`, so it does not care
whether the policy is state- or image-based. Inputs are channel-last, exactly as
both the exported demos (`traj_i/obs_rgb/rgb`) and the evaluation env
(`FlattenRGBDObservationWrapper` + `FrameStack`) provide them:

    obs["state"]  (B, obs_horizon, P)           float
    obs["rgb"]    (B, obs_horizon, H, W, 3*C)   uint8, C cameras stacked on channels

`PlainConv` is copied from ManiSkill examples/baselines/diffusion_policy/
diffusion_policy/plain_conv.py (haosulab/ManiSkill@62ff3a5, Apache-2.0), and
`ObsEncoder` follows `Agent.encode_obs` of train_rgbd.py there: one CNN over all
cameras' channels, 256-d feature per frame, concatenated with the state.
This file only depends on torch, so it can be copied into other trainers as is.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def make_mlp(in_channels, mlp_channels, act_builder=nn.ReLU, last_act=True):
    c_in = in_channels
    module_list = []
    for idx, c_out in enumerate(mlp_channels):
        module_list.append(nn.Linear(c_in, c_out))
        if last_act or idx < len(mlp_channels) - 1:
            module_list.append(act_builder())
        c_in = c_out
    return nn.Sequential(*module_list)


class PlainConv(nn.Module):
    def __init__(self, in_channels=3, out_dim=256, pool_feature_map=False, last_act=True):
        super().__init__()
        # assume input image size is 128x128
        self.out_dim = out_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [64, 64]
            nn.Conv2d(16, 32, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [32, 32]
            nn.Conv2d(32, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [16, 16]
            nn.Conv2d(64, 128, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [8, 8]
            nn.Conv2d(128, 128, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
        )
        if pool_feature_map:
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
            self.fc = make_mlp(128, [out_dim], last_act=last_act)
        else:
            self.pool = None
            self.fc = make_mlp(128 * 4 * 4 * 4, [out_dim], last_act=last_act)
        self.reset_parameters()

    def reset_parameters(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, image):
        x = self.cnn(image)
        if self.pool is not None:
            x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class ObsEncoder(nn.Module):
    """{"state": (B, Th, P), ["rgb": (B, Th, H, W, C) uint8]} -> (B, out_dim).

    Without image_channels the encoder has no parameters and just flattens the
    state history (the state-only policy). State values are used as they are,
    as in the baseline.
    """

    def __init__(self, obs_horizon: int, state_dim: int, image_channels: int | None = None,
                 visual_feature_dim: int = 256):
        super().__init__()
        self.obs_horizon = obs_horizon
        self.visual_encoder = None
        per_frame = state_dim
        if image_channels is not None:
            self.visual_encoder = PlainConv(in_channels=image_channels, out_dim=visual_feature_dim,
                                            pool_feature_map=True)
            per_frame += visual_feature_dim
        self.out_dim = obs_horizon * per_frame

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        state = obs["state"].float()
        if self.visual_encoder is None:
            return state.flatten(start_dim=1)
        rgb = obs["rgb"]
        B, T = rgb.shape[:2]
        images = rgb.flatten(end_dim=1).permute(0, 3, 1, 2).float() / 255.0  # (B*Th, C, H, W)
        visual = self.visual_encoder(images).reshape(B, T, -1)
        return torch.cat((visual, state), dim=-1).flatten(start_dim=1)
