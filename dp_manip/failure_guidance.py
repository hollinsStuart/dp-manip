"""Negative guidance from a second noise predictor (failure-aware plan §5).

:class:`FailureGuidedPolicy` samples with the baseline policy's scheduler and
randomness, but at every denoising step replaces the baseline's noise
prediction ``eps_b`` with

    eps = eps_b + lambda * (eps_b - eps_n)

where ``eps_n`` comes from the negative model: the failure model for arms F/A,
the success-rollout model for control C1. ``lambda = alpha`` (fixed) or
``alpha * (1 - cos(eps_b, eps_n)) / 2`` per sample (adaptive). Both predictors
see the same noisy sample, timestep and observation features, which are
computed once by the baseline encoder; the negative model's encoder must be
bit-identical (it was frozen during fine-tuning), which the constructor checks.
With ``alpha = 0`` the wrapper reproduces ``DiffusionPolicy.get_action``
exactly, because both go through ``DiffusionPolicy.sample_actions``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .metadata import file_sha256
from .policy import DiffusionPolicy

MODES = ("fixed", "adaptive")


def guidance_weight(
    eps_base: torch.Tensor, eps_negative: torch.Tensor, *, mode: str, alpha: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-sample ``(lambda, cosine)``, each shaped ``(B,)``."""
    cosine = F.cosine_similarity(eps_base.flatten(1), eps_negative.flatten(1), dim=1, eps=1e-8)
    if mode == "fixed":
        weight = torch.full_like(cosine, float(alpha))
    elif mode == "adaptive":
        weight = alpha * (1.0 - cosine) / 2.0
    else:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    return weight, cosine


def guided_noise(eps_base: torch.Tensor, eps_negative: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``eps_b + lambda * (eps_b - eps_n)`` with ``lambda`` broadcast per sample."""
    return eps_base + weight.view(-1, *([1] * (eps_base.ndim - 1))) * (eps_base - eps_negative)


class GuidanceDiagnostics:
    """Per-denoising-timestep statistics of the guidance (plan §6.6).

    Accumulates over every sample of every ``get_action`` call. ``summary``
    reports overall means and per-timestep means ordered from the first
    (noisiest) denoising step to the last. ``mean_half_one_minus_cos`` is the
    ``m`` of the success-blind dry run (§5.5).
    """

    FIELDS = ("cosine", "weight", "small_weight", "relative_change", "x0_clip")

    def __init__(self, small_weight_fraction: float = 0.01):
        self.small_weight_fraction = small_weight_fraction
        self.sums: dict[int, dict[str, float]] = {}
        self.counts: dict[int, int] = {}

    def record(
        self,
        timestep: int,
        *,
        alpha: float,
        cosine: torch.Tensor,
        weight: torch.Tensor,
        eps_base: torch.Tensor,
        eps_guided: torch.Tensor,
        sample: torch.Tensor,
        alpha_cumprod: torch.Tensor,
    ) -> None:
        # The scheduler clips this predicted x0 to [-1, 1] (clip_sample=True).
        x0 = (sample - (1.0 - alpha_cumprod).sqrt() * eps_guided) / alpha_cumprod.sqrt()
        base_norm = eps_base.flatten(1).norm(dim=1).clamp_min(1e-12)
        values = {
            "cosine": cosine,
            "weight": weight,
            "small_weight": (weight < self.small_weight_fraction * alpha).float(),
            "relative_change": (eps_guided - eps_base).flatten(1).norm(dim=1) / base_norm,
            "x0_clip": (x0.abs() > 1.0).flatten(1).float().mean(dim=1),
        }
        sums = self.sums.setdefault(timestep, dict.fromkeys(self.FIELDS, 0.0))
        for name, value in values.items():
            sums[name] += float(value.sum())
        self.counts[timestep] = self.counts.get(timestep, 0) + int(cosine.numel())

    def summary(self) -> dict[str, Any]:
        timesteps = sorted(self.counts, reverse=True)
        total = sum(self.counts.values())
        if not total:
            return {"samples": 0}
        overall = {
            name: sum(self.sums[t][name] for t in timesteps) / total for name in self.FIELDS
        }
        per_timestep = {
            name: [self.sums[t][name] / self.counts[t] for t in timesteps] for name in self.FIELDS
        }
        return {
            "samples": total,
            "mean": overall,
            "mean_half_one_minus_cos": (1.0 - overall["cosine"]) / 2.0,
            "timesteps": timesteps,
            "per_timestep": per_timestep,
        }


def _check_compatible(base: DiffusionPolicy, negative: DiffusionPolicy) -> None:
    """Both predictors must share one coordinate system and conditioning (plan §4.2, §5.4)."""
    for name in ("obs_horizon", "act_horizon", "pred_horizon", "action_dim", "num_inference_iters"):
        if getattr(base, name) != getattr(negative, name):
            raise ValueError(f"{name} differs: {getattr(base, name)} vs {getattr(negative, name)}")
    if dict(base.noise_scheduler.config) != dict(negative.noise_scheduler.config):
        raise ValueError("diffusion scheduler configs differ")
    for name in ("action_low", "action_high"):
        if not torch.equal(getattr(base, name), getattr(negative, name)):
            raise ValueError(f"{name} differs: the negative model must reuse the baseline normalization")
    base_encoder = base.observation_encoder.state_dict()
    negative_encoder = negative.observation_encoder.state_dict()
    if base_encoder.keys() != negative_encoder.keys() or any(
        not torch.equal(value, negative_encoder[key]) for key, value in base_encoder.items()
    ):
        raise ValueError(
            "observation encoders differ: shared features require the negative model's "
            "encoder (and proprio normalization) to be bit-identical to the baseline's"
        )
    base_predictor = {key: value.shape for key, value in base.noise_predictor.state_dict().items()}
    negative_predictor = {key: value.shape for key, value in negative.noise_predictor.state_dict().items()}
    if base_predictor != negative_predictor:
        raise ValueError("noise predictor architectures differ")


class FailureGuidedPolicy(nn.Module):
    """``DiffusionPolicy``-compatible sampler with negative guidance."""

    def __init__(
        self,
        base: DiffusionPolicy,
        negative: DiffusionPolicy,
        *,
        mode: str,
        alpha: float,
        diagnostics: GuidanceDiagnostics | None = None,
    ):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if alpha < 0.0:
            raise ValueError("alpha must be non-negative")
        _check_compatible(base, negative)
        self.base = base
        self.negative = negative
        self.mode = mode
        self.alpha = float(alpha)
        self.diagnostics = diagnostics
        self.provenance: dict[str, Any] = {}

    @property
    def obs_horizon(self) -> int:
        return self.base.obs_horizon

    @property
    def act_horizon(self) -> int:
        return self.base.act_horizon

    @property
    def pred_horizon(self) -> int:
        return self.base.pred_horizon

    @property
    def action_dim(self) -> int:
        return self.base.action_dim

    def guided_prediction(
        self, sample: torch.Tensor, timestep: torch.Tensor, obs_features: torch.Tensor
    ) -> torch.Tensor:
        """Guided noise for one denoising step; both predictors get identical inputs."""
        eps_base = self.base.noise_predictor(sample, timestep, obs_features)
        if self.alpha == 0.0 and self.diagnostics is None:
            return eps_base
        eps_negative = self.negative.noise_predictor(sample, timestep, obs_features)
        weight, cosine = guidance_weight(eps_base, eps_negative, mode=self.mode, alpha=self.alpha)
        # alpha = 0 still runs the negative model when diagnostics are on (the
        # success-blind dry run measures cosines along the baseline trajectory)
        # but returns the baseline prediction itself, so sampling is unchanged.
        eps = eps_base if self.alpha == 0.0 else guided_noise(eps_base, eps_negative, weight)
        if self.diagnostics is not None:
            self.diagnostics.record(
                int(timestep),
                alpha=self.alpha,
                cosine=cosine,
                weight=weight,
                eps_base=eps_base,
                eps_guided=eps,
                sample=sample,
                alpha_cumprod=self.base.noise_scheduler.alphas_cumprod[timestep],
            )
        return eps

    @torch.no_grad()
    def get_action(
        self,
        rgb: torch.Tensor,
        proprio: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Return ``(B, act_horizon, action_dim)``, exactly like ``DiffusionPolicy``."""
        obs_features = self.base.observation_features(rgb, proprio)
        return self.base.sample_actions(
            obs_features,
            generator=generator,
            noise_fn=lambda sample, timestep: self.guided_prediction(sample, timestep, obs_features),
        )

    @classmethod
    def from_checkpoints(
        cls,
        base_checkpoint: str | Path,
        negative_checkpoint: str | Path,
        *,
        mode: str,
        alpha: float,
        device: str | torch.device = "cpu",
        diagnostics: GuidanceDiagnostics | None = None,
    ) -> "FailureGuidedPolicy":
        """Load a baseline and a negative model fine-tuned from exactly that baseline.

        The negative checkpoint's fine-tuning record must name the baseline
        file's sha256, so an evaluation cannot pair a failure model with a
        different baseline than the one it was trained from.
        """
        base_path = Path(base_checkpoint).expanduser().resolve()
        negative_path = Path(negative_checkpoint).expanduser().resolve()
        base_sha = file_sha256(base_path)
        negative_payload = torch.load(negative_path, map_location="cpu", weights_only=False)
        record = negative_payload.get("finetune")
        if not record:
            raise ValueError(f"{negative_path} is not a fine-tuned checkpoint (no fine-tuning record)")
        if record["init_checkpoint_sha256"] != base_sha:
            raise ValueError(
                f"{negative_path} was fine-tuned from {record['init_checkpoint_sha256']}, "
                f"not from the baseline {base_path} ({base_sha})"
            )
        base_payload = torch.load(base_path, map_location="cpu", weights_only=False)
        policy = cls(
            DiffusionPolicy.from_checkpoint(base_payload, device),
            DiffusionPolicy.from_checkpoint(negative_payload, device),
            mode=mode,
            alpha=alpha,
            diagnostics=diagnostics,
        )
        policy.provenance = {
            "base_checkpoint": {"path": str(base_path), "sha256": base_sha, "step": int(base_payload["step"])},
            "negative_checkpoint": {
                "path": str(negative_path),
                "sha256": file_sha256(negative_path),
                "step": int(negative_payload["step"]),
                "finetune": record,
            },
            "mode": mode,
            "alpha": float(alpha),
        }
        return policy.to(device)
