"""Phase 6 regression tests: the shared observation boundary is ``(B, To, Dobs)``.

The encoder owns RGB preprocessing, camera encoding, augmentation, and proprio
normalization, and must stay independent of the noise-prediction backbone. Only
the backbone adapter may flatten the sequence; the encoder API never returns the
flattened ``(B, To x Dobs)`` form.
"""

from __future__ import annotations

import dataclasses
import unittest
from pathlib import Path

try:
    import numpy as np
    import torch

    from dp_manip.config import DiffusionConfig, PolicyConfig, VisionConfig, load
    from dp_manip.data import NormalizationStats
    from dp_manip.observation_encoder import ObservationEncoder
    from dp_manip.policy import DiffusionPolicy, load_policy_state_dict
except ModuleNotFoundError:  # torch is only installed in the cluster environment
    HAVE_TORCH = False
else:
    HAVE_TORCH = True


ROOT = Path(__file__).resolve().parents[1]


def make_stats(proprio_dim: int, action_dim: int) -> NormalizationStats:
    return NormalizationStats(
        proprio_mean=np.zeros(proprio_dim, dtype=np.float32),
        proprio_std=np.ones(proprio_dim, dtype=np.float32),
        action_low=-np.ones(action_dim, dtype=np.float32),
        action_high=np.ones(action_dim, dtype=np.float32),
    )


def make_vision(feature_dim: int = 8, random_shift: int = 2, share: bool = True) -> VisionConfig:
    return VisionConfig(feature_dim=feature_dim, random_shift=random_shift, share_camera_encoder=share)


def make_encoder(
    *, num_cameras: int = 2, proprio_dim: int = 5, feature_dim: int = 8, share: bool = True
) -> ObservationEncoder:
    return ObservationEncoder(
        make_vision(feature_dim, share=share),
        obs_horizon=2,
        image_shape=(32, 32, 3 * num_cameras),
        proprio_dim=proprio_dim,
        stats=make_stats(proprio_dim, 4),
    )


@unittest.skipUnless(HAVE_TORCH, "requires the cluster torch environment")
class ObservationEncoderShapeTest(unittest.TestCase):
    def test_output_is_batch_horizon_features(self) -> None:
        encoder = make_encoder().eval()
        rgb = torch.randint(0, 256, (3, 2, 6, 32, 32), dtype=torch.uint8)
        proprio = torch.randn(3, 2, 5)
        features = encoder(rgb, proprio)
        self.assertEqual(features.shape, (3, 2, 2 * 8 + 5))
        self.assertEqual(encoder.output_dim, 2 * 8 + 5)
        self.assertTrue(torch.isfinite(features).all())

    def test_fuses_rgb_and_proprio(self) -> None:
        encoder = make_encoder().eval()
        rgb = torch.randint(0, 256, (2, 2, 6, 32, 32), dtype=torch.uint8)
        proprio = torch.randn(2, 2, 5)
        base = encoder(rgb, proprio)
        self.assertFalse(torch.equal(base, encoder(rgb, proprio + 1.0)))
        self.assertFalse(torch.equal(base, encoder(255 - rgb, proprio)))

    def test_per_camera_encoders_keep_the_feature_width(self) -> None:
        encoder = make_encoder(share=False).eval()
        rgb = torch.randint(0, 256, (2, 2, 6, 32, 32), dtype=torch.uint8)
        features = encoder(rgb, torch.randn(2, 2, 5))
        self.assertEqual(features.shape, (2, 2, 2 * 8 + 5))
        self.assertEqual(len(encoder.image_encoders), 2)

    def test_mismatched_shapes_are_rejected(self) -> None:
        encoder = make_encoder().eval()
        rgb = torch.zeros(2, 2, 6, 32, 32)
        with self.assertRaises(ValueError):
            encoder(rgb, torch.zeros(2, 2, 4))
        with self.assertRaises(ValueError):
            encoder(rgb, torch.zeros(2, 3, 5))
        with self.assertRaises(ValueError):
            encoder(torch.zeros(2, 2, 3, 32, 32), torch.zeros(2, 2, 5))


@unittest.skipUnless(HAVE_TORCH, "requires the cluster torch environment")
class PolicyEncoderBoundaryTest(unittest.TestCase):
    """The UNet adapter flattens; the shared encoder never does."""

    def make_policy(self) -> DiffusionPolicy:
        # Every field comes from baseline.toml; only sizes shrink for fast tests.
        policy_cfg = dataclasses.replace(
            load(ROOT / "configs" / "tasks" / "pickcube.toml").policy,
            obs_horizon=2,
            act_horizon=2,
            pred_horizon=4,
            diffusion_step_embed_dim=32,
            unet_dims=[16, 32],
            kernel_size=3,
            n_groups=4,
        )
        return DiffusionPolicy(
            policy_cfg,
            make_vision(feature_dim=8, random_shift=0),
            DiffusionConfig(num_diffusion_iters=8, num_inference_iters=2),
            image_shape=(32, 32, 6),
            proprio_dim=5,
            action_dim=4,
            stats=make_stats(5, 4),
        )

    def test_policy_exposes_the_shared_sequence(self) -> None:
        policy = self.make_policy().eval()
        rgb = torch.randint(0, 256, (3, 2, 6, 32, 32), dtype=torch.uint8)
        proprio = torch.randn(3, 2, 5)
        features = policy.observation_features(rgb, proprio)
        self.assertEqual(features.shape, (3, 2, 2 * 8 + 5))
        self.assertTrue(torch.equal(features, policy.observation_encoder(rgb, proprio)))

    def test_loss_and_sampling_use_the_encoder(self) -> None:
        policy = self.make_policy()
        rgb = torch.randint(0, 256, (2, 2, 6, 32, 32), dtype=torch.uint8)
        proprio = torch.randn(2, 2, 5)
        loss = policy.compute_loss(rgb, proprio, torch.randn(2, 4, 4))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        policy.eval()
        with torch.no_grad():
            actions = policy.get_action(rgb, proprio)
        self.assertEqual(actions.shape, (2, 2, 4))
        self.assertTrue(torch.isfinite(actions).all())

    def test_pre_encoder_checkpoint_keys_still_load(self) -> None:
        policy = self.make_policy()
        current = policy.state_dict()
        # Reproduce the pre-extraction observation layout: camera weights at the
        # top level and version-1 ``state_*`` proprio buffers. (The backbone
        # rename is covered in tests/test_backbone_interface.py.)
        legacy = {
            key[len("observation_encoder.") :] if key.startswith("observation_encoder.") else key: value
            for key, value in current.items()
        }
        legacy["state_mean"] = legacy.pop("proprio_mean")
        legacy["state_std"] = legacy.pop("proprio_std")
        restored = self.make_policy()
        load_policy_state_dict(restored, legacy)
        for name, value in current.items():
            self.assertTrue(torch.equal(value, restored.state_dict()[name]), name)


if __name__ == "__main__":
    unittest.main()
