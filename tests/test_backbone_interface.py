"""Phase 7 regression tests: one NoisePredictor contract for every backbone.

The policy hands each backbone the shared ``(B, To, Dobs)`` observation
sequence and never flattens it itself. Backbone structure comes from the
resolved ``[policy]`` config, and checkpoints written before the interface
still load through the legacy key mapping.
"""

from __future__ import annotations

import dataclasses
import unittest
from pathlib import Path

from dp_manip.config import DiffusionConfig, PolicyConfig, VisionConfig, from_dict, from_recorded, load

try:
    import numpy as np
    import torch

    from dp_manip.backbones import (
        MLPBackbone,
        NoisePredictor,
        TransformerBackbone,
        UNetBackbone,
        build_noise_predictor,
    )
    from dp_manip.backbones.timestep import SinusoidalPosEmb, expand_timesteps
    from dp_manip.data import NormalizationStats
    from dp_manip.policy import DiffusionPolicy, load_policy_state_dict
except ModuleNotFoundError:  # torch is only installed in the cluster environment
    HAVE_TORCH = False
    # Keep the module importable without torch so unittest reports the tests
    # below as skipped instead of failing to collect them. The names are only
    # referenced by tests guarded by the HAVE_TORCH skip decorators.
    MLPBackbone = NoisePredictor = TransformerBackbone = UNetBackbone = None
    SinusoidalPosEmb = expand_timesteps = None
    NormalizationStats = DiffusionPolicy = load_policy_state_dict = None
else:
    HAVE_TORCH = True

ROOT = Path(__file__).resolve().parents[1]


def config_diff(left: dict, right: dict, prefix: str = "") -> dict:
    """Return dotted paths whose values differ between two resolved configs."""
    differences = {}
    for key in sorted(set(left) | set(right)):
        path = f"{prefix}.{key}" if prefix else key
        if key not in left or key not in right:
            differences[path] = (left.get(key), right.get(key))
        elif isinstance(left[key], dict) and isinstance(right[key], dict):
            differences.update(config_diff(left[key], right[key], path))
        elif left[key] != right[key]:
            differences[path] = (left[key], right[key])
    return differences


def make_policy_config(**overrides) -> "PolicyConfig":
    # Every field comes from baseline.toml; only sizes shrink for fast tests.
    values = dataclasses.asdict(load(ROOT / "configs" / "tasks" / "pickcube.toml").policy)
    values.update(
        obs_horizon=2,
        act_horizon=2,
        pred_horizon=4,
        diffusion_step_embed_dim=32,
        unet_dims=[16, 32],
        kernel_size=3,
        n_groups=4,
        transformer_layers=1,
        transformer_heads=2,
        transformer_embed_dim=8,
        transformer_dropout_emb=0.0,
        transformer_dropout_attn=0.0,
    )
    values.update(overrides)
    return PolicyConfig(**values)


def make_stats(proprio_dim: int, action_dim: int) -> "NormalizationStats":
    return NormalizationStats(
        proprio_mean=np.zeros(proprio_dim, dtype=np.float32),
        proprio_std=np.ones(proprio_dim, dtype=np.float32),
        action_low=-np.ones(action_dim, dtype=np.float32),
        action_high=np.ones(action_dim, dtype=np.float32),
    )


def make_policy(backbone: str = "unet") -> "DiffusionPolicy":
    return DiffusionPolicy(
        make_policy_config(backbone=backbone),
        VisionConfig(
            feature_dim=8, random_shift=0, share_camera_encoder=True, pool="avg", num_keypoints=32
        ),
        DiffusionConfig(num_diffusion_iters=8, num_inference_iters=2),
        image_shape=(32, 32, 6),
        proprio_dim=5,
        action_dim=4,
        stats=make_stats(5, 4),
    )


def make_observations(policy: "DiffusionPolicy", batch: int = 2):
    rgb = torch.randint(0, 256, (batch, policy.obs_horizon, 6, 32, 32), dtype=torch.uint8)
    proprio = torch.randn(batch, policy.obs_horizon, 5)
    return rgb, proprio


@unittest.skipUnless(HAVE_TORCH, "requires the cluster torch environment")
class NoisePredictorInterfaceTest(unittest.TestCase):
    BACKBONES = (
        ("unet", UNetBackbone),
        ("transformer", TransformerBackbone),
        ("mlp", MLPBackbone),
    )

    def test_factory_builds_the_declared_backbone(self) -> None:
        for name, expected in self.BACKBONES:
            with self.subTest(backbone=name):
                backbone = build_noise_predictor(
                    name, make_policy_config(), obs_dim=21, action_dim=4
                )
                self.assertIsInstance(backbone, NoisePredictor)
                self.assertIsInstance(backbone, expected)

    def test_unknown_backbone_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown policy.backbone"):
            build_noise_predictor("banana", make_policy_config(), obs_dim=21, action_dim=4)

    def test_each_backbone_implements_the_contract(self) -> None:
        for name, _ in self.BACKBONES:
            with self.subTest(backbone=name):
                backbone = build_noise_predictor(
                    name, make_policy_config(), obs_dim=21, action_dim=4
                )
                prediction = backbone(
                    torch.randn(3, 4, 4), torch.tensor([0, 1, 2]), torch.randn(3, 2, 21)
                )
                self.assertEqual(prediction.shape, (3, 4, 4))
                self.assertTrue(torch.isfinite(prediction).all())

    def test_mlp_arm_projects_observations_like_the_donor(self) -> None:
        # docs/final-plan.md §6 B2: flatten(obs) -> 256 -> 256, then concatenate
        # with the flattened noisy actions and the time embedding. With the
        # state-PickCube shapes (obs 42, To 2, action 4, Tp 16) VariDP's
        # MLP([84, 256, 256]) + MLPNoisePred has 88,064 + 264,512 parameters.
        policy_cfg = load(ROOT / "configs" / "tasks" / "pickcube.toml").policy
        backbone = MLPBackbone(policy_cfg, obs_dim=42, action_dim=4)
        first = backbone.obs_mlp[0]
        self.assertEqual((first.in_features, first.out_features), (84, 256))
        self.assertEqual(backbone.obs_mlp[-1].out_features, 256)
        self.assertEqual(backbone.net[0].in_features, 16 * 4 + 128 + 256)
        self.assertEqual(sum(p.numel() for p in backbone.parameters()), 352_576)

    def test_backbone_structure_comes_from_policy_config(self) -> None:
        cases = (
            (UNetBackbone, {"unet_dims": [16, 32]}, {"unet_dims": [32, 64]}),
            (TransformerBackbone, {"transformer_layers": 1}, {"transformer_layers": 2}),
            (MLPBackbone, {"mlp_hidden_dim": 16}, {"mlp_hidden_dim": 64}),
        )
        for backbone_cls, small_overrides, large_overrides in cases:
            with self.subTest(backbone=backbone_cls.__name__):
                small = backbone_cls(make_policy_config(**small_overrides), obs_dim=21, action_dim=4)
                large = backbone_cls(make_policy_config(**large_overrides), obs_dim=21, action_dim=4)
                self.assertLess(
                    sum(parameter.numel() for parameter in small.parameters()),
                    sum(parameter.numel() for parameter in large.parameters()),
                )


@unittest.skipUnless(HAVE_TORCH, "requires the cluster torch environment")
class TimestepEmbeddingTest(unittest.TestCase):
    def test_sinusoidal_embedding_shape(self) -> None:
        embedding = SinusoidalPosEmb(8)(torch.tensor([0, 1, 2]))
        self.assertEqual(embedding.shape, (3, 8))
        self.assertTrue(torch.isfinite(embedding).all())

    def test_expand_timesteps_accepts_int_scalar_and_vector(self) -> None:
        expected = torch.tensor([7, 7, 7], dtype=torch.long)
        for value in (7, torch.tensor(7), torch.tensor([7])):
            with self.subTest(value=value):
                actual = expand_timesteps(value, 3, torch.device("cpu"))
                self.assertTrue(torch.equal(actual, expected))


@unittest.skipUnless(HAVE_TORCH, "requires the cluster torch environment")
class PolicyBackboneBoundaryTest(unittest.TestCase):
    """The policy passes the unflattened observation sequence to the backbone."""

    def record_backbone_inputs(self, policy: "DiffusionPolicy") -> dict:
        seen: dict = {}
        original = policy.noise_predictor.forward

        def record(noisy_actions, timestep, obs_features):
            seen["obs_features"] = tuple(obs_features.shape)
            seen["noisy_actions"] = tuple(noisy_actions.shape)
            return original(noisy_actions, timestep, obs_features)

        policy.noise_predictor.forward = record
        return seen

    def test_training_loss_passes_the_sequence(self) -> None:
        policy = make_policy()
        seen = self.record_backbone_inputs(policy)
        rgb, proprio = make_observations(policy)
        loss = policy.compute_loss(rgb, proprio, torch.randn(2, 4, 4))
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(seen["obs_features"], (2, 2, 21))
        self.assertEqual(seen["noisy_actions"], (2, 4, 4))

    def test_sampling_passes_the_sequence(self) -> None:
        policy = make_policy().eval()
        seen = self.record_backbone_inputs(policy)
        rgb, proprio = make_observations(policy)
        with torch.no_grad():
            actions = policy.get_action(rgb, proprio)
        self.assertEqual(seen["obs_features"], (2, 2, 21))
        self.assertEqual(actions.shape, (2, 2, 4))
        self.assertTrue(torch.isfinite(actions).all())

    def test_pre_interface_checkpoint_keys_still_load(self) -> None:
        policy = make_policy()
        current = policy.state_dict()
        # Reproduce the pre-Phase-6/7 layout: top-level camera weights,
        # version-1 ``state_*`` proprio buffers, and the UNet named directly.
        legacy = {}
        for key, value in current.items():
            if key.startswith("observation_encoder."):
                key = key[len("observation_encoder.") :]
            elif key.startswith("noise_predictor.unet."):
                key = "noise_pred_net." + key[len("noise_predictor.unet.") :]
            legacy[key] = value
        legacy["state_mean"] = legacy.pop("proprio_mean")
        legacy["state_std"] = legacy.pop("proprio_std")
        restored = make_policy()
        load_policy_state_dict(restored, legacy)
        for name, value in current.items():
            self.assertTrue(torch.equal(value, restored.state_dict()[name]), name)


class BackboneConfigTest(unittest.TestCase):
    def test_resolved_config_selects_the_canonical_unet(self) -> None:
        resolved = load(ROOT / "configs" / "tasks" / "pickcube.toml")
        self.assertEqual(resolved.policy.backbone, "unet")

    def test_backbone_arms_differ_only_in_the_selector(self) -> None:
        # Gate B: switching arms must not move a single scientific or structural
        # value; only ``policy.backbone`` may change.
        base = load(ROOT / "configs" / "tasks" / "pickcube.toml").to_dict()
        for name in ("transformer", "mlp"):
            with self.subTest(backbone=name):
                arm = load(
                    ROOT / "configs" / "tasks" / "pickcube.toml",
                    [f'policy.backbone="{name}"'],
                ).to_dict()
                self.assertEqual(config_diff(base, arm), {"policy.backbone": ("unet", name)})

    def test_pre_phase_nine_checkpoint_configs_keep_donor_defaults(self) -> None:
        raw = load(ROOT / "configs" / "tasks" / "pickcube.toml").to_dict()
        for name in (
            "transformer_layers",
            "transformer_heads",
            "transformer_embed_dim",
            "transformer_dropout_emb",
            "transformer_dropout_attn",
            "transformer_causal_attn",
            "transformer_cond_layers",
            "mlp_hidden_dim",
            "mlp_layers",
            "mlp_time_embed_dim",
            "mlp_obs_feat_dim",
        ):
            del raw["policy"][name]
        policy = from_recorded(raw).policy
        self.assertEqual(policy.transformer_layers, 8)
        self.assertEqual(policy.transformer_heads, 4)
        self.assertEqual(policy.transformer_embed_dim, 256)
        self.assertEqual(policy.transformer_dropout_emb, 0.0)
        self.assertEqual(policy.transformer_dropout_attn, 0.3)
        self.assertTrue(policy.transformer_causal_attn)
        self.assertEqual(policy.transformer_cond_layers, 0)
        self.assertEqual(policy.mlp_hidden_dim, 256)
        self.assertEqual(policy.mlp_layers, 3)
        self.assertEqual(policy.mlp_time_embed_dim, 128)
        self.assertEqual(policy.mlp_obs_feat_dim, 256)

    def test_invalid_transformer_structure_is_rejected(self) -> None:
        for override in (
            "policy.transformer_heads=3",
            "policy.transformer_embed_dim=7",
            "policy.transformer_layers=0",
            "policy.transformer_dropout_attn=1.0",
            "policy.transformer_cond_layers=-1",
        ):
            with self.subTest(override=override), self.assertRaisesRegex(ValueError, "policy.transformer"):
                load(ROOT / "configs" / "tasks" / "pickcube.toml", [override])

    def test_invalid_mlp_structure_is_rejected(self) -> None:
        for override in (
            "policy.mlp_layers=0",
            "policy.mlp_hidden_dim=0",
            "policy.mlp_time_embed_dim=7",
            "policy.mlp_time_embed_dim=1",
            "policy.mlp_obs_feat_dim=0",
        ):
            with self.subTest(override=override), self.assertRaisesRegex(ValueError, "policy.mlp"):
                load(ROOT / "configs" / "tasks" / "pickcube.toml", [override])

    def test_pre_interface_checkpoint_config_defaults_to_unet(self) -> None:
        raw = load(ROOT / "configs" / "tasks" / "pickcube.toml").to_dict()
        del raw["policy"]["backbone"]
        self.assertEqual(from_recorded(raw).policy.backbone, "unet")


if __name__ == "__main__":
    unittest.main()
