"""Phase 3 of the failure-aware plan: ``FailureGuidedPolicy``.

Covers the plan's required unit tests (§8 Phase 3): alpha = 0 reproduces the
baseline, identical predictors give adaptive lambda = 0, cosine = -1 gives
lambda = alpha, both predictors receive identical diffusion inputs, and the
output has the ``DiffusionPolicy`` interface. Plus the load-time checks that
keep the two predictors in one coordinate system.
"""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import torch

    from dp_manip import config as config_lib
    from dp_manip.data import NormalizationStats
    from dp_manip.failure_guidance import (
        FailureGuidedPolicy,
        GuidanceDiagnostics,
        guidance_weight,
        guided_noise,
    )
    from dp_manip.metadata import file_sha256
    from dp_manip.policy import DiffusionPolicy
except ModuleNotFoundError:  # torch is only installed in the cluster environment
    HAVE_TORCH = False
else:
    HAVE_TORCH = True

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "configs" / "tasks" / "peginsertionside.toml"
PROPRIO_DIM = 5
ACTION_DIM = 8


def tiny_config():
    return config_lib.load(
        str(TASK),
        [
            "vision.feature_dim=8",
            "policy.unet_dims=[16, 32]",
            "policy.kernel_size=3",
            "policy.n_groups=4",
            "diffusion.num_diffusion_iters=6",
            "diffusion.num_inference_iters=6",
        ],
    )


def tiny_policy(cfg, seed: int = 0) -> "DiffusionPolicy":
    torch.manual_seed(seed)
    stats = NormalizationStats(
        proprio_mean=np.zeros(PROPRIO_DIM, np.float32),
        proprio_std=np.ones(PROPRIO_DIM, np.float32),
        action_low=-np.ones(ACTION_DIM, np.float32),
        action_high=np.ones(ACTION_DIM, np.float32),
    )
    return DiffusionPolicy(
        cfg.policy,
        cfg.vision,
        cfg.diffusion,
        image_shape=(64, 64, 3),
        proprio_dim=PROPRIO_DIM,
        action_dim=ACTION_DIM,
        stats=stats,
    ).eval()


def perturbed_copy(policy: "DiffusionPolicy") -> "DiffusionPolicy":
    """A 'fine-tuned' copy: same encoder and normalization, different noise predictor."""
    negative = copy.deepcopy(policy)
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for parameter in negative.noise_predictor.parameters():
            parameter.add_(0.1 * torch.randn(parameter.shape, generator=generator))
    return negative


def observations(policy, batch: int = 3):
    generator = torch.Generator().manual_seed(2)
    rgb = torch.randint(0, 256, (batch, policy.obs_horizon, 3, 64, 64), generator=generator, dtype=torch.uint8)
    proprio = torch.randn(batch, policy.obs_horizon, PROPRIO_DIM, generator=generator)
    return rgb, proprio


def sample(policy, rgb, proprio):
    return policy.get_action(rgb, proprio, generator=torch.Generator().manual_seed(0))


@unittest.skipUnless(HAVE_TORCH, "requires the cluster torch environment")
class GuidanceMathTest(unittest.TestCase):
    def test_cosine_minus_one_gives_lambda_alpha(self) -> None:
        eps = torch.randn(4, 16, ACTION_DIM)
        weight, cosine = guidance_weight(eps, -eps, mode="adaptive", alpha=2.0)
        torch.testing.assert_close(cosine, -torch.ones(4))
        torch.testing.assert_close(weight, torch.full((4,), 2.0))
        torch.testing.assert_close(guided_noise(eps, -eps, weight), 5.0 * eps)

    def test_identical_predictions_give_zero_adaptive_lambda(self) -> None:
        eps = torch.randn(4, 16, ACTION_DIM)
        weight, cosine = guidance_weight(eps, eps.clone(), mode="adaptive", alpha=2.0)
        torch.testing.assert_close(cosine, torch.ones(4))
        torch.testing.assert_close(weight, torch.zeros(4), atol=1e-6, rtol=0)

    def test_fixed_lambda_is_alpha_for_every_sample(self) -> None:
        eps, other = torch.randn(3, 16, ACTION_DIM), torch.randn(3, 16, ACTION_DIM)
        weight, _ = guidance_weight(eps, other, mode="fixed", alpha=0.5)
        torch.testing.assert_close(weight, torch.full((3,), 0.5))
        torch.testing.assert_close(guided_noise(eps, other, weight), 1.5 * eps - 0.5 * other)

    def test_adaptive_lambda_is_per_sample(self) -> None:
        eps = torch.randn(2, 16, ACTION_DIM)
        other = torch.stack((eps[0], -eps[1]))
        weight, _ = guidance_weight(eps, other, mode="adaptive", alpha=1.0)
        torch.testing.assert_close(weight, torch.tensor([0.0, 1.0]), atol=1e-6, rtol=0)


@unittest.skipUnless(HAVE_TORCH, "requires the cluster torch environment")
class FailureGuidedPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = tiny_config()
        self.base = tiny_policy(self.cfg)
        self.negative = perturbed_copy(self.base)
        self.rgb, self.proprio = observations(self.base)

    def test_alpha_zero_reproduces_the_baseline_exactly(self) -> None:
        expected = sample(self.base, self.rgb, self.proprio)
        for mode in ("fixed", "adaptive"):
            for diagnostics in (None, GuidanceDiagnostics()):
                with self.subTest(mode=mode, diagnostics=diagnostics is not None):
                    guided = FailureGuidedPolicy(
                        self.base, self.negative, mode=mode, alpha=0.0, diagnostics=diagnostics
                    )
                    self.assertTrue(torch.equal(sample(guided, self.rgb, self.proprio), expected))
        # The dry-run diagnostics measured cosines along that baseline trajectory.
        summary = diagnostics.summary()
        self.assertEqual(summary["samples"], 3 * self.cfg.diffusion.num_inference_iters)
        self.assertGreater(summary["mean_half_one_minus_cos"], 0.0)

    def test_identical_predictors_leave_adaptive_sampling_unchanged(self) -> None:
        diagnostics = GuidanceDiagnostics()
        guided = FailureGuidedPolicy(
            self.base, copy.deepcopy(self.base), mode="adaptive", alpha=2.0, diagnostics=diagnostics
        )
        torch.testing.assert_close(sample(guided, self.rgb, self.proprio), sample(self.base, self.rgb, self.proprio))
        self.assertAlmostEqual(diagnostics.summary()["mean"]["weight"], 0.0, places=6)

    def test_guidance_changes_actions_and_keeps_the_interface(self) -> None:
        guided = FailureGuidedPolicy(self.base, self.negative, mode="fixed", alpha=1.0)
        actions = sample(guided, self.rgb, self.proprio)
        self.assertEqual(actions.shape, (3, self.base.act_horizon, ACTION_DIM))
        self.assertTrue(torch.isfinite(actions).all())
        self.assertFalse(torch.equal(actions, sample(self.base, self.rgb, self.proprio)))
        for name in ("obs_horizon", "act_horizon", "pred_horizon", "action_dim"):
            self.assertEqual(getattr(guided, name), getattr(self.base, name))
        guided.train()
        self.assertTrue(self.base.training and self.negative.training)
        guided.eval()
        self.assertFalse(guided.training or self.base.training or self.negative.training)

    def test_both_predictors_receive_identical_inputs(self) -> None:
        calls: dict[str, list] = {"base": [], "negative": []}

        def spy(name, module):
            def hook(_module, args, _output):
                calls[name].append(tuple(arg.detach().clone() for arg in args))

            return module.register_forward_hook(hook)

        handles = [spy("base", self.base.noise_predictor), spy("negative", self.negative.noise_predictor)]
        try:
            sample(FailureGuidedPolicy(self.base, self.negative, mode="adaptive", alpha=1.0), self.rgb, self.proprio)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(len(calls["base"]), self.cfg.diffusion.num_inference_iters)
        self.assertEqual(len(calls["base"]), len(calls["negative"]))
        for base_args, negative_args in zip(calls["base"], calls["negative"]):
            for base_arg, negative_arg in zip(base_args, negative_args):
                self.assertTrue(torch.equal(base_arg, negative_arg))

    def test_diagnostics_cover_every_denoising_step(self) -> None:
        diagnostics = GuidanceDiagnostics()
        guided = FailureGuidedPolicy(self.base, self.negative, mode="adaptive", alpha=1.0, diagnostics=diagnostics)
        sample(guided, self.rgb, self.proprio)
        summary = diagnostics.summary()
        steps = self.cfg.diffusion.num_inference_iters
        self.assertEqual(len(summary["timesteps"]), steps)
        self.assertEqual(summary["timesteps"], sorted(summary["timesteps"], reverse=True))
        for name, values in summary["per_timestep"].items():
            self.assertEqual(len(values), steps, name)
        self.assertTrue(0.0 <= summary["mean"]["x0_clip"] <= 1.0)
        self.assertTrue(0.0 <= summary["mean"]["weight"] <= 1.0)
        self.assertGreater(summary["mean"]["relative_change"], 0.0)

    def test_a_different_encoder_is_rejected(self) -> None:
        negative = copy.deepcopy(self.negative)
        with torch.no_grad():
            next(negative.observation_encoder.parameters()).add_(1e-3)
        with self.assertRaisesRegex(ValueError, "observation encoders differ"):
            FailureGuidedPolicy(self.base, negative, mode="fixed", alpha=1.0)

    def test_different_normalization_is_rejected(self) -> None:
        negative = copy.deepcopy(self.negative)
        negative.action_high.mul_(2.0)
        with self.assertRaisesRegex(ValueError, "action_high"):
            FailureGuidedPolicy(self.base, negative, mode="fixed", alpha=1.0)

    def test_invalid_mode_and_alpha_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            FailureGuidedPolicy(self.base, self.negative, mode="both", alpha=1.0)
        with self.assertRaises(ValueError):
            FailureGuidedPolicy(self.base, self.negative, mode="fixed", alpha=-1.0)


class FakeRGBVectorEnv:
    """Minimal stand-in for the ManiSkill vector env ``evaluate`` drives."""

    def __init__(self, num_envs: int = 2, max_steps: int = 20):
        self.num_envs = num_envs
        self.max_steps = max_steps

    def _observation(self):
        rng = np.random.default_rng([*self.seeds, self.t])
        return {
            "rgb": rng.integers(0, 256, (self.num_envs, 64, 64, 3), dtype=np.uint8),
            "state": rng.normal(size=(self.num_envs, PROPRIO_DIM)).astype(np.float32),
        }

    def reset(self, seed):
        self.seeds, self.t = list(seed), 0
        return self._observation(), {}

    def step(self, action):
        action = np.asarray(action)
        assert action.shape == (self.num_envs, ACTION_DIM)
        self.t += 1
        # Success depends on the executed actions, so any change in sampling shows up.
        success = action[:, 0] > 0.9
        truncated = np.full(self.num_envs, self.t >= self.max_steps)
        return self._observation(), action.sum(axis=1), np.zeros(self.num_envs, bool), truncated, {"success": success}


@unittest.skipUnless(HAVE_TORCH, "requires the cluster torch environment")
class ClosedLoopTest(unittest.TestCase):
    def test_evaluate_runs_the_guided_policy_and_alpha_zero_matches_the_baseline(self) -> None:
        from dp_manip.evaluate import evaluate

        cfg = tiny_config()
        base = tiny_policy(cfg)
        negative = perturbed_copy(base)
        seeds = [10_000, 10_001, 10_002, 10_003]

        def run(policy):
            return evaluate(policy, FakeRGBVectorEnv(), seeds, torch.device("cpu"), inference_seed=0)["episodes"]

        baseline = run(base)
        self.assertEqual(run(FailureGuidedPolicy(base, negative, mode="adaptive", alpha=0.0)), baseline)
        guided = run(FailureGuidedPolicy(base, negative, mode="fixed", alpha=2.0))
        self.assertEqual([episode["seed"] for episode in guided], seeds)
        self.assertNotEqual([episode["return"] for episode in guided], [episode["return"] for episode in baseline])


@unittest.skipUnless(HAVE_TORCH, "requires the cluster torch environment")
class FromCheckpointsTest(unittest.TestCase):
    def payload(self, policy, cfg, finetune=None) -> dict:
        payload = {
            "config": cfg.to_dict(),
            "train_data": {"image_shape": [64, 64, 3], "proprio_dim": PROPRIO_DIM, "action_dim": ACTION_DIM},
            "normalization": {
                "proprio_mean": [0.0] * PROPRIO_DIM,
                "proprio_std": [1.0] * PROPRIO_DIM,
                "action_low": [-1.0] * ACTION_DIM,
                "action_high": [1.0] * ACTION_DIM,
            },
            "model": policy.state_dict(),
            "step": 7,
        }
        if finetune is not None:
            payload["finetune"] = finetune
        return payload

    def test_negative_must_be_fine_tuned_from_that_baseline(self) -> None:
        cfg = tiny_config()
        base = tiny_policy(cfg)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_path, other_path = root / "base.pt", root / "other.pt"
            torch.save(self.payload(base, cfg), base_path)
            torch.save(self.payload(tiny_policy(cfg, seed=5), cfg), other_path)
            record = {"init_checkpoint_sha256": file_sha256(base_path)}
            negative_path = root / "negative.pt"
            torch.save(self.payload(perturbed_copy(base), cfg, record), negative_path)

            guided = FailureGuidedPolicy.from_checkpoints(base_path, negative_path, mode="adaptive", alpha=1.0)
            self.assertEqual(guided.provenance["base_checkpoint"]["sha256"], record["init_checkpoint_sha256"])
            self.assertEqual(guided.provenance["negative_checkpoint"]["finetune"], record)
            self.assertEqual(guided.provenance["mode"], "adaptive")

            with self.assertRaisesRegex(ValueError, "was fine-tuned from"):
                FailureGuidedPolicy.from_checkpoints(other_path, negative_path, mode="fixed", alpha=1.0)
            with self.assertRaisesRegex(ValueError, "not a fine-tuned checkpoint"):
                FailureGuidedPolicy.from_checkpoints(base_path, other_path, mode="fixed", alpha=1.0)


if __name__ == "__main__":
    unittest.main()
