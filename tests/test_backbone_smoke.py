"""Phase 9-10 smoke tests: every backbone completes a train step and a sampling pass.

The refactor plan requires each UNet / Transformer / MLP arm to execute one
forward, one backward, one optimizer step and one sampling pass through the
shared policy. The full Gate A lifecycle (train -> save checkpoint -> load
checkpoint -> closed-loop evaluate) runs on the cluster with ManiSkill; this
test pins the backbone math and the shared policy boundary on CPU.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Import the helper module by its file location: ``tests`` is not a package,
# so ``test_backbone_interface`` is not importable by default from the repo
# root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_backbone_interface import make_observations, make_policy

try:
    import torch
except ModuleNotFoundError:  # torch is only installed in the cluster environment
    HAVE_TORCH = False
else:
    HAVE_TORCH = True


@unittest.skipUnless(HAVE_TORCH, "requires the cluster torch environment")
class BackboneSmokeTest(unittest.TestCase):
    BACKBONES = ("unet", "transformer", "mlp")

    def test_one_train_step_then_one_sampling_pass(self) -> None:
        for name in self.BACKBONES:
            with self.subTest(backbone=name):
                torch.manual_seed(0)
                policy = make_policy(name)
                rgb, proprio = make_observations(policy)
                actions = torch.randn(2, policy.pred_horizon, policy.action_dim)
                optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
                before = [parameter.detach().clone() for parameter in policy.parameters()]

                # one forward, one backward, one optimizer step
                optimizer.zero_grad(set_to_none=True)
                loss = policy.compute_loss(rgb, proprio, actions)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                optimizer.step()
                gradients = [
                    parameter.grad for parameter in policy.parameters() if parameter.grad is not None
                ]
                self.assertTrue(gradients, "backward produced no gradients")
                self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
                after = [parameter.detach().clone() for parameter in policy.parameters()]
                self.assertFalse(
                    all(torch.equal(previous, current) for previous, current in zip(before, after)),
                    "optimizer step did not update any parameter",
                )

                # one sampling pass through the shared DDPM scheduler
                policy.eval()
                with torch.no_grad():
                    sampled = policy.get_action(rgb, proprio)
                self.assertEqual(sampled.shape, (2, policy.act_horizon, policy.action_dim))
                self.assertTrue(torch.isfinite(sampled).all())


if __name__ == "__main__":
    unittest.main()
