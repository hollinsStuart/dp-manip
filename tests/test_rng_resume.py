"""Phase 5 regression tests: a requeued run must follow the continuous trajectory.

A Slurm preemption writes ``resume.pt`` and requeues the job. For the result to
be independent of how often that happens, the resumed process must reproduce
the batch and the RNG draws that a continuous run would have used at the same
optimizer step. These tests pin that property down at the training-utility
level; the full trainer is exercised on the cluster.
"""

from __future__ import annotations

import copy
import random
import unittest

try:
    import numpy as np
    import torch

    from dp_manip.training import (
        ExponentialMovingAverage,
        StepSeededIndexSampler,
        resume_checkpoint,
        rng_state,
        set_rng_state,
        step_seed,
    )
except ModuleNotFoundError:  # torch is only installed in the cluster environment
    HAVE_TORCH = False
    CUDA_AVAILABLE = False
else:
    HAVE_TORCH = True
    CUDA_AVAILABLE = torch.cuda.is_available()


@unittest.skipUnless(HAVE_TORCH, "requires the cluster torch environment")
class RngStateTest(unittest.TestCase):
    def test_round_trip_reproduces_every_stream(self) -> None:
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)
        state = rng_state()
        first = (random.random(), float(np.random.rand()), torch.rand(3).tolist())
        self.assertTrue(set_rng_state(state))
        second = (random.random(), float(np.random.rand()), torch.rand(3).tolist())
        self.assertEqual(first, second)

    @unittest.skipUnless(CUDA_AVAILABLE, "requires a CUDA device")
    def test_cuda_state_round_trip(self) -> None:
        torch.cuda.manual_seed(2)
        state = rng_state()
        self.assertIn("torch_cuda", state)
        first = torch.rand(4, device="cuda")
        self.assertTrue(set_rng_state(state))
        second = torch.rand(4, device="cuda")
        self.assertTrue(torch.equal(first, second))

    def test_absent_state_is_reported(self) -> None:
        self.assertFalse(set_rng_state(None))

    def test_incomplete_state_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            set_rng_state({"python": random.getstate()})


@unittest.skipUnless(HAVE_TORCH, "requires the cluster torch environment")
class StepSeededIndexSamplerTest(unittest.TestCase):
    def batches(self, sampler) -> list[list[int]]:
        indices = list(sampler)
        return [
            indices[offset : offset + sampler.batch_size]
            for offset in range(0, len(indices), sampler.batch_size)
        ]

    def test_same_seed_and_step_produce_same_batch(self) -> None:
        left = self.batches(StepSeededIndexSampler(100, 8, 5, first_step=1, last_step=4))
        right = self.batches(StepSeededIndexSampler(100, 8, 5, first_step=1, last_step=4))
        self.assertEqual(left, right)

    def test_resume_suffix_matches_continuous_stream(self) -> None:
        continuous = self.batches(StepSeededIndexSampler(100, 8, 5, first_step=1, last_step=6))
        resumed = self.batches(StepSeededIndexSampler(100, 8, 5, first_step=4, last_step=6))
        self.assertEqual(continuous[3:], resumed)

    def test_batch_depends_on_step_not_on_sampler_creation(self) -> None:
        # A sampler created at resume time must not restart the stream.
        full = self.batches(StepSeededIndexSampler(50, 4, 9, first_step=1, last_step=10))
        late = self.batches(StepSeededIndexSampler(50, 4, 9, first_step=7, last_step=10))
        self.assertEqual(full[6:], late)

    def test_seed_changes_stream(self) -> None:
        left = list(StepSeededIndexSampler(100, 8, 1, first_step=1, last_step=3))
        right = list(StepSeededIndexSampler(100, 8, 2, first_step=1, last_step=3))
        self.assertNotEqual(left, right)

    def test_length_and_index_range(self) -> None:
        sampler = StepSeededIndexSampler(10, 4, 3, first_step=1, last_step=5)
        values = list(sampler)
        self.assertEqual(len(sampler), 20)
        self.assertEqual(len(values), 20)
        self.assertTrue(all(0 <= value < 10 for value in values))

    def test_exhausted_budget_yields_nothing(self) -> None:
        sampler = StepSeededIndexSampler(10, 4, 3, first_step=6, last_step=5)
        self.assertEqual(len(sampler), 0)
        self.assertEqual(list(sampler), [])

    def test_step_seed_is_stable_and_step_dependent(self) -> None:
        self.assertEqual(step_seed(3, 7), step_seed(3, 7))
        self.assertNotEqual(step_seed(3, 7), step_seed(3, 8))
        self.assertNotEqual(step_seed(3, 7), step_seed(4, 7))


@unittest.skipUnless(HAVE_TORCH, "requires the cluster torch environment")
class ResumeTrajectoryTest(unittest.TestCase):
    """Continuous training and preempt-resume must end with identical weights."""

    BATCH_SIZE = 8
    DATA_SIZE = 64
    FEATURES = 6
    TOTAL_STEPS = 6
    PREEMPT_STEP = 3
    SEED = 11

    def data(self) -> torch.Tensor:
        generator = torch.Generator().manual_seed(1234)
        return torch.randn(self.DATA_SIZE, self.FEATURES, generator=generator)

    def model_and_optimizer(self):
        torch.manual_seed(7)
        model = torch.nn.Linear(self.FEATURES, 3)
        return model, torch.optim.AdamW(model.parameters(), lr=0.05)

    @staticmethod
    def loss(model, data: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # Consume the global RNG the way the diffusion loss does.
        noise = torch.randn(batch.shape[0], 3)
        return torch.nn.functional.mse_loss(model(data[batch]), noise)

    def batches(self, first_step: int, last_step: int) -> list[torch.Tensor]:
        sampler = StepSeededIndexSampler(
            self.DATA_SIZE, self.BATCH_SIZE, seed=self.SEED, first_step=first_step, last_step=last_step
        )
        indices = torch.tensor(list(sampler), dtype=torch.long)
        return [indices[offset : offset + self.BATCH_SIZE] for offset in range(0, len(indices), self.BATCH_SIZE)]

    def train(self, model, optimizer, data: torch.Tensor, batches: list[torch.Tensor]) -> None:
        for batch in batches:
            optimizer.zero_grad()
            self.loss(model, data, batch).backward()
            optimizer.step()

    @staticmethod
    def scramble_rng() -> None:
        # A requeued job is a fresh process: its RNG streams carry no memory of
        # the preempted run until resume.pt is restored.
        random.seed(999)
        np.random.seed(999)
        torch.manual_seed(999)

    def resume(self, data: torch.Tensor, checkpoint: dict, *, restore_rng: bool) -> list[torch.Tensor]:
        model, optimizer = self.model_and_optimizer()
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        self.scramble_rng()
        if restore_rng:
            self.assertTrue(set_rng_state(checkpoint["rng"]))
        self.train(model, optimizer, data, self.batches(self.PREEMPT_STEP + 1, self.TOTAL_STEPS))
        return [parameter.detach().clone() for parameter in model.parameters()]

    def test_preempted_run_matches_continuous_run(self) -> None:
        data = self.data()
        model, optimizer = self.model_and_optimizer()
        checkpoint = None
        for step, batch in enumerate(self.batches(1, self.TOTAL_STEPS), start=1):
            self.train(model, optimizer, data, [batch])
            if step == self.PREEMPT_STEP:
                # Deep copies: training keeps mutating parameters and Adam
                # moments in place after the "preemption".
                checkpoint = {
                    "model": copy.deepcopy(model.state_dict()),
                    "optimizer": copy.deepcopy(optimizer.state_dict()),
                    "rng": rng_state(),
                }
        continuous = [parameter.detach().clone() for parameter in model.parameters()]
        self.assertIsNotNone(checkpoint)

        resumed = self.resume(data, checkpoint, restore_rng=True)
        for expected, actual in zip(continuous, resumed):
            self.assertTrue(torch.equal(expected, actual), "resumed weights diverged from continuous training")

        # Guard against a vacuous pass: without the restore, the post-resume
        # noise draws differ and so must the weights.
        unrestored = self.resume(data, checkpoint, restore_rng=False)
        self.assertFalse(
            all(torch.equal(expected, actual) for expected, actual in zip(continuous, unrestored)),
            "test cannot detect a missing RNG restore",
        )

    def test_resume_checkpoint_carries_rng_state(self) -> None:
        model, optimizer = self.model_and_optimizer()
        ema = ExponentialMovingAverage(model, 0.9)
        ema.update(model)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        payload = resume_checkpoint(
            config={"train": {"seed": self.SEED}},
            step=self.PREEMPT_STEP,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            ema=ema,
        )
        self.assertEqual(payload["format_version"], 3)
        self.assertTrue({"python", "numpy", "torch_cpu"} <= set(payload["rng"]))
        self.assertTrue(set_rng_state(payload["rng"]))


if __name__ == "__main__":
    unittest.main()
