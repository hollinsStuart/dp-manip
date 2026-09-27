"""Training utilities shared by the trainer and the tests.

EMA, the step-seeded sampler, RNG state capture and restartable checkpoint
payloads live here; the training loop itself is ``dp_manip.trainer.run_training``.
"""

from __future__ import annotations

import contextlib
import math
import random
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Sampler


class ExponentialMovingAverage:
    """EMA with the same early-step decay ramp used by the prior DP baseline."""

    def __init__(self, model: nn.Module, target_decay: float):
        self.target_decay = target_decay
        self.num_updates = 0
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    def update(self, model: nn.Module) -> None:
        self.num_updates += 1
        decay = min(self.target_decay, (1 + self.num_updates) / (10 + self.num_updates))
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in self.shadow:
                    self.shadow[name].lerp_(parameter.detach(), 1.0 - decay)

    @contextlib.contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        parameters = dict(model.named_parameters())
        backup = {name: parameters[name].detach().clone() for name in self.shadow}
        try:
            with torch.no_grad():
                for name, value in self.shadow.items():
                    parameters[name].copy_(value)
            yield
        finally:
            with torch.no_grad():
                for name, value in backup.items():
                    parameters[name].copy_(value)

    def state_dict(self) -> dict:
        return {
            "target_decay": self.target_decay,
            "num_updates": self.num_updates,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state: dict, model: nn.Module) -> None:
        self.target_decay = float(state["target_decay"])
        self.num_updates = int(state["num_updates"])
        parameters = dict(model.named_parameters())
        self.shadow = {
            name: value.to(device=parameters[name].device, dtype=parameters[name].dtype)
            for name, value in state["shadow"].items()
        }


# Stable 63-bit mixers for deriving one sampler seed per optimizer step. Any
# expression that depends only on ``(training seed, step)`` would work; fixed
# odd multipliers avoid the correlation of the naive ``seed + step``.
_BATCH_SEED_MULTIPLIER = 0x9E3779B97F4A7C15
_STEP_SEED_MULTIPLIER = 0xBF58476D1CE4E5B9
_SEED_MASK = (1 << 63) - 1


def step_seed(seed: int, step: int) -> int:
    """Return the deterministic sampler seed for one optimizer step."""
    return (seed * _BATCH_SEED_MULTIPLIER + step * _STEP_SEED_MULTIPLIER) & _SEED_MASK


class StepSeededIndexSampler(Sampler[int]):
    """Draw with-replacement indices from a fresh per-step seed.

    Slurm can preempt a run at any moment and DataLoader workers prefetch
    batches beyond the last optimizer step, so a stateful ``RandomSampler``
    cannot promise that a resumed run sees the same batch at the same step as a
    continuous run. Deriving every batch from ``(seed, step)`` makes the index
    stream a pure function of the step counter: the number of preemptions no
    longer changes the stochastic trajectory.
    """

    def __init__(
        self,
        num_samples: int,
        batch_size: int,
        seed: int,
        first_step: int,
        last_step: int,
    ):
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if first_step < 1:
            raise ValueError("first_step must be positive")
        if last_step < first_step - 1:
            raise ValueError("last_step must not precede first_step - 1")
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.seed = seed
        self.first_step = first_step
        self.last_step = last_step

    def __iter__(self) -> Iterator[int]:
        for step in range(self.first_step, self.last_step + 1):
            generator = torch.Generator().manual_seed(step_seed(self.seed, step))
            batch = torch.randint(self.num_samples, (self.batch_size,), generator=generator)
            yield from batch.tolist()

    def __len__(self) -> int:
        return max(0, self.last_step - self.first_step + 1) * self.batch_size


def rng_state() -> dict[str, Any]:
    """Capture every RNG stream a training step can consume."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: Mapping[str, Any] | None) -> bool:
    """Restore :func:`rng_state` output; return ``False`` when it is absent."""
    if state is None:
        return False
    missing = [key for key in ("python", "numpy", "torch_cpu") if key not in state]
    if missing:
        raise ValueError(f"RNG state is missing required entries: {missing}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # ``torch.load(map_location=device)`` may have moved the CPU byte tensors;
    # set_rng_state expects them on the CPU even when resuming on a GPU.
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = state.get("torch_cuda")
    if cuda_states is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_states])
    return True


def resume_checkpoint(
    *,
    config: dict[str, Any],
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    ema: ExponentialMovingAverage,
) -> dict[str, Any]:
    """Build the restartable checkpoint written by the training loop.

    Unlike the inference checkpoints, this payload captures the full training
    state including RNG streams, so a requeued run continues the trajectory of
    a continuous run from the same optimizer step.
    """
    return {
        "format_version": 3,
        "config": config,
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "ema": ema.state_dict(),
        "rng": rng_state(),
    }


def cosine_warmup(step: int, *, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def atomic_torch_save(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def averaged_state_dict(model: nn.Module, ema: ExponentialMovingAverage) -> dict[str, torch.Tensor]:
    with ema.average_parameters(model):
        return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
