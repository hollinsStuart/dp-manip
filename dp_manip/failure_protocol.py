"""Task-agnostic protocol of the failure-aware study (``configs/failure_aware``).

The protocol file is the pre-registered part of the design; per-task derived
values live in the task's lock file. :meth:`FailureProtocol.check_against`
ties the protocol to a resolved task config, so a seed range can never collide
with that task's expert, validation or test seeds.
"""

from __future__ import annotations

import dataclasses
import hashlib
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Config

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs" / "failure_aware" / "protocol.toml"

# Expert demonstrations use [0, 4000) for training and [4000, 5000) for
# validation (``trainer.check_dataset``); rollout seeds must stay above both.
EXPERT_SEED_END = 5_000


@dataclass(frozen=True)
class SeedRanges:
    collection_train: tuple[int, int]
    collection_holdout: tuple[int, int]
    guidance_tuning: tuple[int, int]


@dataclass(frozen=True)
class CollectionConfig:
    dataset_size: int
    rollout_cap: int
    holdout_episodes: int
    pilot_holdout_episodes: int
    failure_truncation_factor: float
    low_success_min: int


@dataclass(frozen=True)
class BaselineCellConfig:
    min_val_success: float
    max_val_success: float
    val_seeds: tuple[int, ...]
    checkpoint_seeds: tuple[int, ...]


@dataclass(frozen=True)
class PilotConfig:
    learning_rates: tuple[float, ...]
    steps: tuple[int, ...]
    relative_tie: float


@dataclass(frozen=True)
class GuidanceConfig:
    alpha_grid: tuple[float, ...]
    dry_run_episodes: int
    tuning_episodes: int
    tie_episodes: int


@dataclass(frozen=True)
class EvaluationConfig:
    # Leading test seeds of each checkpoint's recorded [eval] range used per
    # (arm, checkpoint): the whole range in the study, a few in a smoke run.
    test_episodes: int


@dataclass(frozen=True)
class FailureProtocol:
    seeds: SeedRanges
    collection: CollectionConfig
    baseline_cell: BaselineCellConfig
    pilot: PilotConfig
    guidance: GuidanceConfig
    evaluation: EvaluationConfig
    path: Path | None = None
    sha256: str | None = None

    def train_seeds(self) -> list[int]:
        """Candidate train-collection seeds, in the order they are rolled out."""
        start = self.seeds.collection_train[0]
        return list(range(start, start + self.collection.rollout_cap))

    def holdout_seeds(self) -> list[int]:
        start = self.seeds.collection_holdout[0]
        return list(range(start, start + self.collection.holdout_episodes))

    def test_seeds(self, cfg: Config) -> list[int]:
        """The test seeds every arm is evaluated on for a checkpoint with config ``cfg``."""
        return cfg.test_seeds()[: self.evaluation.test_episodes]

    def pilot_holdout_end(self) -> int:
        """First holdout seed that belongs to the offline gate, not the pilot."""
        return self.seeds.collection_holdout[0] + self.collection.pilot_holdout_episodes

    def validate(self) -> None:
        ranges = dataclasses.asdict(self.seeds)
        for name, (start, end) in ranges.items():
            if not EXPERT_SEED_END <= start < end:
                raise ValueError(f"seeds.{name} must satisfy {EXPERT_SEED_END} <= start < end")
        ordered = sorted(ranges.items(), key=lambda item: item[1][0])
        for (left, (_, left_end)), (right, (right_start, _)) in zip(ordered, ordered[1:]):
            if right_start < left_end:
                raise ValueError(f"seeds.{left} and seeds.{right} overlap")
        collection = self.collection
        if min(collection.dataset_size, collection.rollout_cap, collection.holdout_episodes) < 1:
            raise ValueError("collection sizes must be positive")
        if collection.dataset_size > collection.rollout_cap:
            raise ValueError("collection.dataset_size cannot exceed collection.rollout_cap")
        if collection.rollout_cap > _width(self.seeds.collection_train):
            raise ValueError("collection.rollout_cap exceeds seeds.collection_train")
        if collection.holdout_episodes > _width(self.seeds.collection_holdout):
            raise ValueError("collection.holdout_episodes exceeds seeds.collection_holdout")
        if not 0 < collection.pilot_holdout_episodes < collection.holdout_episodes:
            raise ValueError("collection.pilot_holdout_episodes must leave episodes for the gate")
        if collection.failure_truncation_factor < 1.0:
            raise ValueError("collection.failure_truncation_factor must be at least 1")
        if collection.low_success_min < 1:
            raise ValueError("collection.low_success_min must be positive")
        cell = self.baseline_cell
        if not 0.0 <= cell.min_val_success < cell.max_val_success <= 1.0:
            raise ValueError("baseline_cell bounds must satisfy 0 <= min < max <= 1")
        if not cell.checkpoint_seeds or not set(cell.checkpoint_seeds) <= set(cell.val_seeds):
            raise ValueError("baseline_cell.checkpoint_seeds must be a non-empty subset of val_seeds")
        pilot = self.pilot
        if not pilot.learning_rates or any(lr <= 0.0 for lr in pilot.learning_rates):
            raise ValueError("pilot.learning_rates must be positive")
        if not pilot.steps or any(step < 1 for step in pilot.steps):
            raise ValueError("pilot.steps must be positive")
        if not 0.0 <= pilot.relative_tie < 1.0:
            raise ValueError("pilot.relative_tie must be in [0, 1)")
        guidance = self.guidance
        if not guidance.alpha_grid or any(alpha <= 0.0 for alpha in guidance.alpha_grid):
            raise ValueError("guidance.alpha_grid must be positive")
        if not 0 < guidance.dry_run_episodes <= guidance.tuning_episodes:
            raise ValueError("guidance.dry_run_episodes must be in [1, tuning_episodes]")
        if guidance.tuning_episodes > _width(self.seeds.guidance_tuning):
            raise ValueError("guidance.tuning_episodes exceeds seeds.guidance_tuning")
        if guidance.tie_episodes < 0:
            raise ValueError("guidance.tie_episodes must be non-negative")
        if self.evaluation.test_episodes < 1:
            raise ValueError("evaluation.test_episodes must be positive")

    def check_against(self, cfg: Config) -> None:
        """Reject a task config whose rollout seeds or env count clash with the protocol."""
        evaluation = set(cfg.val_seeds()) | set(cfg.test_seeds())
        for name, (start, end) in dataclasses.asdict(self.seeds).items():
            clash = sorted(seed for seed in evaluation if start <= seed < end)
            if clash:
                raise ValueError(f"seeds.{name} overlaps {cfg.task.name} evaluation seeds {clash[:5]}")
        if self.evaluation.test_episodes > cfg.eval.test_episodes:
            raise ValueError(
                f"evaluation.test_episodes={self.evaluation.test_episodes} exceeds "
                f"{cfg.task.name}'s eval.test_episodes={cfg.eval.test_episodes}"
            )
        num_envs = cfg.eval.num_envs
        # ``evaluate`` runs full waves of num_envs episodes. The pilot/gate split
        # of the holdout happens after collection, so it need not divide.
        counts = {
            "collection.rollout_cap": self.collection.rollout_cap,
            "collection.holdout_episodes": self.collection.holdout_episodes,
            "guidance.dry_run_episodes": self.guidance.dry_run_episodes,
            "guidance.tuning_episodes": self.guidance.tuning_episodes,
            "evaluation.test_episodes": self.evaluation.test_episodes,
        }
        for name, count in counts.items():
            if count % num_envs:
                raise ValueError(f"{name}={count} is not divisible by eval.num_envs={num_envs}")


def _width(bounds: tuple[int, int]) -> int:
    return bounds[1] - bounds[0]


def _table(raw: dict[str, Any], name: str) -> dict[str, Any]:
    values = raw.get(name)
    if not isinstance(values, dict):
        raise ValueError(f"protocol section [{name}] is missing or not a table")
    return values


def _build(cls, values: dict[str, Any], name: str):
    known = {field.name for field in dataclasses.fields(cls)}
    if set(values) != known:
        raise ValueError(
            f"protocol [{name}] must define exactly {sorted(known)}, got {sorted(values)}"
        )
    converted = {
        key: tuple(tuple(item) if isinstance(item, list) else item for item in value)
        if isinstance(value, list)
        else value
        for key, value in values.items()
    }
    return cls(**converted)


def load_protocol(path: str | Path = DEFAULT_PROTOCOL) -> FailureProtocol:
    path = Path(path).resolve()
    payload = path.read_bytes()
    raw = tomllib.loads(payload.decode("utf-8"))
    sections = {
        "seeds": SeedRanges,
        "collection": CollectionConfig,
        "baseline_cell": BaselineCellConfig,
        "pilot": PilotConfig,
        "guidance": GuidanceConfig,
        "evaluation": EvaluationConfig,
    }
    unknown = set(raw) - set(sections)
    if unknown:
        raise ValueError(f"unknown protocol sections: {sorted(unknown)}")
    built = {name: _build(cls, _table(raw, name), name) for name, cls in sections.items()}
    seeds = built["seeds"]
    for name, bounds in dataclasses.asdict(seeds).items():
        if len(bounds) != 2 or not all(isinstance(value, int) for value in bounds):
            raise ValueError(f"seeds.{name} must be [start, end]")
    protocol = FailureProtocol(
        **built, path=path, sha256=hashlib.sha256(payload).hexdigest()
    )
    protocol.validate()
    return protocol
