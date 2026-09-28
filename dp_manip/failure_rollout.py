"""Closed-loop rollout collection and success/failure datasets (plan §3).

Collection runs the baseline policy through :func:`dp_manip.evaluate.evaluate`
with a :class:`RolloutRecorder`, so recorded episodes come from the exact loop
that produces the reported metrics. Every rollout is written in full to a raw
HDF5 file; :func:`build_datasets` then selects the seed-ordered K-prefixes and
applies the truncation rules. All written files use the ``maniskill-demogen``
schema, so ``read_dataset_info`` and ``RGBWindowDataset`` load them unchanged.
Per-step success is stored as ``rollout_success`` rather than ``success``:
``read_dataset_info`` rejects groups whose ``success`` ends false, which is
correct for expert exports but not for failed rollouts.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .failure_protocol import FailureProtocol

RAW_TRAIN = "raw_train.h5"
RAW_HOLDOUT = "raw_holdout.h5"
DATASET_DIR = "datasets"
SUMMARY = "summary.json"


@dataclass(frozen=True)
class RolloutEpisode:
    seed: int
    rgb: np.ndarray  # (T+1, H, W, 3C) uint8
    proprio: np.ndarray  # (T+1, P) float32
    actions: np.ndarray  # (T, A) float32
    success: np.ndarray  # (T,) bool, info["success"] after each action
    reward: np.ndarray  # (T,) float32

    @property
    def length(self) -> int:
        return len(self.actions)

    @property
    def success_once(self) -> bool:
        return bool(self.success.any())

    @property
    def success_at_end(self) -> bool:
        return bool(self.success[-1]) if self.length else False

    @property
    def first_success_step(self) -> int | None:
        """Number of executed actions when success was first observed."""
        hits = np.flatnonzero(self.success)
        return int(hits[0]) + 1 if hits.size else None


class RolloutRecorder:
    """``evaluate`` observer that turns every finished wave into episodes.

    ``sink`` receives each episode in seed order; ``stop`` is asked after each
    wave whether collection is complete.
    """

    def __init__(
        self,
        sink: Callable[[RolloutEpisode], None],
        stop: Callable[[], bool] | None = None,
    ):
        self.sink = sink
        self.stop = stop
        self._seeds: list[int] = []
        self._frames: list[dict[str, list[Any]]] = []

    def on_reset(self, seeds: list[int], rgb: np.ndarray, proprio: np.ndarray) -> None:
        self._seeds = [int(seed) for seed in seeds]
        self._frames = [
            {
                "rgb": [np.array(rgb[index], dtype=np.uint8)],
                "proprio": [np.array(proprio[index], dtype=np.float32)],
                "actions": [],
                "success": [],
                "reward": [],
            }
            for index in range(len(seeds))
        ]

    def on_step(
        self,
        actions: np.ndarray,
        rgb: np.ndarray,
        proprio: np.ndarray,
        reward: np.ndarray,
        success: np.ndarray,
    ) -> None:
        for index, frames in enumerate(self._frames):
            frames["actions"].append(np.array(actions[index], dtype=np.float32))
            frames["rgb"].append(np.array(rgb[index], dtype=np.uint8))
            frames["proprio"].append(np.array(proprio[index], dtype=np.float32))
            frames["reward"].append(np.float32(reward[index]))
            frames["success"].append(bool(success[index]))

    def on_wave_end(self, episodes: list[dict]) -> bool:
        if [record["seed"] for record in episodes] != self._seeds:
            raise RuntimeError("recorder and evaluate disagree on the wave's seeds")
        for record, seed, frames in zip(episodes, self._seeds, self._frames):
            episode = RolloutEpisode(
                seed=seed,
                rgb=np.stack(frames["rgb"]),
                proprio=np.stack(frames["proprio"]),
                actions=np.stack(frames["actions"]),
                success=np.asarray(frames["success"], dtype=bool),
                reward=np.asarray(frames["reward"], dtype=np.float32),
            )
            if (
                episode.length != record["episode_len"]
                or episode.success_once != record["success_once"]
                or episode.success_at_end != record["success_at_end"]
            ):
                raise RuntimeError(f"recorded episode {seed} does not match evaluate's metrics")
            self.sink(episode)
        self._seeds, self._frames = [], []
        return bool(self.stop()) if self.stop is not None else False


class ClassQuota:
    """Stop rule for train collection: both classes have reached ``target`` (§3.4)."""

    def __init__(self, target: int):
        self.target = target
        self.successes = 0
        self.failures = 0

    def count(self, episode: RolloutEpisode) -> None:
        if episode.success_once:
            self.successes += 1
        else:
            self.failures += 1

    def reached(self) -> bool:
        return self.successes >= self.target and self.failures >= self.target


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _export_info_path(path: Path) -> Path:
    return path.with_name(path.stem + ".export_info.json")


def _write_group(
    file: h5py.File,
    episode_id: int,
    rgb: np.ndarray,
    proprio: np.ndarray,
    actions: np.ndarray,
    success: np.ndarray,
    reward: np.ndarray,
) -> None:
    group = file.create_group(f"traj_{episode_id}")
    # Same RGB compression as the maniskill-demogen exporter.
    group.create_dataset("obs_rgb/rgb", data=rgb, compression="gzip", compression_opts=5)
    group.create_dataset("obs_rgb/state", data=proprio)
    group.create_dataset("actions", data=actions)
    group.create_dataset("rollout_success", data=success)
    group.create_dataset("rollout_reward", data=reward)


class RolloutWriter:
    """Write episodes to ``<path>`` with dataset sidecars, atomically on success.

    The HDF5 file is written as ``<path>.partial`` and renamed only after every
    episode is in; the JSON sidecars are written last, so a complete sidecar
    always describes a complete file.
    """

    def __init__(
        self,
        path: Path,
        *,
        env_info: dict[str, Any],
        export_info: dict[str, Any],
        overwrite: bool = False,
    ):
        self.path = Path(path)
        if self.path.exists() and not overwrite:
            raise FileExistsError(f"{self.path} exists; pass overwrite=True to replace it")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.env_info = env_info
        self.export_info = dict(export_info)
        self.entries: list[dict[str, Any]] = []
        self._partial = self.path.with_name(self.path.name + ".partial")
        self._file: h5py.File | None = h5py.File(self._partial, "w")
        self._image_shape: tuple[int, ...] | None = None

    def add(
        self,
        episode: RolloutEpisode,
        *,
        stored_length: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Write ``episode``, keeping its first ``stored_length`` actions."""
        if self._file is None:
            raise RuntimeError("writer is closed")
        length = episode.length if stored_length is None else stored_length
        if not 0 < length <= episode.length:
            raise ValueError(f"stored length {length} outside [1, {episode.length}]")
        episode_id = len(self.entries)
        _write_group(
            self._file,
            episode_id,
            episode.rgb[: length + 1],
            episode.proprio[: length + 1],
            episode.actions[:length],
            episode.success[:length],
            episode.reward[:length],
        )
        image_shape = tuple(int(value) for value in episode.rgb.shape[1:])
        if self._image_shape is None:
            self._image_shape = image_shape
        elif image_shape != self._image_shape:
            raise ValueError("image shape changed between episodes")
        self.entries.append(
            {
                "episode_id": episode_id,
                "episode_seed": episode.seed,
                "success_once": episode.success_once,
                "success_at_end": episode.success_at_end,
                "first_success_step": episode.first_success_step,
                "rollout_length": episode.length,
                "stored_length": length,
                "return": float(episode.reward.sum()),
                **(extra or {}),
            }
        )

    def close(self, provenance: dict[str, Any]) -> None:
        if self._file is None:
            return
        self._file.close()
        self._file = None
        os.replace(self._partial, self.path)
        export_info = dict(self.export_info)
        if self._image_shape is not None:
            export_info["obs_rgb_image_shape"] = list(self._image_shape)
        _write_json(_export_info_path(self.path), export_info)
        _write_json(
            self.path.with_suffix(".json"),
            {"env_info": self.env_info, "episodes": self.entries, "rollout_provenance": provenance},
        )

    def abort(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
        self._partial.unlink(missing_ok=True)


def collect(
    policy,
    envs,
    seeds: Sequence[int],
    device,
    *,
    inference_seed: int,
    writer: RolloutWriter,
    quota: ClassQuota | None = None,
) -> dict:
    """Roll out ``seeds`` in order, writing every episode in full.

    With a ``quota``, collection stops after the first wave at which both
    classes have reached it. Returns ``evaluate``'s result for the waves run.
    """
    from .evaluate import evaluate  # torch is only needed when collecting

    def sink(episode: RolloutEpisode) -> None:
        writer.add(episode)
        if quota is not None:
            quota.count(episode)

    recorder = RolloutRecorder(sink, quota.reached if quota is not None else None)
    return evaluate(
        policy, envs, seeds, device, inference_seed=inference_seed, observer=recorder
    )


def success_storage_length(first_success_step: int, length: int, act_horizon: int) -> int:
    """End of the executed action chunk that contains the first success (§3.3)."""
    if first_success_step < 1:
        raise ValueError("first_success_step must be positive")
    return min(length, math.ceil(first_success_step / act_horizon) * act_horizon)


def failure_storage_length(
    success_lengths: Sequence[int], factor: float, act_horizon: int, max_steps: int
) -> int:
    """``L_fail = min(max_steps, ceil_to_act_horizon(factor * median))`` (§3.3).

    Without any success to measure, failures are kept in full.
    """
    if not success_lengths:
        return max_steps
    target = factor * float(np.median(success_lengths))
    # Guard against float noise such as 1.5 * 96 landing just above 144.
    chunks = math.ceil(target / act_horizon - 1e-9)
    return min(max_steps, chunks * act_horizon)


def _read_raw(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sidecar = path.with_suffix(".json")
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"raw rollouts are missing or incomplete: {path}")
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    entries = sorted(meta["episodes"], key=lambda entry: int(entry["episode_seed"]))
    return meta, entries


def _load_episode(file: h5py.File, entry: dict[str, Any]) -> RolloutEpisode:
    group = file[f"traj_{int(entry['episode_id'])}"]
    return RolloutEpisode(
        seed=int(entry["episode_seed"]),
        rgb=np.asarray(group["obs_rgb/rgb"]),
        proprio=np.asarray(group["obs_rgb/state"]),
        actions=np.asarray(group["actions"]),
        success=np.asarray(group["rollout_success"], dtype=bool),
        reward=np.asarray(group["rollout_reward"], dtype=np.float32),
    )


def _stored_length(entry: dict[str, Any], act_horizon: int, l_fail: int) -> int:
    length = int(entry["rollout_length"])
    if entry["success_once"]:
        return success_storage_length(int(entry["first_success_step"]), length, act_horizon)
    return min(length, l_fail)


def build_datasets(rollout_dir: Path, protocol: FailureProtocol, *, overwrite: bool = False) -> dict:
    """Build the train/pilot/gate success and failure datasets of one checkpoint.

    Reads ``raw_train.h5`` and ``raw_holdout.h5`` from ``rollout_dir`` and
    writes ``datasets/{failure,success}_{train,pilot,gate}.h5`` plus
    ``datasets/summary.json``. Empty holdout subsets are not written.
    """
    rollout_dir = Path(rollout_dir)
    train_meta, train_entries = _read_raw(rollout_dir / RAW_TRAIN)
    holdout_meta, holdout_entries = _read_raw(rollout_dir / RAW_HOLDOUT)
    train_source = train_meta["rollout_provenance"]
    holdout_source = holdout_meta["rollout_provenance"]
    if train_source["source_checkpoint"]["sha256"] != holdout_source["source_checkpoint"]["sha256"]:
        raise ValueError("train and holdout rollouts come from different checkpoints")
    if train_source["protocol"]["sha256"] != protocol.sha256:
        raise ValueError("raw train rollouts were collected under a different protocol file")
    if holdout_source["protocol"]["sha256"] != protocol.sha256:
        raise ValueError("raw holdout rollouts were collected under a different protocol file")
    act_horizon = int(train_source["act_horizon"])
    max_steps = int(train_source["max_episode_steps"])
    collection = protocol.collection

    successes = [entry for entry in train_entries if entry["success_once"]]
    failures = [entry for entry in train_entries if not entry["success_once"]]
    k = min(collection.dataset_size, len(successes), len(failures))
    if k < 1:
        raise ValueError(
            f"cannot build datasets: {len(successes)} successes and {len(failures)} failures"
        )
    successes, failures = successes[:k], failures[:k]
    success_lengths = [
        success_storage_length(int(entry["first_success_step"]), int(entry["rollout_length"]), act_horizon)
        for entry in successes
    ]
    l_fail = failure_storage_length(
        success_lengths, collection.failure_truncation_factor, act_horizon, max_steps
    )

    pilot_end = protocol.pilot_holdout_end()
    subsets = {
        "train": {"success": successes, "failure": failures},
        "pilot": {
            label: [
                entry
                for entry in holdout_entries
                if bool(entry["success_once"]) == (label == "success")
                and int(entry["episode_seed"]) < pilot_end
            ]
            for label in ("success", "failure")
        },
        "gate": {
            label: [
                entry
                for entry in holdout_entries
                if bool(entry["success_once"]) == (label == "success")
                and int(entry["episode_seed"]) >= pilot_end
            ]
            for label in ("success", "failure")
        },
    }

    output = rollout_dir / DATASET_DIR
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / SUMMARY
    if summary_path.exists() and not overwrite:
        raise FileExistsError(f"{summary_path} exists; pass overwrite=True to rebuild")
    derived = {
        "dataset_size_requested": collection.dataset_size,
        "dataset_size": k,
        "failure_truncation_factor": collection.failure_truncation_factor,
        "l_fail": l_fail,
        "act_horizon": act_horizon,
        "max_episode_steps": max_steps,
    }
    written: dict[str, Any] = {}
    for split, by_label in subsets.items():
        raw_path = rollout_dir / (RAW_TRAIN if split == "train" else RAW_HOLDOUT)
        raw_meta = train_meta if split == "train" else holdout_meta
        export_info = json.loads(_export_info_path(raw_path).read_text(encoding="utf-8"))
        with h5py.File(raw_path, "r") as raw:
            for label, entries in by_label.items():
                name = f"{label}_{split}"
                if not entries:
                    written[name] = {"episodes": 0}
                    continue
                writer = RolloutWriter(
                    output / f"{name}.h5",
                    env_info=raw_meta["env_info"],
                    export_info={**export_info, "dataset_type": f"rollout_{label}", "split": split},
                    overwrite=overwrite,
                )
                total_steps = stored_steps = 0
                try:
                    for entry in entries:
                        stored = _stored_length(entry, act_horizon, l_fail)
                        writer.add(
                            _load_episode(raw, entry),
                            stored_length=stored,
                            extra={"source_episode_id": int(entry["episode_id"])},
                        )
                        total_steps += int(entry["rollout_length"])
                        stored_steps += stored
                except BaseException:
                    writer.abort()
                    raise
                writer.close(
                    {
                        **raw_meta["rollout_provenance"],
                        "dataset_type": f"rollout_{label}",
                        "split": split,
                        "source_raw": str(raw_path.resolve()),
                        **derived,
                    }
                )
                written[name] = {
                    "episodes": len(entries),
                    "seeds": [int(entry["episode_seed"]) for entry in entries],
                    "rollout_steps": total_steps,
                    "stored_steps": stored_steps,
                    "fraction_truncated": 1.0 - stored_steps / total_steps,
                }

    train_successes = sum(1 for entry in train_entries if entry["success_once"])
    summary = {
        "source_checkpoint": train_source["source_checkpoint"],
        "protocol": train_source["protocol"],
        "train_rollouts": len(train_entries),
        "train_successes": train_successes,
        "train_failures": len(train_entries) - train_successes,
        "train_success_rate": train_successes / len(train_entries),
        "low_success": train_successes < collection.low_success_min,
        **derived,
        "datasets": written,
    }
    _write_json(summary_path, summary)
    return summary
