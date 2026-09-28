"""Phase 2 of the failure-aware plan: fine-tuning through the shared trainer.

A tiny baseline is trained on synthetic expert data, synthetic rollout datasets
are written with that checkpoint's hash as their source, and
``scripts/finetune_dp.py`` fine-tunes it. The checks are the plan's Phase 2
acceptance criteria: only the noise predictor trains, the encoder and the
normalization stay bit-identical, the checkpoint reloads with the baseline's
schema, the constant schedule is recorded, and mismatched inputs are rejected.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import h5py
    import numpy as np
    import torch

    from dp_manip import config as config_lib
    from dp_manip.failure_rollout import RolloutEpisode, RolloutWriter
    from dp_manip.finetune import freeze_modules
    from dp_manip.metadata import file_sha256
    from dp_manip.policy import DiffusionPolicy
    from dp_manip.trainer import run_training
    from dp_manip.training import constant_warmup
except ModuleNotFoundError:  # torch is only installed in the cluster environment
    HAVE_TORCH = False
else:
    HAVE_TORCH = True

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "configs" / "tasks" / "peginsertionside.toml"
ENV_INFO = {"env_id": "PegInsertionSide-v1", "env_kwargs": {"control_mode": "pd_joint_pos"}}
IMAGE = (64, 64, 3)
PROPRIO_DIM = 5
ACTION_DIM = 8


def load_script():
    path = ROOT / "scripts" / "finetune_dp.py"
    spec = importlib.util.spec_from_file_location("finetune_dp", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_expert(path: Path, seeds: list[int]) -> None:
    rng = np.random.default_rng(0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as file:
        for episode_id in range(len(seeds)):
            group = file.create_group(f"traj_{episode_id}")
            group.create_dataset("obs_rgb/rgb", data=rng.integers(0, 256, (7, *IMAGE), dtype=np.uint8))
            group.create_dataset("obs_rgb/state", data=rng.normal(size=(7, PROPRIO_DIM)).astype(np.float32))
            group.create_dataset("actions", data=rng.uniform(-1, 1, (6, ACTION_DIM)).astype(np.float32))
            group.create_dataset("success", data=np.ones(7, dtype=bool))
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "env_info": ENV_INFO,
                "episodes": [
                    {"episode_id": index, "episode_seed": seed} for index, seed in enumerate(seeds)
                ],
            }
        )
    )


def write_rollouts(path: Path, seeds: list[int], source_sha: str, *, success: bool) -> None:
    rng = np.random.default_rng(seeds[0])
    writer = RolloutWriter(path, env_info=ENV_INFO, export_info={"cameras": ["camera_0"]}, overwrite=True)
    for seed in seeds:
        writer.add(
            RolloutEpisode(
                seed=seed,
                rgb=rng.integers(0, 256, (9, *IMAGE), dtype=np.uint8),
                proprio=rng.normal(size=(9, PROPRIO_DIM)).astype(np.float32),
                # Offset actions so the fine-tuning data differs from the expert data.
                actions=(rng.uniform(-1, 1, (8, ACTION_DIM)) * 0.5 + 0.3).astype(np.float32),
                success=np.array([False] * 7 + [success]),
                reward=np.zeros(8, dtype=np.float32),
            )
        )
    writer.close({"source_checkpoint": {"sha256": source_sha}})


@unittest.skipUnless(HAVE_TORCH, "requires the cluster torch environment")
class FinetuneTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        root = Path(cls.temporary.name)
        data_root = root / "data"
        cfg = config_lib.load(
            str(TASK),
            [
                f"data.root={json.dumps(str(data_root))}",
                "data.num_demos=4",
                "data.val_num_demos=1",
                "vision.feature_dim=8",
                "policy.unet_dims=[16, 32]",
                "policy.kernel_size=3",
                "policy.n_groups=4",
                "diffusion.num_diffusion_iters=4",
                "diffusion.num_inference_iters=2",
                "train.total_iters=2",
                "train.batch_size=2",
                "train.num_workers=0",
                "train.log_freq=1",
                "train.resume_freq=100",
                "train.validation_steps=[2]",
                "train.checkpoint_steps=[]",
                "train.amp=false",
                "ema.decay=0.9",
            ],
        )
        write_expert(data_root / cfg.data.train_path, [0, 1, 2, 3])
        write_expert(data_root / cfg.data.val_path, [4000])
        runs = root / "runs"
        assert run_training(cfg, output_root=runs, device="cpu") == 0
        cls.baseline_path = runs / config_lib.default_run_name(cfg) / "checkpoints" / "final.pt"
        cls.baseline_sha = file_sha256(cls.baseline_path)
        cls.rollout_dir = root / "rollouts"
        cls.script = load_script()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def setUp(self) -> None:
        datasets = self.rollout_dir / "datasets"
        write_rollouts(datasets / "failure_train.h5", [20_001, 20_003, 20_005], self.baseline_sha, success=False)
        write_rollouts(datasets / "failure_pilot.h5", [21_001], self.baseline_sha, success=False)

    def finetune(self, *extra: str, exp: str = "ft") -> int:
        return self.script.main(
            [
                str(self.baseline_path),
                "--label", "failure",
                "--lr", "1e-3",
                "--steps", "3",
                "--rollout-dir", str(self.rollout_dir),
                "--exp", exp,
                "--device", "cpu",
                "--set", "train.warmup_steps=1",
                *extra,
            ]
        )

    def load(self, path: Path) -> dict:
        return torch.load(path, map_location="cpu", weights_only=False)

    def test_only_the_noise_predictor_changes(self) -> None:
        self.assertEqual(self.finetune(exp="ft_main"), 0)
        run_dir = self.rollout_dir / "ft_main"
        baseline = self.load(self.baseline_path)
        tuned = self.load(run_dir / "checkpoints" / "final.pt")

        self.assertEqual(tuned["normalization"], baseline["normalization"])
        self.assertEqual(tuned["model"].keys(), baseline["model"].keys())
        changed = []
        for key, value in baseline["model"].items():
            if key.startswith("observation_encoder.") or key in ("action_low", "action_high"):
                self.assertTrue(torch.equal(tuned["model"][key], value), key)
            elif not torch.equal(tuned["model"][key], value):
                changed.append(key)
        self.assertTrue(changed and all(key.startswith("noise_predictor.") for key in changed))

        record = tuned["finetune"]
        self.assertEqual(record["init_checkpoint_sha256"], self.baseline_sha)
        self.assertEqual(record["spec"]["lr_schedule"], "constant_with_warmup")
        self.assertEqual(record["spec"]["frozen_modules"], ["observation_encoder"])
        self.assertEqual(record["spec"]["train_seed_range"], [20_000, 21_000])

        run_info = json.loads((run_dir / "run.json").read_text())
        self.assertEqual(run_info["finetune"], record)
        self.assertTrue(run_info["frozen_parameters"])
        self.assertTrue(all(name.startswith("observation_encoder.") for name in run_info["frozen_parameters"]))
        self.assertEqual(run_info["config"]["train"]["lr"], 1e-3)
        for section in ("task", "vision", "policy", "diffusion", "eval"):
            self.assertEqual(run_info["config"][section], baseline["config"][section], section)
        lrs = [json.loads(line)["lr"] for line in (run_dir / "metrics.jsonl").read_text().splitlines() if "lr" in line]
        self.assertEqual(lrs, [1e-3] * len(lrs))

        policy = DiffusionPolicy.from_checkpoint(tuned, "cpu").eval()
        rgb = torch.randint(0, 256, (2, policy.obs_horizon, 3, 64, 64), dtype=torch.uint8)
        actions = policy.get_action(rgb, torch.randn(2, policy.obs_horizon, PROPRIO_DIM))
        self.assertEqual(actions.shape, (2, policy.act_horizon, ACTION_DIM))

        # Same invocation again: the finished run is recognised and not retrained.
        self.assertEqual(self.finetune(exp="ft_main"), 0)

        # A preempted fine-tune resumes from resume.pt, whose fine-tuning
        # record must match, and reproduces the same final weights.
        final_path = run_dir / "checkpoints" / "final.pt"
        final_path.unlink()
        resume = self.load(run_dir / "checkpoints" / "resume.pt")
        self.assertEqual(resume["finetune"], record)
        self.assertEqual(self.finetune(exp="ft_main"), 0)
        resumed = self.load(final_path)
        for key, value in tuned["model"].items():
            self.assertTrue(torch.equal(resumed["model"][key], value), key)

    def test_frozen_encoder_receives_no_gradient(self) -> None:
        policy = DiffusionPolicy.from_checkpoint(self.load(self.baseline_path), "cpu").train()
        frozen = freeze_modules(policy, ["observation_encoder"])
        self.assertTrue(frozen)
        rgb = torch.randint(0, 256, (2, policy.obs_horizon, 3, 64, 64), dtype=torch.uint8)
        proprio = torch.randn(2, policy.obs_horizon, PROPRIO_DIM)
        policy.compute_loss(rgb, proprio, torch.randn(2, policy.pred_horizon, ACTION_DIM)).backward()
        for name, parameter in policy.named_parameters():
            if name.startswith("observation_encoder."):
                self.assertIsNone(parameter.grad, name)
            else:
                self.assertIsNotNone(parameter.grad, name)
        with self.assertRaisesRegex(ValueError, "no parameters"):
            freeze_modules(policy, ["not_a_module"])

    def test_rollouts_from_another_checkpoint_are_rejected(self) -> None:
        write_rollouts(
            self.rollout_dir / "datasets" / "failure_train.h5", [20_001, 20_003], "0" * 64, success=False
        )
        with self.assertRaisesRegex(ValueError, "collected from checkpoint"):
            self.finetune(exp="ft_other_source")

    def test_rollout_seeds_outside_the_declared_range_are_rejected(self) -> None:
        write_rollouts(
            self.rollout_dir / "datasets" / "failure_train.h5", [3_001, 3_003], self.baseline_sha, success=False
        )
        with self.assertRaisesRegex(ValueError, r"seeds must lie in \[20000, 21000\)"):
            self.finetune(exp="ft_bad_seeds")

    def test_locked_sections_cannot_be_overridden(self) -> None:
        with self.assertRaisesRegex(ValueError, "policy.kernel_size"):
            self.finetune("--set", "policy.kernel_size=5", exp="ft_locked")

    def test_constant_schedule_holds_after_warmup(self) -> None:
        self.assertEqual([constant_warmup(step, warmup_steps=4) for step in range(6)], [0.25, 0.5, 0.75, 1.0, 1.0, 1.0])


class RecordedOverridesTest(unittest.TestCase):
    def test_overrides_apply_to_a_recorded_config(self) -> None:
        from dp_manip import config as config_lib

        base = config_lib.load(TASK).to_dict()
        tuned = config_lib.from_recorded(base, ["train.lr=1e-05", "train.total_iters=20000"])
        self.assertEqual(tuned.train.lr, 1e-5)
        self.assertEqual(tuned.train.total_iters, 20_000)
        self.assertEqual(tuned.policy, config_lib.from_recorded(base).policy)


if __name__ == "__main__":
    unittest.main()
