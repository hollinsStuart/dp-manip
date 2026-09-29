"""M17 regression test: the full checkpoint lifecycle on synthetic data.

Plan §20 requires a ``train -> save -> load -> resume -> evaluate`` test. The
closed-loop evaluation itself needs ManiSkill (cluster only), so this test pins
everything up to a real policy forward pass: a tiny synthetic HDF5 dataset, a
two-step run through the shared trainer, loading ``final.pt`` exactly like
``scripts/eval_dp.py`` does, sampling actions, and then resuming from
``resume.pt``.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    import h5py
    import numpy as np
    import torch

    from dp_manip import config as config_lib
    from dp_manip.policy import DiffusionPolicy
    from dp_manip.trainer import run_training
except ModuleNotFoundError:  # torch is only installed in the cluster environment
    HAVE_TORCH = False
else:
    HAVE_TORCH = True

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "configs" / "tasks" / "pickcube.toml"


def write_dataset(data_root: Path, relative: str, seeds: list[int]) -> None:
    rng = np.random.default_rng(0)
    path = data_root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    episodes = []
    with h5py.File(path, "w") as file:
        for episode_id, seed in enumerate(seeds):
            group = file.create_group(f"traj_{episode_id}")
            group.create_dataset(
                "obs_rgb/rgb", data=rng.integers(0, 256, (6, 64, 64, 3), dtype=np.uint8)
            )
            group.create_dataset("obs_rgb/state", data=rng.normal(size=(6, 5)).astype(np.float32))
            group.create_dataset(
                "actions", data=rng.uniform(-1.0, 1.0, size=(5, 4)).astype(np.float32)
            )
            group.create_dataset("success", data=np.ones(6, dtype=bool))
            episodes.append({"episode_id": episode_id, "episode_seed": int(seed)})
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "env_info": {
                    "env_id": "PickCube-v1",
                    "env_kwargs": {"control_mode": "pd_ee_delta_pos"},
                },
                "episodes": episodes,
            }
        ),
        encoding="utf-8",
    )


def smoke_config(data_root: Path):
    cfg = config_lib.load(
        str(TASK),
        [
            f"data.root={data_root}",
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
    write_dataset(data_root, cfg.data.train_path, [0, 1, 2, 3])
    write_dataset(data_root, cfg.data.val_path, [4000])
    return cfg


@unittest.skipUnless(HAVE_TORCH, "requires the cluster torch environment")
class CheckpointLifecycleTest(unittest.TestCase):
    def test_train_load_sample_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = smoke_config(root / "data")
            output_root = root / "runs"
            self.assertEqual(run_training(cfg, output_root=output_root, device="cpu"), 0)

            run_dir = output_root / config_lib.default_run_name(cfg)
            final_path = run_dir / "checkpoints" / "final.pt"
            resume_path = run_dir / "checkpoints" / "resume.pt"
            self.assertTrue(final_path.is_file())
            self.assertTrue(resume_path.is_file())

            # Load exactly the way scripts/eval_dp.py does and run one real
            # sampling pass through the shared DDPM scheduler.
            checkpoint = torch.load(final_path, map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["step"], 2)
            self.assertIn("sidecar_sha256", checkpoint["train_data"]["fingerprint"])
            policy = DiffusionPolicy.from_checkpoint(checkpoint, "cpu").eval()
            generator = torch.Generator().manual_seed(cfg.eval.inference_seed)
            rgb = torch.randint(0, 256, (2, policy.obs_horizon, 3, 64, 64), dtype=torch.uint8)
            proprio = torch.randn(2, policy.obs_horizon, 5)
            with torch.no_grad():
                features = policy.observation_features(rgb, proprio)
                actions = policy.get_action(rgb, proprio, generator=generator)
            self.assertEqual(features.shape, (2, policy.obs_horizon, 8 + 5))
            self.assertEqual(actions.shape, (2, policy.act_horizon, 4))
            self.assertTrue(torch.isfinite(actions).all())

            # Resume: removing only final.pt makes the trainer continue from
            # resume.pt at the interrupted step and finish the budget.
            final_path.unlink()
            self.assertEqual(run_training(cfg, output_root=output_root, device="cpu"), 0)
            resumed = torch.load(final_path, map_location="cpu", weights_only=False)
            self.assertEqual(resumed["step"], 2)

    def test_finished_run_with_another_config_is_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = smoke_config(root / "data")
            output_root = root / "runs"
            self.assertEqual(run_training(cfg, output_root=output_root, device="cpu"), 0)

            run_dir = output_root / config_lib.default_run_name(cfg)
            run_info_path = run_dir / "run.json"
            run_info = json.loads(run_info_path.read_text(encoding="utf-8"))
            run_info["config"]["train"]["seed"] += 1
            run_info_path.write_text(json.dumps(run_info), encoding="utf-8")

            # final.pt belongs to a different config, so the trainer must refuse
            # instead of silently reusing it (the planner reports this conflict).
            with self.assertRaises(FileExistsError):
                run_training(cfg, output_root=output_root, device="cpu")

    def test_resume_never_accepts_only_a_fresh_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = smoke_config(root / "data")
            output_root = root / "runs"
            # A fresh directory trains, although the trainer creates checkpoints/ itself.
            self.assertEqual(run_training(cfg, output_root=output_root, device="cpu", resume="never"), 0)

            # Leftovers from an interrupted run (no final.pt) are rejected, not resumed.
            run_dir = output_root / config_lib.default_run_name(cfg)
            (run_dir / "checkpoints" / "final.pt").unlink()
            (run_dir / "checkpoints" / "resume.pt").unlink()
            with self.assertRaisesRegex(FileExistsError, "is not empty"):
                run_training(cfg, output_root=output_root, device="cpu", resume="never")


if __name__ == "__main__":
    unittest.main()
