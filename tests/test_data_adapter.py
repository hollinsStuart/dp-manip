from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    import h5py
    import numpy as np
    from dp_manip.data import RGBWindowDataset, compute_normalization, read_dataset_info
except ModuleNotFoundError:
    h5py = None


@unittest.skipIf(h5py is None, "h5py/torch test dependencies are not installed")
class Hdf5AdapterTest(unittest.TestCase):
    def test_obs_rgb_state_maps_to_proprio_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demo.h5"
            with h5py.File(path, "w") as file:
                trajectory = file.create_group("traj_0")
                observations = trajectory.create_group("obs_rgb")
                observations.create_dataset("rgb", data=np.zeros((3, 8, 8, 3), dtype=np.uint8))
                observations.create_dataset(
                    "state",
                    data=np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
                )
                trajectory.create_dataset(
                    "actions", data=np.asarray([[0.1], [0.2]], dtype=np.float32)
                )
                trajectory.create_dataset("success", data=np.asarray([False, True]))
            metadata = {
                "env_info": {
                    "env_id": "PickCube-v1",
                    "env_kwargs": {"control_mode": "pd_ee_delta_pos"},
                },
                "episodes": [{"episode_id": 0, "episode_seed": 7}],
            }
            path.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")

            info = read_dataset_info(path)
            dataset = RGBWindowDataset(info, obs_horizon=2, pred_horizon=2)
            batch = dataset[0]
            self.assertEqual(set(batch), {"rgb", "proprio", "actions"})
            self.assertNotIn("state", batch)
            np.testing.assert_array_equal(batch["proprio"], np.asarray([[1.0, 2.0], [1.0, 2.0]]))

            stats = compute_normalization(info)
            np.testing.assert_allclose(stats.proprio_mean, [2.0, 3.0])
            self.assertEqual(
                set(stats.to_dict()),
                {"proprio_mean", "proprio_std", "action_low", "action_high"},
            )
            dataset.close()

    def test_preload_matches_lazy_reads_bitwise(self) -> None:
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demo.h5"
            episodes = []
            with h5py.File(path, "w") as file:
                for episode_id, length in enumerate((5, 9)):
                    trajectory = file.create_group(f"traj_{episode_id}")
                    observations = trajectory.create_group("obs_rgb")
                    # Same layout as the exporter: gzip, chunks spanning several frames.
                    observations.create_dataset(
                        "rgb",
                        data=rng.integers(0, 256, (length + 1, 8, 8, 6), dtype=np.uint8),
                        chunks=(3, 4, 4, 2),
                        compression="gzip",
                        compression_opts=5,
                    )
                    observations.create_dataset(
                        "state", data=rng.standard_normal((length + 1, 3)).astype(np.float32)
                    )
                    trajectory.create_dataset(
                        "actions", data=rng.standard_normal((length, 2)).astype(np.float32)
                    )
                    episodes.append({"episode_id": episode_id, "episode_seed": 100 + episode_id})
            metadata = {
                "env_info": {
                    "env_id": "PickCube-v1",
                    "env_kwargs": {"control_mode": "pd_ee_delta_pos"},
                },
                "episodes": episodes,
            }
            path.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")

            info = read_dataset_info(path)
            lazy = RGBWindowDataset(info, obs_horizon=2, pred_horizon=4, preload=False)
            preloaded = RGBWindowDataset(info, obs_horizon=2, pred_horizon=4)
            self.assertEqual(len(lazy), len(preloaded))
            for item in range(len(lazy)):
                expected, actual = lazy[item], preloaded[item]
                self.assertEqual(set(expected), set(actual))
                for key in expected:
                    self.assertEqual(expected[key].dtype, actual[key].dtype)
                    np.testing.assert_array_equal(expected[key], actual[key])
            # Preloaded windows never open the HDF5 file.
            self.assertIsNone(preloaded._file)
            lazy.close()


if __name__ == "__main__":
    unittest.main()
