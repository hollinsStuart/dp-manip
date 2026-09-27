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


if __name__ == "__main__":
    unittest.main()
