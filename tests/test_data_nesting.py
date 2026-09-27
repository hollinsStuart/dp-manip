"""Automatic check of the data-size nested subset invariant."""

from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path
from typing import Any

try:
    import h5py
    import numpy as np
    from dp_manip.data import read_dataset_info
except ModuleNotFoundError:
    h5py = None


DATA_SIZE_GRID = (25, 50, 100, 200)
FULL_POOL = 200


@unittest.skipIf(h5py is None, "h5py/torch test dependencies are not installed")
class NestedSubsetTest(unittest.TestCase):
    """N demos must always be the first N demos ordered by demonstration seed."""

    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.path = Path(self._temporary.name) / "train.h5"
        with h5py.File(self.path, "w") as file:
            for episode_id in range(FULL_POOL):
                group = file.create_group(f"traj_{episode_id}")
                observations = group.create_group("obs_rgb")
                observations.create_dataset("rgb", data=np.zeros((2, 2, 2, 3), dtype=np.uint8))
                observations.create_dataset("state", data=np.zeros((2, 3), dtype=np.float32))
                group.create_dataset("actions", data=np.zeros((1, 4), dtype=np.float32))
                group.create_dataset("success", data=np.asarray([False, True]))
        # Seeds run opposite to episode_id: an episode-id-ordered subset fails.
        self.metadata: dict[str, Any] = {
            "env_info": {
                "env_id": "PickCube-v1",
                "env_kwargs": {"control_mode": "pd_ee_delta_pos"},
            },
            "episodes": [
                {"episode_id": episode_id, "episode_seed": FULL_POOL - 1 - episode_id}
                for episode_id in range(FULL_POOL)
            ],
        }
        self.write_metadata(self.metadata)

    def write_metadata(self, metadata: dict[str, Any]) -> None:
        self.path.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")

    def test_grid_sizes_select_seed_sorted_prefixes(self) -> None:
        for num_demos in DATA_SIZE_GRID:
            with self.subTest(num_demos=num_demos):
                info = read_dataset_info(self.path, num_demos)
                self.assertEqual(info.seeds, list(range(num_demos)))

    def test_subsets_are_nested(self) -> None:
        selections = {
            num_demos: set(read_dataset_info(self.path, num_demos).seeds)
            for num_demos in DATA_SIZE_GRID
        }
        for smaller, larger in zip(DATA_SIZE_GRID, DATA_SIZE_GRID[1:]):
            with self.subTest(smaller=smaller, larger=larger):
                self.assertEqual(len(selections[smaller]), smaller)
                self.assertLessEqual(selections[smaller], selections[larger])

    def test_sidecar_listing_order_does_not_change_subset(self) -> None:
        expected = read_dataset_info(self.path, 25).seeds
        shuffled = dict(self.metadata)
        shuffled["episodes"] = list(self.metadata["episodes"])
        random.Random(0).shuffle(shuffled["episodes"])
        self.write_metadata(shuffled)
        self.assertEqual(read_dataset_info(self.path, 25).seeds, expected)

    def test_duplicate_episode_ids_are_rejected(self) -> None:
        metadata = dict(self.metadata)
        metadata["episodes"] = [dict(entry) for entry in self.metadata["episodes"]]
        metadata["episodes"][1]["episode_id"] = metadata["episodes"][0]["episode_id"]
        self.write_metadata(metadata)
        with self.assertRaisesRegex(ValueError, "episode_id values must be unique"):
            read_dataset_info(self.path, 25)

    def test_duplicate_seeds_are_rejected(self) -> None:
        metadata = dict(self.metadata)
        metadata["episodes"] = [dict(entry) for entry in self.metadata["episodes"]]
        metadata["episodes"][1]["episode_seed"] = metadata["episodes"][0]["episode_seed"]
        self.write_metadata(metadata)
        with self.assertRaisesRegex(ValueError, "duplicate demonstration seeds: \\[199\\]"):
            read_dataset_info(self.path, 25)

    def test_missing_seed_is_rejected(self) -> None:
        metadata = dict(self.metadata)
        metadata["episodes"] = [dict(entry) for entry in self.metadata["episodes"]]
        metadata["episodes"][37].pop("episode_seed")
        self.write_metadata(metadata)
        with self.assertRaisesRegex(ValueError, "has no reset seed"):
            read_dataset_info(self.path, 25)


if __name__ == "__main__":
    unittest.main()
