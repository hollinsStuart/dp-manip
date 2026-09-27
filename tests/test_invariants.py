"""Phase 13-14 regression tests: the declared-difference rules of a matrix.

``dp_manip.invariants`` is the single home of Gate B semantics: which keys may
differ between cells, how configs are diffed and pruned, and the control hash
recorded in every run.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from dp_manip.config import load, load_experiment
from dp_manip.invariants import (
    BACKBONE_VARIABLE,
    allowed_keys,
    config_differences,
    control_hash,
    experiment_context,
    without_keys,
)

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "configs" / "tasks"
EXPERIMENTS = ROOT / "configs" / "experiments"


class ConfigDifferenceTest(unittest.TestCase):
    def test_nested_differences_are_reported_by_dotted_path(self) -> None:
        left = {"train": {"seed": 1, "batch_size": 64}, "policy": {"backbone": "unet"}}
        right = {"train": {"seed": 2, "batch_size": 128}, "policy": {"backbone": "mlp"}}
        self.assertEqual(
            config_differences(left, right),
            {
                "policy.backbone": ("unet", "mlp"),
                "train.batch_size": (64, 128),
                "train.seed": (1, 2),
            },
        )

    def test_without_keys_prunes_only_the_allowed_paths(self) -> None:
        raw = {"train": {"seed": 1, "batch_size": 64}, "data": {"root": "/a", "num_demos": 100}}
        self.assertEqual(
            without_keys(raw, {"train.seed", "data.root"}),
            {"train": {"batch_size": 64}, "data": {"num_demos": 100}},
        )


class AllowedKeysTest(unittest.TestCase):
    def test_structural_keys_are_only_allowed_in_the_backbone_experiment(self) -> None:
        structural = {"policy.unet_dims", "policy.mlp_hidden_dim", "policy.transformer_layers"}
        backbone = allowed_keys(load_experiment(EXPERIMENTS / "backbone.toml"))
        data_size = allowed_keys(load_experiment(EXPERIMENTS / "data_size.toml"))
        self.assertTrue(structural <= backbone)
        self.assertFalse(structural & data_size)
        self.assertIn(BACKBONE_VARIABLE, backbone)
        self.assertIn("data.num_demos", data_size)
        self.assertIn("train.seed", data_size)
        self.assertIn("data.root", data_size)
        self.assertNotIn("train.batch_size", data_size)


class ControlHashTest(unittest.TestCase):
    def config(self, *overrides: str):
        return load(TASKS / "pickcube.toml", list(overrides))

    def test_control_hash_ignores_allowed_keys_only(self) -> None:
        allowed = {"train.seed", "policy.backbone"}
        left = self.config("train.seed=1")
        arm = self.config("train.seed=2", "policy.backbone=transformer")
        self.assertEqual(control_hash(left, allowed), control_hash(arm, allowed))
        drifted = self.config("train.seed=2", "train.batch_size=128")
        self.assertNotEqual(control_hash(left, allowed), control_hash(drifted, allowed))


class ExperimentContextTest(unittest.TestCase):
    def test_context_describes_the_declared_cell(self) -> None:
        experiment = EXPERIMENTS / "backbone.toml"
        spec = load_experiment(experiment)
        cfg = load(
            TASKS / "pickcube.toml", ["train.seed=3"], experiment=experiment, experiment_value="transformer"
        )
        context = experiment_context(cfg, spec, "transformer", spec_path=experiment)
        self.assertEqual(context["name"], "backbone")
        self.assertEqual(context["variable"], "policy.backbone")
        self.assertEqual(context["value"], "transformer")
        self.assertEqual(context["seed"], 3)
        self.assertIn("backbone.toml", context["spec"])
        self.assertEqual(len(context["control_hash"]), 64)

    def test_arms_share_one_control_hash(self) -> None:
        experiment = EXPERIMENTS / "backbone.toml"
        spec = load_experiment(experiment)
        hashes = set()
        for arm in spec.values:
            cfg = load(
                TASKS / "pickcube.toml", ["train.seed=1"], experiment=experiment, experiment_value=arm
            )
            hashes.add(experiment_context(cfg, spec, arm)["control_hash"])
        self.assertEqual(len(hashes), 1)

    def test_control_change_moves_the_hash(self) -> None:
        experiment = EXPERIMENTS / "backbone.toml"
        spec = load_experiment(experiment)
        base = load(TASKS / "pickcube.toml", experiment=experiment, experiment_value="unet")
        drifted = load(
            TASKS / "pickcube.toml",
            ["train.batch_size=32"],
            experiment=experiment,
            experiment_value="transformer",
        )
        self.assertNotEqual(
            experiment_context(base, spec, "unet")["control_hash"],
            experiment_context(drifted, spec, "transformer")["control_hash"],
        )


if __name__ == "__main__":
    unittest.main()
