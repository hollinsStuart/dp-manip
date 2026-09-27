"""Phase 12 regression tests: one resolver, one trainer, two entry points.

The unified entry point only resolves the canonical config layering and
delegates to the shared trainer; it must not branch on the experiment name.
These tests pin the resolution behavior, the string CLI values of every grid,
and the equivalence with the sweep cells and the legacy training CLI.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

from dp_manip.config import load_experiment, match_experiment_value

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class ExperimentValueMatchingTest(unittest.TestCase):
    def test_declared_values_match_by_string_form(self) -> None:
        data_size = load_experiment(ROOT / "configs" / "experiments" / "data_size.toml")
        self.assertEqual(match_experiment_value(data_size, "50"), 50)
        self.assertEqual(match_experiment_value(data_size, 50), 50)
        backbone = load_experiment(ROOT / "configs" / "experiments" / "backbone.toml")
        self.assertEqual(match_experiment_value(backbone, "transformer"), "transformer")

    def test_unknown_values_are_rejected(self) -> None:
        backbone = load_experiment(ROOT / "configs" / "experiments" / "backbone.toml")
        with self.assertRaisesRegex(ValueError, "is not in experiment"):
            match_experiment_value(backbone, "banana")


class RunExperimentResolutionTest(unittest.TestCase):
    def resolve(self, *argv):
        module = load_script("run_experiment")
        return module.resolve_config(module.parse_args(list(argv)))

    def test_data_size_cell(self) -> None:
        cfg = self.resolve(
            "--task", "pickcube", "--experiment", "data_size", "--value", "50", "--seed", "2"
        )
        self.assertEqual(cfg.data.num_demos, 50)
        self.assertEqual(cfg.train.seed, 2)
        self.assertEqual(cfg.policy.backbone, "unet")

    def test_backbone_arms(self) -> None:
        for arm in ("unet", "transformer", "mlp"):
            with self.subTest(arm=arm):
                cfg = self.resolve(
                    "--task", "pickcube", "--experiment", "backbone", "--value", arm, "--seed", "3"
                )
                self.assertEqual(cfg.policy.backbone, arm)
                self.assertEqual(cfg.train.seed, 3)
                self.assertEqual(cfg.train.batch_size, 64)  # still the baseline budget

    def test_hard_task_n_b_override(self) -> None:
        cfg = self.resolve(
            "--task", "pickcube", "--experiment", "backbone", "--value", "mlp", "--num-demos", "200"
        )
        self.assertEqual(cfg.data.num_demos, 200)

    def test_matches_the_sweep_cell_config(self) -> None:
        sweep = load_script("sweep")
        run = next(
            run
            for run in sweep.runs(sweep.DEFAULT_EXPERIMENT)
            if run.task == "pickcube" and run.value == 50 and run.seed == 2
        )
        unified = self.resolve(
            "--task", "pickcube", "--experiment", "data_size", "--value", "50", "--seed", "2"
        )
        self.assertEqual(unified.to_dict(), run.resolve().to_dict())

    def test_both_entry_points_resolve_the_same_cell(self) -> None:
        train_dp = load_script("train_dp")
        run_experiment = load_script("run_experiment")
        legacy = train_dp.resolve_config(
            train_dp.parse_args(
                [
                    "--config",
                    str(ROOT / "configs" / "tasks" / "pickcube.toml"),
                    "--experiment",
                    str(ROOT / "configs" / "experiments" / "data_size.toml"),
                    "--experiment-value",
                    "50",
                    "--seed",
                    "2",
                ]
            )
        )
        unified = run_experiment.resolve_config(
            run_experiment.parse_args(
                ["--task", "pickcube", "--experiment", "data_size", "--value", "50", "--seed", "2"]
            )
        )
        self.assertEqual(legacy.to_dict(), unified.to_dict())

    def test_new_experiment_is_config_only(self) -> None:
        # The runner has no code path per experiment: a brand-new spec defined
        # outside configs/experiments resolves through the same layering.
        spec_text = """
[experiment]
name = "batch_size"
variable = "train.batch_size"
values = [16, 32]

[replicates]
"16" = [1]
"32" = [1]
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "batch_size.toml"
            path.write_text(spec_text, encoding="utf-8")
            cfg = self.resolve("--task", "pickcube", "--experiment", str(path), "--value", "32")
            self.assertEqual(cfg.train.batch_size, 32)

    def test_unknown_task_experiment_and_value_are_rejected(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "unknown task"):
            self.resolve("--task", "banana", "--experiment", "data_size", "--value", "50")
        with self.assertRaisesRegex(FileNotFoundError, "unknown experiment"):
            self.resolve("--task", "pickcube", "--experiment", "banana", "--value", "50")
        with self.assertRaisesRegex(ValueError, "is not in experiment"):
            self.resolve("--task", "pickcube", "--experiment", "backbone", "--value", "banana")


if __name__ == "__main__":
    unittest.main()
