"""Task 6/7 regression tests: the single-job dual-GPU Slurm entry points.

Neither script may use a job array; both request exactly 2 GPUs / 8 CPUs /
64 GB, forward their configuration through the environment to the matching
queue entry point and requeue the job when the queue reports interrupted runs
(exit 75). The behavior tests run the real scripts with a fake PYTHON and a
fake scontrol on PATH, so they need no Slurm, no GPU and no torch.
"""

from __future__ import annotations

import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "slurm" / "train_dual_gpu.sbatch"
EVAL_SCRIPT = ROOT / "slurm" / "eval_dual_gpu.sbatch"
LEGACY_ARRAY = ROOT / "slurm" / "train_array.sbatch"


def sbatch_directives(path: Path) -> dict[str, str]:
    directives: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#SBATCH "):
            key, _, value = line[len("#SBATCH ") :].partition("=")
            directives[key.strip()] = value.strip()
    return directives


class DualGpuShapeMixin:
    script: Path

    def setUp(self) -> None:
        self.text = self.script.read_text(encoding="utf-8")
        self.directives = sbatch_directives(self.script)

    def test_requests_exactly_two_gpus_eight_cpus_and_64gb(self) -> None:
        self.assertEqual(self.directives.get("--gres"), "gpu:2")
        self.assertEqual(self.directives.get("--cpus-per-task"), "8")
        self.assertEqual(self.directives.get("--mem"), "64G")

    def test_is_not_a_job_array(self) -> None:
        self.assertNotIn("--array", self.text)
        self.assertNotIn("%a", self.directives.get("--output", ""))

    def test_handles_preemption_and_requeue(self) -> None:
        self.assertIn("--requeue", self.directives)
        self.assertIn("USR1", self.directives.get("--signal", ""))
        self.assertIn("trap 'kill -USR1", self.text)
        self.assertIn("if [[ $status -eq 75 ]]", self.text)
        self.assertIn("scontrol requeue", self.text)
        self.assertIn("scontrol requeue failed", self.text)

    def test_forwards_python_and_requires_task(self) -> None:
        self.assertIn("PYTHON=${PYTHON:-.venv/bin/python}", self.text)
        self.assertIn("${TASK:?", self.text)


class TrainDualGpuShapeTest(DualGpuShapeMixin, unittest.TestCase):
    script = TRAIN_SCRIPT

    def test_forwards_task_experiment_data_root_run_root(self) -> None:
        for flag, variable in (
            ("--task", "TASK"),
            ("--experiment", "EXPERIMENT"),
            ("--data-root", "DATA_ROOT"),
            ("--output-root", "RUN_ROOT"),
        ):
            with self.subTest(flag=flag):
                self.assertIn(f'{flag} "${variable}"', self.text)
        self.assertIn("scripts/train_queue.py", self.text)

    def test_requires_data_root(self) -> None:
        self.assertIn("${DATA_ROOT:?", self.text)

    def test_legacy_array_entry_is_still_present(self) -> None:
        # The array workflow is kept until the README stops recommending it.
        self.assertIn("--array", sbatch_directives(LEGACY_ARRAY))


class EvalDualGpuShapeTest(DualGpuShapeMixin, unittest.TestCase):
    script = EVAL_SCRIPT

    def test_forwards_checkpoint_split_num_envs_and_run_root(self) -> None:
        for flag, variable in (
            ("--task", "TASK"),
            ("--experiment", "EXPERIMENT"),
            ("--output-root", "RUN_ROOT"),
            ("--checkpoint", "CHECKPOINT"),
            ("--split", "SPLIT"),
            ("--num-envs", "NUM_ENVS"),
        ):
            with self.subTest(flag=flag):
                self.assertIn(f'{flag} "${variable}"', self.text)
        self.assertIn("scripts/eval_queue.py", self.text)

    def test_keeps_the_array_workflow_defaults(self) -> None:
        self.assertIn("CHECKPOINT=${CHECKPOINT:-final.pt}", self.text)
        self.assertIn("SPLIT=${SPLIT:-test}", self.text)
        self.assertIn("NUM_ENVS=${NUM_ENVS:-4}", self.text)

    def test_does_not_require_the_training_dataset(self) -> None:
        self.assertNotIn("${DATA_ROOT:?", self.text)


class DualGpuBehaviorMixin:
    """Runs a real script with a fake PYTHON and a fake scontrol."""

    script: Path
    required_variables: tuple[str, ...] = ("TASK",)

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.fakebin = self.root / "bin"
        self.fakebin.mkdir()
        self.args_file = self.root / "python-args.txt"
        self.cwd_file = self.root / "python-cwd.txt"
        self.scontrol_file = self.root / "scontrol.txt"
        self.python = self.fake_script(
            "python",
            "#!/usr/bin/env bash\n"
            'printf \'%s\\n\' "$PWD" > "$FAKE_CWD"\n'
            'printf \'%s\\n\' "$@" > "$FAKE_ARGS"\n'
            'exit "${FAKE_EXIT:-0}"\n',
        )
        self.fake_script(
            "scontrol",
            "#!/usr/bin/env bash\n"
            'printf \'%s\\n\' "$@" >> "$FAKE_SCONTROL"\n'
            'exit "${FAKE_SCONTROL_EXIT:-0}"\n',
        )

    def fake_script(self, name: str, body: str) -> Path:
        path = self.fakebin / name
        path.write_text(body, encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
        return path

    def run_script(
        self, *, cwd: Path | None = None, arguments: tuple[str, ...] = (), **overrides
    ):
        environment = dict(os.environ)
        environment.update(
            {
                "PATH": f"{self.fakebin}:{environment.get('PATH', '')}",
                "PYTHON": str(self.python),
                "TASK": "peginsertionside",
                "EXPERIMENT": "configs/experiments/backbone.toml",
                "DATA_ROOT": "/tmp/intended-data",
                "RUN_ROOT": "/tmp/intended-runs",
                "CHECKPOINT": "step_010000.pt",
                "SPLIT": "val",
                "NUM_ENVS": "4",
                "SLURM_JOB_ID": "4242",
                "SLURM_SUBMIT_DIR": str(ROOT),
                "FAKE_ARGS": str(self.args_file),
                "FAKE_CWD": str(self.cwd_file),
                "FAKE_SCONTROL": str(self.scontrol_file),
            }
        )
        for key, value in overrides.items():
            if value is None:
                environment.pop(key, None)
            else:
                environment[key] = value
        return subprocess.run(
            ["bash", str(self.script), *arguments],
            cwd=cwd or ROOT,
            env=environment,
            capture_output=True,
            text=True,
        )

    def test_exit_75_requeues_the_job(self) -> None:
        result = self.run_script(FAKE_EXIT="75")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("requeue", result.stdout + result.stderr)
        self.assertEqual(
            self.scontrol_file.read_text(encoding="utf-8").split(), ["requeue", "4242"]
        )

    def test_failed_requeue_is_not_reported_as_success(self) -> None:
        # If scontrol requeue fails the interrupted runs were not rescheduled,
        # so the job must end non-zero instead of claiming success.
        result = self.run_script(FAKE_EXIT="75", FAKE_SCONTROL_EXIT="1")
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertEqual(
            self.scontrol_file.read_text(encoding="utf-8").split(), ["requeue", "4242"]
        )
        self.assertIn("scontrol requeue failed", result.stderr)
        self.assertIn("resubmit", result.stderr)

    def test_other_nonzero_exit_is_not_requeued(self) -> None:
        result = self.run_script(FAKE_EXIT="1")
        self.assertEqual(result.returncode, 1)
        self.assertFalse(self.scontrol_file.exists())

    def test_resolves_the_repository_root_without_slurm_submit_dir(self) -> None:
        result = self.run_script(cwd=self.root, SLURM_SUBMIT_DIR=None, FAKE_EXIT="0")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.cwd_file.read_text(encoding="utf-8").strip(), str(ROOT))

    def test_uses_the_normal_submit_dir(self) -> None:
        result = self.run_script(SLURM_SUBMIT_DIR=str(ROOT), FAKE_EXIT="0")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.cwd_file.read_text(encoding="utf-8").strip(), str(ROOT))

    def test_missing_required_variable_fails_before_training(self) -> None:
        for key in self.required_variables:
            with self.subTest(missing=key):
                result = self.run_script(**{key: None})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(key, result.stderr)
                self.assertFalse(self.args_file.exists())


class TrainDualGpuBehaviorTest(DualGpuBehaviorMixin, unittest.TestCase):
    script = TRAIN_SCRIPT
    required_variables = ("TASK", "DATA_ROOT")

    def test_forwards_every_argument_without_requeueing(self) -> None:
        result = self.run_script(FAKE_EXIT="0")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            self.args_file.read_text(encoding="utf-8").splitlines(),
            [
                "scripts/train_queue.py",
                "--task",
                "peginsertionside",
                "--experiment",
                "configs/experiments/backbone.toml",
                "--data-root",
                "/tmp/intended-data",
                "--output-root",
                "/tmp/intended-runs",
            ],
        )
        self.assertFalse(self.scontrol_file.exists())

    def test_forwards_extra_script_arguments_as_set_overrides(self) -> None:
        result = self.run_script(
            FAKE_EXIT="0",
            arguments=(
                "--set",
                "train.total_iters=200",
                "--set",
                "train.checkpoint_steps=[200]",
            ),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = self.args_file.read_text(encoding="utf-8").splitlines()
        self.assertEqual(
            arguments[-4:],
            ["--set", "train.total_iters=200", "--set", "train.checkpoint_steps=[200]"],
        )


class EvalDualGpuBehaviorTest(DualGpuBehaviorMixin, unittest.TestCase):
    script = EVAL_SCRIPT
    required_variables = ("TASK",)

    def test_forwards_every_argument_without_requeueing(self) -> None:
        result = self.run_script(FAKE_EXIT="0")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            self.args_file.read_text(encoding="utf-8").splitlines(),
            [
                "scripts/eval_queue.py",
                "--task",
                "peginsertionside",
                "--experiment",
                "configs/experiments/backbone.toml",
                "--output-root",
                "/tmp/intended-runs",
                "--checkpoint",
                "step_010000.pt",
                "--split",
                "val",
                "--num-envs",
                "4",
            ],
        )
        self.assertFalse(self.scontrol_file.exists())


if __name__ == "__main__":
    unittest.main()
