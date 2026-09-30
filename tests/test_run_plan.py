"""Task 2 regression tests: trainer-consistent completion and a read-only plan.

The dry-run planner must answer "what would the trainer do for this run?" with
the same ``final.pt``/``run.json`` rule the trainer uses, without starting
training or touching run directories.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from dp_manip.completion import Completion, RunState, completion_state
from dp_manip.config import data_root_override
from dp_manip.runlist import Run, plan_runs, runs


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "configs" / "experiments" / "backbone.toml"
TASK = "peginsertionside"


def write_checkpoints(output_root: Path, run: Run) -> Path:
    """Create ``<run>/checkpoints`` under ``output_root`` and return it."""
    checkpoint_dir = run.directory(output_root) / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    return checkpoint_dir


def write_run_json(output_root: Path, run: Run, config: dict) -> Path:
    """Record ``config`` as a finished run's ``run.json``."""
    run_dir = run.directory(output_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "run.json"
    path.write_text(json.dumps({"config": config}) + "\n", encoding="utf-8")
    return path


def tree_snapshot(root: Path) -> list[tuple[str, int, int]]:
    """Names, sizes and mtimes of every file below ``root`` (empty if absent)."""
    if not root.exists():
        return []
    return sorted(
        (str(path.relative_to(root)), path.stat().st_size, path.stat().st_mtime_ns)
        for path in root.rglob("*")
    )


class CompletionStateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.run = runs(EXPERIMENT, TASK)[0]
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.output_root = Path(self._tmp.name)

    def completion(self) -> Completion:
        return completion_state(self.run.resolve(), self.run.directory(self.output_root))

    def test_missing_final_checkpoint_is_pending(self) -> None:
        completion = self.completion()
        self.assertIs(completion.state, RunState.PENDING)
        self.assertFalse(completion.resumable)
        self.assertIsNone(completion.reason)

    def test_matching_final_checkpoint_is_completed(self) -> None:
        checkpoint_dir = write_checkpoints(self.output_root, self.run)
        (checkpoint_dir / "final.pt").write_bytes(b"")
        write_run_json(self.output_root, self.run, self.run.resolve().to_dict())
        self.assertIs(self.completion().state, RunState.COMPLETED)

    def test_final_checkpoint_without_run_json_matches_the_trainer(self) -> None:
        # The trainer checks the recorded config only when run.json is present,
        # so the plan must not invent a stricter missing-run.json rule.
        checkpoint_dir = write_checkpoints(self.output_root, self.run)
        (checkpoint_dir / "final.pt").write_bytes(b"")
        self.assertIs(self.completion().state, RunState.COMPLETED)

    def test_resume_checkpoint_without_final_stays_pending(self) -> None:
        checkpoint_dir = write_checkpoints(self.output_root, self.run)
        (checkpoint_dir / "resume.pt").write_bytes(b"")
        completion = self.completion()
        self.assertIs(completion.state, RunState.PENDING)
        self.assertTrue(completion.resumable)

    def test_mismatched_final_checkpoint_is_a_conflict(self) -> None:
        recorded = self.run.resolve().to_dict()
        recorded["train"]["seed"] += 1
        checkpoint_dir = write_checkpoints(self.output_root, self.run)
        (checkpoint_dir / "final.pt").write_bytes(b"")
        write_run_json(self.output_root, self.run, recorded)
        completion = self.completion()
        self.assertIs(completion.state, RunState.CONFLICT)
        self.assertIn("different config", completion.reason)

    def test_lazily_recorded_final_checkpoint_is_completed(self) -> None:
        # data.preload changes how windows are read, never the samples, so a
        # run finished with lazy reads is not retrained now that preload is on.
        recorded = self.run.resolve().to_dict()
        self.assertTrue(recorded["data"]["preload"])
        del recorded["data"]["preload"]
        checkpoint_dir = write_checkpoints(self.output_root, self.run)
        (checkpoint_dir / "final.pt").write_bytes(b"")
        write_run_json(self.output_root, self.run, recorded)
        self.assertIs(self.completion().state, RunState.COMPLETED)

    def test_unreadable_run_json_is_a_conflict(self) -> None:
        checkpoint_dir = write_checkpoints(self.output_root, self.run)
        (checkpoint_dir / "final.pt").write_bytes(b"")
        (self.run.directory(self.output_root) / "run.json").write_text(
            "{not json", encoding="utf-8"
        )
        completion = self.completion()
        self.assertIs(completion.state, RunState.CONFLICT)
        self.assertIn("unreadable", completion.reason)


class RunPlanTest(unittest.TestCase):
    def test_plan_marks_only_the_finished_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            grid = runs(EXPERIMENT, TASK)
            finished = grid[2]
            checkpoint_dir = write_checkpoints(output_root, finished)
            (checkpoint_dir / "final.pt").write_bytes(b"")
            write_run_json(output_root, finished, finished.resolve().to_dict())

            planned = plan_runs(EXPERIMENT, TASK, output_root=output_root)
            self.assertEqual([item.run for item in planned], grid)
            states = {item.run.name: item.completion.state for item in planned}
            self.assertEqual(len(states), 15)
            self.assertIs(states[finished.name], RunState.COMPLETED)
            self.assertEqual(sum(state is RunState.PENDING for state in states.values()), 14)
            self.assertEqual(sum(state is RunState.CONFLICT for state in states.values()), 0)

    def test_plan_is_read_only_and_never_creates_the_output_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "does-not-exist"
            planned = plan_runs(EXPERIMENT, TASK, output_root=output_root)
            self.assertEqual(len(planned), 15)
            self.assertFalse(output_root.exists())

    def test_plan_does_not_modify_existing_run_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            run = runs(EXPERIMENT, TASK)[0]
            (write_checkpoints(output_root, run) / "resume.pt").write_bytes(b"")
            write_run_json(output_root, run, run.resolve().to_dict())
            before = tree_snapshot(output_root)
            plan_runs(EXPERIMENT, TASK, output_root=output_root)
            self.assertEqual(tree_snapshot(output_root), before)

    def test_resolve_applies_overrides_like_the_trainer(self) -> None:
        run = runs(EXPERIMENT, TASK)[0]
        resolved = run.resolve(("train.total_iters=200",))
        self.assertEqual(resolved.train.total_iters, 200)
        self.assertEqual(resolved.train.seed, run.seed)

    def test_overrides_participate_in_the_completion_check(self) -> None:
        # A smoke run trained with --set train.total_iters=200 is "completed"
        # only when the plan is given the same overrides; without them the
        # recorded config looks like a different run and becomes a conflict.
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            run = runs(EXPERIMENT, TASK)[0]
            checkpoint_dir = write_checkpoints(output_root, run)
            (checkpoint_dir / "final.pt").write_bytes(b"")
            write_run_json(output_root, run, run.resolve(("train.total_iters=200",)).to_dict())

            planned = plan_runs(
                EXPERIMENT,
                TASK,
                output_root=output_root,
                overrides=("train.total_iters=200",),
            )
            completed = next(item for item in planned if item.run.name == run.name)
            self.assertIs(completed.completion.state, RunState.COMPLETED)

            without = plan_runs(EXPERIMENT, TASK, output_root=output_root)
            conflicting = next(item for item in without if item.run.name == run.name)
            self.assertIs(conflicting.completion.state, RunState.CONFLICT)

    def test_data_root_participates_in_the_completion_check(self) -> None:
        # The Slurm entry points export DATA_ROOT as --data-root; the planner
        # must receive the same override or every finished run looks like a
        # config conflict and is queued again.
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            run = runs(EXPERIMENT, TASK)[0]
            data_root = "/scratch/user/dp-data/dataset"
            checkpoint_dir = write_checkpoints(output_root, run)
            (checkpoint_dir / "final.pt").write_bytes(b"")
            write_run_json(
                output_root,
                run,
                run.resolve((data_root_override(data_root),)).to_dict(),
            )

            planned = plan_runs(
                EXPERIMENT, TASK, output_root=output_root, data_root=data_root
            )
            completed = next(item for item in planned if item.run.name == run.name)
            self.assertIs(completed.completion.state, RunState.COMPLETED)

            without = plan_runs(EXPERIMENT, TASK, output_root=output_root)
            conflicting = next(item for item in without if item.run.name == run.name)
            self.assertIs(conflicting.completion.state, RunState.CONFLICT)


class SweepPlanCliTest(unittest.TestCase):
    def run_plan(
        self, output_root: Path, *extra: str
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "sweep.py"),
                "plan",
                "--experiment",
                str(EXPERIMENT),
                "--task",
                TASK,
                "--output-root",
                str(output_root),
                *extra,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

    def test_plan_reports_every_run_and_the_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            finished = runs(EXPERIMENT, TASK)[0]
            checkpoint_dir = write_checkpoints(output_root, finished)
            (checkpoint_dir / "final.pt").write_bytes(b"")
            write_run_json(output_root, finished, finished.resolve().to_dict())

            result = self.run_plan(output_root)
            self.assertEqual(result.returncode, 0)
            lines = result.stdout.strip().splitlines()
            self.assertEqual(len(lines), 16)  # 15 runs + one summary line
            self.assertIn("1 completed (skipped)", lines[-1])
            self.assertIn("14 pending", lines[-1])
            self.assertIn("0 conflict", lines[-1])
            status = {line.split()[1]: line.split()[2] for line in lines[:-1]}
            self.assertEqual(status[finished.name], "completed")
            self.assertEqual(sum(value == "completed" for value in status.values()), 1)

    def test_plan_accepts_the_data_root_the_slurm_scripts_export(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            finished = runs(EXPERIMENT, TASK)[0]
            data_root = "/scratch/user/dp-data/dataset"
            checkpoint_dir = write_checkpoints(output_root, finished)
            (checkpoint_dir / "final.pt").write_bytes(b"")
            write_run_json(
                output_root,
                finished,
                finished.resolve((data_root_override(data_root),)).to_dict(),
            )

            result = self.run_plan(output_root, "--data-root", data_root)
            self.assertEqual(result.returncode, 0)
            self.assertIn("1 completed (skipped)", result.stdout)

    def test_plan_shows_a_resumable_run_as_pending(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            run = runs(EXPERIMENT, TASK)[0]
            (write_checkpoints(output_root, run) / "resume.pt").write_bytes(b"")
            result = self.run_plan(output_root)
            self.assertEqual(result.returncode, 0)
            line = next(line for line in result.stdout.splitlines() if run.name in line)
            self.assertIn(" pending (resume.pt)", line)

    def test_plan_rejects_a_config_conflict_with_a_nonzero_exit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            run = runs(EXPERIMENT, TASK)[0]
            recorded = run.resolve().to_dict()
            recorded["train"]["total_iters"] += 1
            checkpoint_dir = write_checkpoints(output_root, run)
            (checkpoint_dir / "final.pt").write_bytes(b"")
            write_run_json(output_root, run, recorded)

            result = self.run_plan(output_root)
            self.assertEqual(result.returncode, 1)
            self.assertIn("conflict", result.stdout)
            self.assertIn("1 conflict", result.stdout)


class TrainerConsistencyTest(unittest.TestCase):
    def test_run_training_delegates_to_the_shared_completion_check(self) -> None:
        # run_training imports torch, which the local environment may not have,
        # so pin the call site textually: planner and trainer must not grow two
        # implementations of the final.pt/run.json reuse rule.
        source = (ROOT / "dp_manip" / "trainer.py").read_text(encoding="utf-8")
        self.assertIn("from .completion import RunState, completion_state", source)
        # Fine-tuning runs also pass their init record (dp_manip.finetune).
        self.assertIn("completion_state(cfg, run_dir, finetune=finetune_record)", source)


if __name__ == "__main__":
    unittest.main()
