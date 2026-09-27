"""Task 7 regression tests: the evaluation queue on fake commands.

Evaluation reuses the training run list, the dynamic queue and the per-worker
GPU environment; it keeps the checkpoint path, split and episode rules and
reports missing checkpoints as failures. Every job is a short fake script, so
these tests need no GPU, no torch and no ManiSkill.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from dp_manip.config import load_experiment
from dp_manip.runlist import checkpoint_path, eval_command, runs


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "configs" / "experiments" / "backbone.toml"
DATA_SIZE = ROOT / "configs" / "experiments" / "data_size.toml"
TASK = "peginsertionside"

FAKE_EVAL = '''
import json
import os
import sys
import time

name, record, sleep, code = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
visible = [device for device in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if device]
if len(visible) != 1 or visible[0] not in {"0", "1"}:
    print(f"{name}: expected exactly one GPU, saw {visible!r}", file=sys.stderr, flush=True)
    sys.exit(9)
time.sleep(sleep)
with open(record, "a", encoding="utf-8") as stream:
    stream.write(json.dumps({"name": name, "visible": visible[0]}) + "\\n")
sys.exit(code)
'''


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class EvalCommandTest(unittest.TestCase):
    def setUp(self) -> None:
        self.run = runs(EXPERIMENT, TASK)[0]
        self.output_root = ROOT / "runs"

    def test_checkpoint_split_and_environment_arguments(self) -> None:
        command = eval_command(
            self.run,
            output_root=self.output_root,
            checkpoint="step_010000.pt",
            split="val",
            num_envs=4,
            render_backend="cpu",
        )
        self.assertEqual(
            command[2],
            str(self.output_root / self.run.name / "checkpoints" / "step_010000.pt"),
        )
        self.assertEqual(command[command.index("--split") + 1], "val")
        self.assertEqual(command[command.index("--num-envs") + 1], "4")
        self.assertEqual(command[command.index("--render-backend") + 1], "cpu")
        self.assertNotIn("--episodes", command)

    def test_train_split_uses_the_experiment_diagnostic_budget(self) -> None:
        implicit = eval_command(self.run, output_root=self.output_root, split="train")
        self.assertEqual(
            implicit[implicit.index("--episodes") + 1],
            str(load_experiment(EXPERIMENT).train_eval_episodes),
        )
        explicit = eval_command(
            self.run, output_root=self.output_root, split="train", episodes=3
        )
        self.assertEqual(explicit[explicit.index("--episodes") + 1], "3")

    def test_checkpoint_path_matches_the_array_workflow(self) -> None:
        # The old sweep built <output-root>/<run>/checkpoints/<name> literally;
        # it must stay relative/unresolved so commands do not drift.
        run = runs(DATA_SIZE, "pickcube")[0]
        path = checkpoint_path(run, "runs", "final.pt")
        self.assertEqual(path, Path("runs") / run.name / "checkpoints" / "final.pt")
        self.assertFalse(path.is_absolute())


class EvalQueueTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.script = self.root / "fake_eval.py"
        self.script.write_text(FAKE_EVAL, encoding="utf-8")
        self.records = self.root / "records.jsonl"

    def checkpoint(self, run, name: str = "final.pt") -> Path:
        path = checkpoint_path(run, self.root, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
        return path

    def fake_builder(self, sleep: float = 0.0, fail: frozenset[str] = frozenset()):
        def build(run) -> list[str]:
            exit_code = "7" if run.name in fail else "0"
            return [
                sys.executable,
                str(self.script),
                run.name,
                str(self.records),
                str(sleep),
                exit_code,
            ]

        return build

    def queue(self, evaluation_runs, command_builder, **kwargs):
        module = load_script("eval_queue")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            summary = module.queue_evaluation(
                evaluation_runs,
                output_root=self.root,
                command_builder=command_builder,
                **kwargs,
            )
        return summary, output.getvalue()

    def read_records(self) -> dict[str, str]:
        return {
            entry["name"]: entry["visible"]
            for entry in map(json.loads, self.records.read_text(encoding="utf-8").splitlines())
        }

    def test_each_worker_sees_only_its_own_gpu_and_logs_separately(self) -> None:
        evaluation_runs = runs(EXPERIMENT, TASK)
        for run in evaluation_runs:
            self.checkpoint(run)
        summary, _ = self.queue(
            evaluation_runs, self.fake_builder(sleep=0.05), workers=2
        )

        self.assertEqual(len(summary.completed), 15)
        self.assertEqual(summary.pre_failed, ())
        self.assertEqual(summary.exit_code, 0)
        self.assertEqual({result.worker for result in summary.completed}, {0, 1})
        records = self.read_records()
        expected_dir = Path(self.root).resolve() / "logs" / "eval"
        for result in summary.completed:
            with self.subTest(run=result.job.name):
                self.assertEqual(records[result.job.name], str(result.worker))
                expected_log = expected_dir / f"{result.job.name}.log"
                self.assertEqual(result.log_path, expected_log)
                self.assertTrue(expected_log.is_file())

    def test_missing_checkpoints_fail_in_the_preflight(self) -> None:
        evaluation_runs = runs(EXPERIMENT, TASK)
        calls = []

        def builder(run) -> list[str]:
            calls.append(run.name)
            return [sys.executable, str(self.script), run.name, str(self.records), "0.0", "0"]

        summary, output = self.queue(evaluation_runs, builder)

        self.assertEqual(summary.pre_failed, tuple(run.name for run in evaluation_runs))
        self.assertEqual(summary.failed_count, 15)
        self.assertEqual(summary.exit_code, 1)
        self.assertEqual(calls, [])
        self.assertEqual(summary.results, ())
        self.assertIn("missing checkpoint", output)
        self.assertIn(f"missing checkpoint {checkpoint_path(evaluation_runs[0], self.root)}", output)
        # Nothing ran, so no evaluation logs were created.
        self.assertFalse((self.root / "logs").exists())

    def test_only_checkpointed_runs_are_queued(self) -> None:
        evaluation_runs = runs(EXPERIMENT, TASK)
        for run in evaluation_runs[:2]:
            self.checkpoint(run)
        summary, _ = self.queue(evaluation_runs, self.fake_builder(), workers=2)

        self.assertEqual(
            [result.job.name for result in summary.completed],
            [run.name for run in evaluation_runs[:2]],
        )
        self.assertEqual(
            summary.pre_failed, tuple(run.name for run in evaluation_runs[2:])
        )
        self.assertEqual(summary.failed_count, 13)
        self.assertEqual(summary.exit_code, 1)

    def test_failed_evaluation_does_not_stop_the_rest(self) -> None:
        evaluation_runs = runs(EXPERIMENT, TASK)[:3]
        for run in evaluation_runs:
            self.checkpoint(run)
        failing = evaluation_runs[1]
        summary, _ = self.queue(
            evaluation_runs, self.fake_builder(fail=frozenset({failing.name})), workers=2
        )
        self.assertEqual([result.job.name for result in summary.failed], [failing.name])
        self.assertEqual(len(summary.completed), 2)
        self.assertEqual(summary.exit_code, 1)

    def test_default_builder_uses_the_shared_eval_command(self) -> None:
        module = load_script("eval_queue")
        self.assertIs(module.eval_command, eval_command)
        for run in runs(EXPERIMENT, TASK)[:2]:
            self.checkpoint(run, "step_010000.pt")
        calls = []

        def spy(run, **kwargs):
            calls.append((run, kwargs))
            return [sys.executable, str(self.script), run.name, str(self.records), "0.0", "0"]

        evaluation_runs = runs(EXPERIMENT, TASK)[:2]
        with mock.patch.object(module, "eval_command", spy):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                summary = module.queue_evaluation(
                    evaluation_runs,
                    output_root=self.root,
                    checkpoint="step_010000.pt",
                    split="val",
                    num_envs=2,
                    render_backend="cpu",
                    workers=1,
                )
        self.assertEqual([run for run, _ in calls], [item for item in evaluation_runs])
        for _, kwargs in calls:
            self.assertEqual(Path(kwargs["output_root"]), self.root)
            self.assertEqual(kwargs["checkpoint"], "step_010000.pt")
            self.assertEqual(kwargs["split"], "val")
            self.assertEqual(kwargs["num_envs"], 2)
            self.assertEqual(kwargs["render_backend"], "cpu")
        self.assertEqual(len(summary.completed), 2)
        self.assertEqual(summary.exit_code, 0)

    def test_main_forwards_checkpoint_split_and_num_envs(self) -> None:
        module = load_script("eval_queue")
        captured = {}

        def fake_queue(evaluation_runs, **kwargs):
            captured["run_names"] = [run.name for run in evaluation_runs]
            captured.update(kwargs)
            return SimpleNamespace(exit_code=0)

        with mock.patch.object(module, "queue_evaluation", fake_queue):
            exit_code = module.main(
                [
                    "--task",
                    TASK,
                    "--checkpoint",
                    "step_010000.pt",
                    "--split",
                    "val",
                    "--num-envs",
                    "2",
                    "--output-root",
                    str(self.root),
                ]
            )
        self.assertEqual(exit_code, 0)
        self.assertEqual(captured["checkpoint"], "step_010000.pt")
        self.assertEqual(captured["split"], "val")
        self.assertEqual(captured["num_envs"], 2)
        self.assertEqual(captured["output_root"], self.root)
        self.assertEqual(captured["run_names"], [run.name for run in runs(module.DEFAULT_EXPERIMENT, TASK)])

    def test_cli_reads_checkpoint_split_and_num_envs_from_the_environment(self) -> None:
        module = load_script("eval_queue")
        environment = {"CHECKPOINT": "step_010000.pt", "SPLIT": "val", "NUM_ENVS": "4"}
        with mock.patch.dict(os.environ, environment):
            args = module.parse_args([])
        self.assertEqual(args.checkpoint, "step_010000.pt")
        self.assertEqual(args.split, "val")
        self.assertEqual(args.num_envs, 4)

    def test_cli_flags_override_the_environment(self) -> None:
        module = load_script("eval_queue")
        environment = {"CHECKPOINT": "step_010000.pt", "SPLIT": "val", "NUM_ENVS": "4"}
        with mock.patch.dict(os.environ, environment):
            args = module.parse_args(
                ["--checkpoint", "final.pt", "--split", "test", "--num-envs", "2"]
            )
        self.assertEqual(args.checkpoint, "final.pt")
        self.assertEqual(args.split, "test")
        self.assertEqual(args.num_envs, 2)

    def test_cli_falls_back_when_the_environment_is_unset(self) -> None:
        module = load_script("eval_queue")
        with mock.patch.dict(os.environ, {}, clear=True):
            args = module.parse_args([])
        self.assertEqual(args.checkpoint, "final.pt")
        self.assertEqual(args.split, "test")
        self.assertIsNone(args.num_envs)
        self.assertIsNone(args.episodes)
        self.assertEqual(args.workers, 2)
        self.assertEqual(args.output_root, ROOT / "runs")


if __name__ == "__main__":
    unittest.main()
