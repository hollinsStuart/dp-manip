"""Task 4 regression tests: sweep runs through the GPU queue.

The queue entry point must consume the sweep's run list unchanged, skip
completed runs, and give each worker only its own GPU. Training is faked with a
short script that records ``CUDA_VISIBLE_DEVICES`` and fails if it sees anything
other than exactly one valid device, so these tests need no GPU, no torch and no
real dataset.
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

from dp_manip.completion import RunState
from dp_manip.runlist import plan_runs, runs, train_command
from dp_manip.scheduler import gpu_environment


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "configs" / "experiments" / "backbone.toml"
DATA_SIZE = ROOT / "configs" / "experiments" / "data_size.toml"
TASK = "peginsertionside"

FAKE_TRAIN = '''
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


class TrainQueueTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.script = self.root / "fake_train.py"
        self.script.write_text(FAKE_TRAIN, encoding="utf-8")
        self.records = self.root / "records.jsonl"

    def plan(self):
        return plan_runs(EXPERIMENT, TASK, output_root=self.root)

    def fake_builder(self, sleep: float = 0.0, fail: frozenset[str] = frozenset()):
        def build(run) -> list[str]:
            exit_code = "7" if run.name in fail else "0"
            return [sys.executable, str(self.script), run.name, str(self.records), str(sleep), exit_code]

        return build

    def read_records(self) -> dict[str, str]:
        return {
            entry["name"]: entry["visible"]
            for entry in map(json.loads, self.records.read_text(encoding="utf-8").splitlines())
        }

    def run_queue_training(self, planned, command_builder, **kwargs):
        module = load_script("train_queue")
        with contextlib.redirect_stdout(io.StringIO()):
            return module.queue_training(
                planned,
                output_root=self.root,
                logs_dir=self.root / "logs",
                command_builder=command_builder,
                **kwargs,
            )

    def test_gpu_environment_assigns_one_device_per_worker(self) -> None:
        self.assertEqual(
            gpu_environment(2),
            {0: {"CUDA_VISIBLE_DEVICES": "0"}, 1: {"CUDA_VISIBLE_DEVICES": "1"}},
        )
        self.assertEqual(gpu_environment(1), {0: {"CUDA_VISIBLE_DEVICES": "0"}})

    def test_each_worker_sees_only_its_own_gpu_and_completed_runs_are_skipped(self) -> None:
        planned = self.plan()
        finished = planned[0]
        checkpoint_dir = finished.run.directory(self.root) / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "final.pt").write_bytes(b"")
        (finished.run.directory(self.root) / "run.json").write_text(
            json.dumps({"config": finished.run.resolve().to_dict()}) + "\n", encoding="utf-8"
        )
        planned = self.plan()

        summary = self.run_queue_training(
            planned, self.fake_builder(sleep=0.05), workers=2
        )

        self.assertEqual(len(planned), 15)
        self.assertEqual(summary.skipped, (finished.run.name,))
        self.assertEqual(len(summary.completed), 14)
        self.assertEqual(summary.failed, ())
        self.assertEqual(summary.exit_code, 0)
        self.assertEqual({result.worker for result in summary.completed}, {0, 1})
        records = self.read_records()
        self.assertEqual(len(records), 14)
        for result in summary.completed:
            with self.subTest(run=result.job.name):
                self.assertEqual(records[result.job.name], str(result.worker))
                self.assertTrue(result.log_path.is_file())
        self.assertFalse((self.root / "logs" / f"{finished.run.name}.log").exists())

    def test_failed_run_does_not_stop_the_rest_and_fails_the_queue(self) -> None:
        planned = self.plan()[:3]
        failing = planned[1].run.name
        summary = self.run_queue_training(
            planned, self.fake_builder(fail=frozenset({failing})), workers=2
        )
        self.assertEqual([result.job.name for result in summary.failed], [failing])
        self.assertEqual(len(summary.completed), 2)
        self.assertEqual(summary.exit_code, 1)

    def test_default_builder_uses_the_shared_train_command(self) -> None:
        module = load_script("train_queue")
        self.assertIs(module.train_command, train_command)
        calls = []

        def spy(run, *, python, output_root, data_root, overrides):
            calls.append((run, python, Path(output_root), data_root, tuple(overrides)))
            return [sys.executable, str(self.script), run.name, str(self.records), "0.0", "0"]

        planned = self.plan()[:2]
        with mock.patch.object(module, "train_command", spy):
            with contextlib.redirect_stdout(io.StringIO()):
                summary = module.queue_training(
                    planned,
                    output_root=self.root,
                    data_root=self.root / "data",
                    overrides=("train.total_iters=200",),
                    workers=1,
                )
        self.assertEqual([run for run, *_ in calls], [item.run for item in planned])
        self.assertTrue(all(root_arg == self.root for _, _, root_arg, _, _ in calls))
        self.assertTrue(all(data_arg == self.root / "data" for *_, data_arg, _ in calls))
        self.assertTrue(
            all(overrides == ("train.total_iters=200",) for *_, overrides in calls)
        )
        self.assertEqual(len(summary.completed), 2)
        self.assertEqual(summary.exit_code, 0)

    def test_train_command_derives_every_argument_from_the_run(self) -> None:
        samples = [
            runs(DATA_SIZE, "pickcube")[0],
            runs(EXPERIMENT, TASK)[2],
            runs(DATA_SIZE, "plugcharger")[-1],
        ]
        for run in samples:
            with self.subTest(task=run.task, value=run.value, seed=run.seed):
                command = train_command(
                    run, output_root=self.root / "runs", data_root=self.root / "data"
                )
                self.assertEqual(command[command.index("--config") + 1], str(run.config))
                self.assertEqual(command[command.index("--experiment") + 1], str(run.experiment))
                self.assertEqual(command[command.index("--experiment-value") + 1], str(run.value))
                self.assertEqual(command[command.index("--seed") + 1], str(run.seed))
                self.assertEqual(command[command.index("--exp") + 1], run.name)
                self.assertEqual(
                    command[command.index("--output-root") + 1], str(self.root / "runs")
                )
                self.assertEqual(
                    command[command.index("--data-root") + 1], str(self.root / "data")
                )
                self.assertEqual(command[command.index("--resume") + 1], "auto")
                # train_dp.py keeps its own cuda default; the GPU comes from the
                # worker environment, never from a per-run flag.
                self.assertNotIn("--device", command)

    def test_train_command_appends_set_overrides(self) -> None:
        run = runs(EXPERIMENT, TASK)[0]
        command = train_command(
            run,
            output_root=self.root / "runs",
            overrides=("train.total_iters=200", "train.checkpoint_steps=[200]"),
        )
        self.assertEqual(
            command[-4:],
            ["--set", "train.total_iters=200", "--set", "train.checkpoint_steps=[200]"],
        )

    def test_train_command_omits_data_root_when_not_requested(self) -> None:
        command = train_command(runs(EXPERIMENT, TASK)[0], output_root=self.root / "runs")
        self.assertNotIn("--data-root", command)

    def test_second_start_skips_completed_and_resumes_interrupted(self) -> None:
        module = load_script("train_queue")
        planned = self.plan()
        finished, interrupted = planned[0], planned[1]
        checkpoint_dir = finished.run.directory(self.root) / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "final.pt").write_bytes(b"")
        (finished.run.directory(self.root) / "run.json").write_text(
            json.dumps({"config": finished.run.resolve().to_dict()}) + "\n", encoding="utf-8"
        )
        resume_dir = interrupted.run.directory(self.root) / "checkpoints"
        resume_dir.mkdir(parents=True)
        (resume_dir / "resume.pt").write_bytes(b"")

        # Second launch: the plan sees one finished run and one checkpointed run.
        planned = self.plan()
        self.assertIs(planned[0].completion.state, RunState.COMPLETED)
        self.assertIs(planned[1].completion.state, RunState.PENDING)
        self.assertTrue(planned[1].completion.resumable)
        self.assertIs(planned[2].completion.state, RunState.PENDING)
        self.assertFalse(planned[2].completion.resumable)

        commands = []

        def spy(run, *, python, output_root, data_root, overrides):
            commands.append(
                train_command(
                    run, python=python, output_root=output_root, data_root=data_root
                )
            )
            return [sys.executable, str(self.script), run.name, str(self.records), "0.0", "0"]

        with mock.patch.object(module, "train_command", spy):
            with contextlib.redirect_stdout(io.StringIO()):
                summary = module.queue_training(planned[:3], output_root=self.root, workers=2)

        self.assertEqual(summary.skipped, (finished.run.name,))
        self.assertEqual(
            [result.job.name for result in summary.completed],
            [interrupted.run.name, planned[2].run.name],
        )
        self.assertEqual(
            [command[command.index("--exp") + 1] for command in commands],
            [interrupted.run.name, planned[2].run.name],
        )
        # The interrupted run resumes through the trainer's --resume auto.
        self.assertEqual(commands[0][commands[0].index("--resume") + 1], "auto")
        self.assertFalse((self.root / "logs" / f"{finished.run.name}.log").exists())

    def test_cli_defaults_to_two_workers_and_rejects_more(self) -> None:
        module = load_script("train_queue")
        args = module.parse_args([])
        self.assertEqual(args.workers, 2)
        self.assertEqual(args.task, None)
        self.assertEqual(args.overrides, [])
        args = module.parse_args(["--set", "train.total_iters=200", "--set", "train.amp=false"])
        self.assertEqual(args.overrides, ["train.total_iters=200", "train.amp=false"])
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                module.parse_args(["--workers", "3"])

    def test_cli_reads_run_root_and_data_root_from_the_environment(self) -> None:
        module = load_script("train_queue")
        environment = {"DATA_ROOT": "/tmp/intended-data", "RUN_ROOT": "/tmp/intended-runs"}
        with mock.patch.dict(os.environ, environment):
            args = module.parse_args(["--task", TASK])
        self.assertEqual(args.data_root, Path("/tmp/intended-data"))
        self.assertEqual(args.output_root, Path("/tmp/intended-runs"))

    def test_cli_falls_back_when_the_environment_is_unset(self) -> None:
        module = load_script("train_queue")
        with mock.patch.dict(os.environ, {}, clear=True):
            args = module.parse_args([])
        self.assertIsNone(args.data_root)
        self.assertEqual(args.output_root, ROOT / "runs")

    def test_cli_flags_override_the_environment(self) -> None:
        module = load_script("train_queue")
        environment = {"DATA_ROOT": "/tmp/intended-data", "RUN_ROOT": "/tmp/intended-runs"}
        with mock.patch.dict(os.environ, environment):
            args = module.parse_args(["--data-root", "/cli/data", "--output-root", "/cli/runs"])
        self.assertEqual(args.data_root, Path("/cli/data"))
        self.assertEqual(args.output_root, Path("/cli/runs"))

    def test_main_forwards_the_resolved_roots_to_plan_and_queue(self) -> None:
        module = load_script("train_queue")
        captured = {}

        def fake_plan(experiment, task, *, output_root, overrides, data_root):
            captured["plan"] = (experiment, task, output_root, tuple(overrides), data_root)
            return []

        def fake_queue(planned, *, output_root, data_root, overrides, workers):
            captured["queue"] = (output_root, data_root, workers, tuple(overrides))
            return SimpleNamespace(exit_code=0)

        environment = {"DATA_ROOT": "/tmp/intended-data", "RUN_ROOT": "/tmp/intended-runs"}
        with mock.patch.dict(os.environ, environment):
            with mock.patch.object(module, "plan_runs", fake_plan):
                with mock.patch.object(module, "queue_training", fake_queue):
                    exit_code = module.main(
                        ["--task", TASK, "--set", "train.total_iters=200"]
                    )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            captured["plan"],
            (
                module.DEFAULT_EXPERIMENT,
                TASK,
                Path("/tmp/intended-runs"),
                ("train.total_iters=200",),
                Path("/tmp/intended-data"),
            ),
        )
        self.assertEqual(
            captured["queue"],
            (Path("/tmp/intended-runs"), Path("/tmp/intended-data"), 2, ("train.total_iters=200",)),
        )


if __name__ == "__main__":
    unittest.main()
