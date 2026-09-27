"""Task 3 regression tests: dynamic two-worker queue on fake commands.

The queue must keep at most two subprocesses alive, hand the next pending run
to whichever worker frees up first, log every run separately, keep going after
a failure and report that failure in the final summary. Every job here is a
tiny Python script, so these tests need no GPU, no torch and no real training.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from dp_manip.scheduler import Job, QueueSummary, gpu_environment, job_log_path, run_queue


ROOT = Path(__file__).resolve().parents[1]


FAKE_JOB = '''
import argparse
import os
import pathlib
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True)
parser.add_argument("--events", required=True)
parser.add_argument("--sleep", type=float, default=0.0)
parser.add_argument("--wait-for", default="")
parser.add_argument("--exit", type=int, default=0)
arguments = parser.parse_args()


def record(kind):
    with open(arguments.events, "a", encoding="utf-8") as stream:
        stream.write(f"{kind} {arguments.name} {time.time()}\\n")


record("start")
print(f"job {arguments.name} running", flush=True)
print(
    f"marker={os.environ.get('DP_FAKE_MARKER', '')} cwd={os.getcwd()} "
    f"gpus={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
    flush=True,
)
if arguments.wait_for:
    deadline = time.time() + 30
    while not pathlib.Path(arguments.wait_for).exists():
        if time.time() > deadline:
            print(f"job {arguments.name} timed out", file=sys.stderr, flush=True)
            sys.exit(3)
        time.sleep(0.01)
if arguments.sleep:
    time.sleep(arguments.sleep)
if arguments.exit:
    print(f"failure marker for {arguments.name}", file=sys.stderr, flush=True)
record("end")
sys.exit(arguments.exit)
'''


FAKE_SIGNAL_TRAINER = '''
import signal
import sys
import time
from pathlib import Path

name, started, resume, mode = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
started.parent.mkdir(parents=True, exist_ok=True)
started.write_text("started\\n", encoding="utf-8")


def checkpoint(signum, _frame):
    # Emulate the trainer: finish writing resume.pt, then exit 75. A scheduler
    # that gave up before the write would leave no resume.pt behind.
    time.sleep(0.3)
    resume.parent.mkdir(parents=True, exist_ok=True)
    resume.write_text(f"checkpoint after signal {signum}\\n", encoding="utf-8")
    sys.exit(75)


if mode == "checkpoint":
    signal.signal(signal.SIGTERM, checkpoint)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, checkpoint)

while True:
    time.sleep(0.02)
'''

QUEUE_DRIVER = '''
import json
import sys

from dp_manip.scheduler import Job, run_queue

spec = json.loads(sys.argv[1])
jobs = [Job(name=item["name"], command=item["command"]) for item in spec["jobs"]]
summary = run_queue(jobs, workers=2, logs_dir=spec["logs"])
report = {
    "exit_code": summary.exit_code,
    "results": [
        {
            "name": result.job.name,
            "status": result.status.value,
            "worker": result.worker,
            "exit_code": result.exit_code,
        }
        for result in summary.results
    ],
}
with open(spec["report"], "w", encoding="utf-8") as stream:
    json.dump(report, stream)
sys.exit(summary.exit_code)
'''


def max_concurrency(results) -> int:
    """Upper bound on simultaneous jobs from the scheduler's own timestamps."""
    events = [(result.started, 1) for result in results if result.started is not None]
    events += [(result.finished, -1) for result in results if result.finished is not None]
    # At equal timestamps, count starts before ends so the bound never under-reports.
    events.sort(key=lambda item: (item[0], -item[1]))
    current = maximum = 0
    for _, delta in events:
        current += delta
        maximum = max(maximum, current)
    return maximum


class QueueTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.script = self.root / "fake_job.py"
        self.script.write_text(FAKE_JOB, encoding="utf-8")
        self.events = self.root / "events.txt"
        self.logs = self.root / "logs"

    def job(
        self,
        name: str,
        *,
        sleep: float = 0.0,
        wait_for: Path | None = None,
        exit_code: int = 0,
    ) -> Job:
        command = [sys.executable, str(self.script), "--name", name, "--events", str(self.events)]
        if sleep:
            command.extend(("--sleep", str(sleep)))
        if wait_for is not None:
            command.extend(("--wait-for", str(wait_for)))
        if exit_code:
            command.extend(("--exit", str(exit_code)))
        return Job(name=name, command=command)

    def event_names(self, kind: str) -> set[str]:
        if not self.events.is_file():
            return set()
        return {
            line.split()[1]
            for line in self.events.read_text(encoding="utf-8").splitlines()
            if line.startswith(f"{kind} ")
        }

    def wait_for_event(self, kind: str, name: str, timeout: float = 30.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if name in self.event_names(kind):
                return
            time.sleep(0.01)
        self.fail(f"timed out waiting for {kind} {name}")

    def run_jobs(self, jobs, **kwargs) -> tuple[QueueSummary, str]:
        """Run the queue with scheduler events captured for assertions."""
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            summary = run_queue(jobs, logs_dir=self.logs, **kwargs)
        return summary, output.getvalue()


class QueueExecutionTest(QueueTestCase):
    def test_jobs_run_concurrently_with_independent_logs(self) -> None:
        jobs = [
            self.job("alpha", sleep=0.3),
            self.job("beta", sleep=0.3),
            self.job("gamma", sleep=0.3),
        ]
        summary, output = self.run_jobs(jobs, workers=2)

        self.assertEqual([result.job.name for result in summary.completed], [job.name for job in jobs])
        self.assertEqual(summary.failed, ())
        self.assertEqual(summary.exit_code, 0)
        self.assertLessEqual(max_concurrency(summary.results), 2)
        for job in jobs:
            log = self.logs / f"{job.name}.log"
            self.assertTrue(log.is_file(), log)
            text = log.read_text(encoding="utf-8")
            self.assertIn(f"job {job.name} running", text)
            for other in set(job.name for job in jobs) - {job.name}:
                self.assertNotIn(other, text)
        self.assertIn("completed: 3 skipped: 0 failed: 0", output)

    def test_path_like_names_get_independent_logs(self) -> None:
        # Regression: replacing "/" with "_" mapped a/b and a_b to the same
        # a_b.log and silently overwrote one run's log.
        jobs = [self.job("a/b"), self.job("a_b")]
        summary, _ = self.run_jobs(jobs, workers=2)
        self.assertEqual(len(summary.completed), 2)
        slash_log = job_log_path(self.logs, "a/b")
        underscore_log = job_log_path(self.logs, "a_b")
        self.assertNotEqual(slash_log, underscore_log)
        self.assertIn("job a/b running", slash_log.read_text(encoding="utf-8"))
        self.assertIn("job a_b running", underscore_log.read_text(encoding="utf-8"))

    def test_log_path_mapping_is_injective(self) -> None:
        names = ["a/b", "a_b", "a%2Fb", "a b", "plain_run-name_1"]
        paths = [job_log_path(self.logs, name) for name in names]
        self.assertEqual(len(set(paths)), len(names))

    def test_free_worker_immediately_claims_the_next_job(self) -> None:
        # The blocker holds one worker until the test releases it; the other
        # worker must finish "quick" and immediately pick up "second" instead
        # of waiting for the blocker to finish. This also pins concurrency at 2.
        release = self.root / "release.txt"
        jobs = [self.job("blocker", wait_for=release), self.job("quick"), self.job("second")]
        collected: list[QueueSummary] = []
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            thread = threading.Thread(
                target=lambda: collected.append(run_queue(jobs, workers=2, logs_dir=self.logs))
            )
            thread.start()
            try:
                self.wait_for_event("end", "quick")
                self.assertNotIn("blocker", self.event_names("end"))
                self.wait_for_event("end", "second")
            finally:
                release.write_text("go", encoding="utf-8")
                thread.join(timeout=30)
        self.assertFalse(thread.is_alive(), "scheduler did not stop after the blocker was released")

        summary = collected[0]
        by_name = {result.job.name: result for result in summary.results}
        self.assertEqual(len(summary.completed), 3)
        self.assertNotEqual(by_name["blocker"].worker, by_name["quick"].worker)
        self.assertEqual(by_name["second"].worker, by_name["quick"].worker)
        self.assertEqual(max_concurrency(summary.results), 2)

    def test_single_job_queue(self) -> None:
        summary, _ = self.run_jobs([self.job("only")], workers=2)
        self.assertEqual(len(summary.completed), 1)
        self.assertIn(summary.completed[0].worker, (0, 1))
        self.assertEqual(summary.exit_code, 0)
        self.assertTrue(self.logs.joinpath("only.log").is_file())

    def test_job_env_and_cwd_reach_the_child(self) -> None:
        job = Job(
            name="env-check",
            command=[
                sys.executable,
                str(self.script),
                "--name",
                "env-check",
                "--events",
                str(self.events),
            ],
            cwd=self.root,
            env={"DP_FAKE_MARKER": "present"},
        )
        summary, _ = self.run_jobs([job], workers=2)
        self.assertEqual(len(summary.completed), 1)
        text = self.logs.joinpath("env-check.log").read_text(encoding="utf-8")
        self.assertIn("marker=present", text)
        self.assertIn(f"cwd={self.root.resolve()}", text)

    def test_worker_environment_is_applied_per_worker(self) -> None:
        jobs = [self.job(f"env-{index}") for index in range(4)]
        summary, _ = self.run_jobs(jobs, workers=2, worker_env=gpu_environment(2))
        self.assertEqual(len(summary.completed), 4)
        for result in summary.completed:
            with self.subTest(job=result.job.name, worker=result.worker):
                text = result.log_path.read_text(encoding="utf-8")
                self.assertIn(f"gpus={result.worker}", text)

    def test_job_environment_overrides_the_worker_environment(self) -> None:
        template = self.job("override")
        job = Job(
            name=template.name,
            command=template.command,
            env={"CUDA_VISIBLE_DEVICES": "7"},
        )
        summary, _ = self.run_jobs([job], workers=2, worker_env=gpu_environment(2))
        self.assertIn("gpus=7", summary.completed[0].log_path.read_text(encoding="utf-8"))

    def test_queue_inherits_the_caller_environment_without_worker_env(self) -> None:
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "5"}):
            summary, _ = self.run_jobs([self.job("inherit")], workers=2)
        self.assertIn("gpus=5", summary.completed[0].log_path.read_text(encoding="utf-8"))

    def test_worker_env_keys_must_be_valid_worker_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "worker_env"):
            run_queue(
                [self.job("a")],
                workers=2,
                logs_dir=self.logs,
                worker_env={2: {"CUDA_VISIBLE_DEVICES": "2"}},
            )

    def test_empty_queue_reports_skipped_runs_only(self) -> None:
        summary, output = self.run_jobs([], workers=2, skipped=["done-a", "done-b"])
        self.assertEqual(summary.results, ())
        self.assertEqual(summary.skipped, ("done-a", "done-b"))
        self.assertEqual(summary.exit_code, 0)
        self.assertIn("completed: 0 skipped: 2 failed: 0", output)
        # An empty queue has no side effects: no logs directory is created.
        self.assertFalse(self.logs.exists())


class QueueFailureTest(QueueTestCase):
    def test_failure_does_not_stop_other_jobs_and_fails_the_queue(self) -> None:
        jobs = [
            self.job("ok-first", sleep=0.05),
            self.job("bad", exit_code=7),
            self.job("ok-last", sleep=0.05),
        ]
        summary, output = self.run_jobs(jobs, workers=2)

        self.assertEqual([result.job.name for result in summary.completed], ["ok-first", "ok-last"])
        self.assertEqual(len(summary.failed), 1)
        failed = summary.failed[0]
        self.assertEqual(failed.job.name, "bad")
        self.assertEqual(failed.exit_code, 7)
        self.assertEqual(summary.exit_code, 1)
        text = self.logs.joinpath("bad.log").read_text(encoding="utf-8")
        self.assertIn("job bad running", text)
        self.assertIn("failure marker for bad", text)  # stderr shares the run log
        for name in ("ok-first", "ok-last"):
            self.assertTrue(self.logs.joinpath(f"{name}.log").is_file())
        self.assertIn("failed bad exit=7", output)
        self.assertIn("completed: 2 skipped: 0 failed: 1", output)

    def test_command_that_cannot_start_is_a_recorded_failure(self) -> None:
        summary, _ = self.run_jobs(
            [Job(name="missing", command=[str(self.root / "no-such-program")])],
            workers=2,
        )
        self.assertEqual(len(summary.failed), 1)
        self.assertEqual(summary.failed[0].exit_code, 127)
        self.assertEqual(summary.exit_code, 1)
        self.assertIn(
            "failed to start", self.logs.joinpath("missing.log").read_text(encoding="utf-8")
        )

    def test_worker_exception_is_recorded_instead_of_losing_the_job(self) -> None:
        summary, _ = self.run_jobs(
            [Job(name="broken", command=["program\x00"]), self.job("fine")],
            workers=2,
        )
        self.assertEqual(len(summary.results), 2)
        self.assertEqual(len(summary.completed), 1)
        self.assertEqual(len(summary.failed), 1)
        self.assertIn("embedded null byte", summary.failed[0].error)
        self.assertEqual(summary.exit_code, 1)

    def test_exit_75_is_interrupted_while_other_nonzero_codes_fail(self) -> None:
        jobs = [self.job("requeue-me", exit_code=75), self.job("broken", exit_code=3)]
        summary, output = self.run_jobs(jobs, workers=2)
        self.assertEqual([result.job.name for result in summary.interrupted], ["requeue-me"])
        self.assertEqual([result.job.name for result in summary.failed], ["broken"])
        self.assertEqual(summary.interrupted[0].exit_code, 75)
        self.assertEqual(summary.failed[0].exit_code, 3)
        # An interrupted checkpoint wants a requeue even when another run failed.
        self.assertEqual(summary.exit_code, 75)
        self.assertIn("interrupted requeue-me exit=75", output)
        self.assertIn("failed broken exit=3", output)
        self.assertIn("interrupted: 1", output)

    def test_pre_failed_runs_are_reported_without_spawning(self) -> None:
        summary, output = self.run_jobs(
            [self.job("ready")], workers=2, failed=["missing-a", "missing-b"]
        )
        self.assertEqual(summary.pre_failed, ("missing-a", "missing-b"))
        self.assertEqual(len(summary.completed), 1)
        self.assertEqual(summary.failed_count, 2)
        self.assertEqual(summary.exit_code, 1)
        self.assertIn("failed: 2", output)
        self.assertFalse((self.logs / "missing-a.log").exists())

    def test_pre_failed_runs_cannot_also_be_pending_or_skipped(self) -> None:
        with self.assertRaisesRegex(ValueError, "pre-failed"):
            run_queue([self.job("a")], workers=2, logs_dir=self.logs, failed=["a"])
        with self.assertRaisesRegex(ValueError, "pre-failed"):
            run_queue([], workers=2, logs_dir=self.logs, skipped=["a"], failed=["a"])


class QueueValidationTest(QueueTestCase):
    def test_invalid_worker_counts_and_names_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "workers must be between 1 and 2"):
            run_queue([self.job("a")], workers=3, logs_dir=self.logs)
        with self.assertRaisesRegex(ValueError, "workers must be between 1 and 2"):
            run_queue([self.job("a")], workers=0, logs_dir=self.logs)
        with self.assertRaisesRegex(ValueError, "unique"):
            run_queue([self.job("a"), self.job("a")], workers=2, logs_dir=self.logs)
        with self.assertRaisesRegex(ValueError, "both pending and skipped"):
            run_queue([self.job("a")], workers=2, logs_dir=self.logs, skipped=["a"])
        with self.assertRaisesRegex(ValueError, "empty command"):
            Job(name="empty", command=())
        with self.assertRaisesRegex(ValueError, "non-empty"):
            Job(name="", command=("true",))


class QueueSignalTest(unittest.TestCase):
    """SIGTERM/SIGUSR1 must drain the running children, checkpoint first."""

    def wait_for(self, predicate, timeout: float = 30.0, message: str = "condition") -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.02)
        self.fail(f"timed out waiting for {message}")

    def run_case(self, signum, mode: str, names: tuple[str, ...]):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "fake_signal_trainer.py"
            script.write_text(FAKE_SIGNAL_TRAINER, encoding="utf-8")
            jobs = [
                {
                    "name": name,
                    "command": [
                        sys.executable,
                        str(script),
                        name,
                        str(root / "started" / f"{name}.flag"),
                        str(root / "resume" / f"{name}.pt"),
                        mode,
                    ],
                }
                for name in names
            ]
            spec = {"jobs": jobs, "logs": str(root / "logs"), "report": str(root / "report.json")}
            process = subprocess.Popen(
                [sys.executable, "-c", QUEUE_DRIVER, json.dumps(spec)],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            output = ""
            try:
                self.wait_for(
                    lambda: len(list((root / "started").glob("*.flag"))) >= 2,
                    message="two runs to start",
                )
                started = sorted(path.stem for path in (root / "started").glob("*.flag"))
                self.assertEqual(len(started), 2, f"unexpected started runs: {started}")
                process.send_signal(signum)
                output, _ = process.communicate(timeout=60)
            finally:
                if process.poll() is None:
                    process.kill()
                    output, _ = process.communicate()
            report = json.loads((root / "report.json").read_text(encoding="utf-8"))
            resumes = {
                name: (root / "resume" / f"{name}.pt").read_text(encoding="utf-8")
                for name in names
                if (root / "resume" / f"{name}.pt").is_file()
            }
            return process.returncode, report, output, started, resumes

    def test_stop_signals_drain_running_runs_and_requeue_the_rest(self) -> None:
        for signum, prefix in (
            (signal.SIGTERM, "term"),
            (getattr(signal, "SIGUSR1", None), "usr1"),
        ):
            if signum is None:
                continue
            names = (f"{prefix}-one", f"{prefix}-two", f"{prefix}-three")
            with self.subTest(signal=signum):
                returncode, report, output, started, resumes = self.run_case(
                    signum, "checkpoint", names
                )
                self.assertEqual(returncode, 75, output)
                self.assertEqual(report["exit_code"], 75)
                self.assertEqual(len(started), 2)
                self.assertIn("received signal", output)
                self.assertIn("no new runs will start", output)
                entries = {entry["name"]: entry for entry in report["results"]}
                for name in names:
                    self.assertEqual(entries[name]["status"], "interrupted")
                self.assertEqual(set(resumes), set(started))
                for name in started:
                    self.assertEqual(entries[name]["exit_code"], 75)
                    self.assertIn("checkpoint after signal", resumes[name])
                unstarted = set(names) - set(started)
                self.assertEqual(len(unstarted), 1)
                name = unstarted.pop()
                self.assertIsNone(entries[name]["exit_code"])

    def test_signal_without_checkpoint_is_a_failure_not_a_requeue(self) -> None:
        # A child that dies on the forwarded signal never wrote resume.pt, so
        # it must stay failed; only exit 75 means "checkpointed, requeue".
        names = ("dead-one", "dead-two")
        returncode, report, output, started, resumes = self.run_case(
            getattr(signal, "SIGUSR1", signal.SIGTERM), "die", names
        )
        self.assertEqual(returncode, 1, output)
        self.assertEqual(report["exit_code"], 1)
        self.assertEqual(len(started), 2)
        self.assertEqual(resumes, {})
        entries = {entry["name"]: entry for entry in report["results"]}
        for name in names:
            self.assertEqual(entries[name]["status"], "failed")
            self.assertLess(entries[name]["exit_code"], 0)
        self.assertIn("interrupted: 0", output)


if __name__ == "__main__":
    unittest.main()
