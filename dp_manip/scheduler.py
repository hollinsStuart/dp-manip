"""Dynamic two-worker queue for independent runs.

The cluster QOS allows one Slurm job with at most two GPUs, so the runs of a
sweep share one pending queue: whichever worker finishes first claims the next
run instead of waiting for a fixed partner (plan §5.5 and §6.5). This module
owns that generic behavior -- bounded worker threads, one subprocess per run,
an independent ``stdout``/``stderr`` log per run, lifecycle records and a
completed/skipped/failed summary -- and deliberately imports nothing heavier
than the standard library. Building the real training commands, binding GPUs
and handling Slurm signals belong to the entry points that use this queue.

Interruption (plan §6.7): SIGTERM and SIGUSR1 stop the queue from claiming new
runs, are forwarded to every running child, and the queue waits for all of them
before returning. The trainer's existing contract is that a scheduler signal
writes ``resume.pt`` and exits 75; such runs are recorded as ``interrupted``
and make the summary's exit code 75, the Slurm requeue convention. Runs that
were never claimed are ``interrupted`` too, so a second launch skips finished
runs and resumes the rest through ``--resume auto``. Any other non-zero exit
remains an ordinary ``failed`` run.
"""

from __future__ import annotations

import enum
import os
import queue
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote


# The accounting/QOS policy that motivated the single-job refactor also caps
# the usable GPUs per job at two, so the queue refuses to oversubscribe.
MAX_WORKERS = 2

# ``scripts/train_dp.py`` returns 75 after a scheduler signal wrote resume.pt;
# ``slurm/train_array.sbatch`` requeues the job on exactly that code.
REQUEUE_EXIT_CODE = 75

# Scheduler events are the top-level Slurm log, so keep each line atomic even
# when two workers finish at the same moment.
_EVENT_LOCK = threading.Lock()

# SIGUSR1 is the Slurm preemption notification; SIGTERM arrives on walltime or
# ``scancel``. SIGUSR1 is absent on Windows, so install what the platform has.
_HANDLED_SIGNALS = tuple(
    number
    for number in (getattr(signal, "SIGUSR1", None), signal.SIGTERM)
    if number is not None
)


def _event(message: str) -> None:
    with _EVENT_LOCK:
        print(message, flush=True)


class JobStatus(enum.Enum):
    """How one queued job ended."""

    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    # The trainer checkpointed after a scheduler signal (exit 75), or the job
    # was never claimed because the queue stopped. Both stay unfinished and are
    # resumed by the next launch of the same command.
    INTERRUPTED = "interrupted"


@dataclass(frozen=True)
class Job:
    """One independent command in the shared pending queue."""

    name: str  # run id, used for events and the log file name
    command: Sequence[str]
    cwd: str | Path | None = None
    env: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("job name must be non-empty")
        command = tuple(self.command)
        if not command:
            raise ValueError(f"job {self.name!r} has an empty command")
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "env", dict(self.env))


@dataclass(frozen=True)
class JobResult:
    """Lifecycle record for one job: who ran it, when, and how it ended."""

    index: int
    job: Job
    status: JobStatus
    worker: int | None
    exit_code: int | None
    started: float | None
    finished: float | None
    log_path: Path
    error: str | None = None

    @property
    def duration(self) -> float | None:
        if self.started is None or self.finished is None:
            return None
        return self.finished - self.started


@dataclass(frozen=True)
class QueueSummary:
    """Outcome of one queue run, in the order the jobs were declared."""

    results: tuple[JobResult, ...] = ()
    skipped: tuple[str, ...] = ()
    # Runs known to have failed before the queue started (for example, an
    # evaluation whose checkpoint does not exist); no subprocess was spawned.
    pre_failed: tuple[str, ...] = ()

    def with_status(self, status: JobStatus) -> tuple[JobResult, ...]:
        return tuple(result for result in self.results if result.status is status)

    @property
    def completed(self) -> tuple[JobResult, ...]:
        return self.with_status(JobStatus.COMPLETED)

    @property
    def failed(self) -> tuple[JobResult, ...]:
        return self.with_status(JobStatus.FAILED)

    @property
    def interrupted(self) -> tuple[JobResult, ...]:
        return self.with_status(JobStatus.INTERRUPTED)

    @property
    def failed_count(self) -> int:
        """Every failed run, from the queue and from the preflight."""
        return len(self.failed) + len(self.pre_failed)

    @property
    def exit_code(self) -> int:
        """75 requests a requeue, 1 reports failures, 0 means everything ran.

        Interrupted runs take precedence: they are unfinished work that a
        requeue can continue, while a failure without an interruption is final.
        """
        if self.interrupted:
            return REQUEUE_EXIT_CODE
        return 1 if self.failed_count else 0


def job_log_path(logs_dir: str | Path, name: str) -> Path:
    """Log file for one job, with an injective mapping from job name to file.

    The name is percent-encoded, so path-like names cannot alias each other:
    ``a/b`` becomes ``a%2Fb.log`` while ``a_b`` stays ``a_b.log``. Plain run
    names (letters, digits, ``_`` and ``-``) are unchanged.
    """
    return Path(logs_dir) / f"{quote(name, safe='')}.log"


def gpu_environment(workers: int = MAX_WORKERS) -> dict[int, dict[str, str]]:
    """One visible GPU per worker: worker N may only see physical GPU N.

    Each trainer keeps requesting ``--device cuda``; with a single visible
    device that is the worker's own card, so two concurrent workers can never
    see the same GPU (plan §5.4).
    """
    return {worker: {"CUDA_VISIBLE_DEVICES": str(worker)} for worker in range(workers)}


def run_queue(
    jobs: Sequence[Job],
    *,
    workers: int = MAX_WORKERS,
    logs_dir: str | Path,
    skipped: Sequence[str] = (),
    failed: Sequence[str] = (),
    worker_env: Mapping[int, Mapping[str, str]] | None = None,
) -> QueueSummary:
    """Run every job with at most ``workers`` concurrent subprocesses.

    The queue is shared and dynamic: a worker claims the next pending job the
    moment it becomes free, so runs are never statically paired. Each job's
    stdout and stderr are redirected to ``<logs_dir>/<job name>.log``; start,
    completion, failure and the final summary are printed as scheduler events.
    A failing job never stops the others, but it makes the summary's
    ``exit_code`` non-zero. ``skipped`` names runs that were already complete
    (for example, the finished cells of the plan) and only enter the summary;
    ``failed`` names runs that are known to be unrunnable (for example, an
    evaluation whose checkpoint is missing) and enter the summary as failures
    without spawning a process.

    ``worker_env`` adds environment variables that depend on which worker runs
    the job (for example ``gpu_environment()``); a job's own ``env`` still wins
    over the worker value.

    SIGTERM/SIGUSR1 stop the queue from claiming new jobs, are forwarded to all
    running children, and the call waits for them to finish (which includes the
    trainer writing ``resume.pt``). Exit-75 children and never-claimed jobs are
    ``interrupted``.
    """
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError(
            f"workers must be between 1 and {MAX_WORKERS} for the single Slurm job; got {workers}"
        )
    if worker_env is not None:
        invalid = sorted(
            str(worker)
            for worker in worker_env
            if isinstance(worker, bool) or not isinstance(worker, int) or not 0 <= worker < workers
        )
        if invalid:
            raise ValueError(
                f"worker_env keys must be worker ids in [0, {workers}); got {invalid}"
            )
    names = [job.name for job in jobs]
    log_files: dict[str, list[str]] = {}
    for job in jobs:
        log_files.setdefault(job_log_path(logs_dir, job.name).name, []).append(job.name)
    collisions = {log: colliding for log, colliding in log_files.items() if len(colliding) > 1}
    if collisions:
        raise ValueError(
            f"job names must map to unique log files, but these collide: {collisions}"
        )
    overlap = sorted(set(names) & set(skipped))
    if overlap:
        raise ValueError(f"runs cannot be both pending and skipped: {overlap}")
    already_failed = sorted((set(names) | set(skipped)) & set(failed))
    if already_failed:
        raise ValueError(
            f"pre-failed runs must not also be pending or skipped: {already_failed}"
        )

    logs_dir = Path(logs_dir)
    pending: queue.Queue[tuple[int, Job]] = queue.Queue()
    for index, job in enumerate(jobs):
        pending.put((index, job))
    results: list[JobResult] = []
    results_lock = threading.Lock()
    stop = threading.Event()
    stop_signal: list[int] = []
    running: dict[int, subprocess.Popen] = {}
    running_lock = threading.Lock()

    def record(result: JobResult) -> None:
        with results_lock:
            results.append(result)

    def interrupted(index: int, job: Job) -> JobResult:
        return JobResult(
            index=index,
            job=job,
            status=JobStatus.INTERRUPTED,
            worker=None,
            exit_code=None,
            started=None,
            finished=None,
            log_path=job_log_path(logs_dir, job.name),
        )

    def register(worker: int, process: subprocess.Popen) -> None:
        # Claiming the process and remembering the stop request share one lock,
        # so a signal cannot slip between spawning and forwarding.
        with running_lock:
            running[worker] = process
            pending_signal = stop_signal[0] if stop_signal else None
        if pending_signal is not None:
            _send_signal(process, pending_signal)

    def unregister(worker: int) -> None:
        with running_lock:
            running.pop(worker, None)

    def request_stop(signum: int) -> None:
        with running_lock:
            if not stop_signal:
                stop_signal.append(signum)
            children = list(running.values())
        if not stop.is_set():
            stop.set()
            _event(f"received signal {signum}; no new runs will start")
        for process in children:
            _send_signal(process, signum)

    def handle_signal(signum, _frame) -> None:
        request_stop(signum)

    def consume(worker: int) -> None:
        while not stop.is_set():
            try:
                index, job = pending.get_nowait()
            except queue.Empty:
                return
            if stop.is_set():
                record(interrupted(index, job))
                continue
            try:
                result = _execute(
                    index, job, worker, logs_dir, worker_env, register, unregister
                )
            except Exception as error:  # never lose a job if a worker hits a bug
                result = JobResult(
                    index=index,
                    job=job,
                    status=JobStatus.FAILED,
                    worker=worker,
                    exit_code=None,
                    started=None,
                    finished=time.monotonic(),
                    log_path=job_log_path(logs_dir, job.name),
                    error=f"{type(error).__name__}: {error}",
                )
                _event(f"[worker {worker}] failed {job.name} ({result.error})")
            record(result)

    previous_handlers: dict[int, object] = {}
    if threading.current_thread() is threading.main_thread():
        # signal.signal() only works on the main thread; queues started from a
        # helper thread simply keep the caller's handlers.
        for signum in _HANDLED_SIGNALS:
            previous_handlers[signum] = signal.signal(signum, handle_signal)

    threads = [
        threading.Thread(target=consume, args=(worker,), name=f"queue-worker-{worker}")
        for worker in range(workers)
    ]
    try:
        for thread in threads:
            thread.start()
        try:
            for thread in threads:
                thread.join()
        except KeyboardInterrupt:
            # Ctrl-C must also let the trainers checkpoint before we exit.
            request_stop(signal.SIGTERM)
            for thread in threads:
                thread.join()
    finally:
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)

    # Whatever the stop left in the queue never started; it must be resumed by
    # the next launch, not silently dropped.
    while True:
        try:
            index, job = pending.get_nowait()
        except queue.Empty:
            break
        record(interrupted(index, job))

    summary = QueueSummary(
        results=tuple(sorted(results, key=lambda result: result.index)),
        skipped=tuple(skipped),
        pre_failed=tuple(failed),
    )
    _event(
        f"completed: {len(summary.completed)} skipped: {len(summary.skipped)} "
        f"failed: {summary.failed_count} interrupted: {len(summary.interrupted)}"
    )
    return summary


def _send_signal(process: subprocess.Popen, signum: int) -> None:
    """Forward ``signum`` to a child that is still running."""
    if process.poll() is not None:
        return
    try:
        process.send_signal(signum)
    except (ProcessLookupError, OSError):
        pass


def _execute(
    index: int,
    job: Job,
    worker: int,
    logs_dir: Path,
    worker_env: Mapping[int, Mapping[str, str]] | None,
    register: Callable[[int, subprocess.Popen], None],
    unregister: Callable[[int], None],
) -> JobResult:
    log_path = job_log_path(logs_dir, job.name)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    _event(f"[worker {worker}] starting {job.name}")
    environment = dict(os.environ)
    if worker_env is not None:
        environment.update(worker_env.get(worker, {}))
    environment.update(job.env)
    with log_path.open("w", encoding="utf-8") as log:
        try:
            process = subprocess.Popen(
                job.command,
                cwd=job.cwd,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        except OSError as error:  # missing executable, unreadable cwd, ...
            print(f"failed to start {job.name!r}: {error}", file=log, flush=True)
            exit_code = 127
        else:
            register(worker, process)
            try:
                exit_code = process.wait()
            finally:
                unregister(worker)
    finished = time.monotonic()
    if exit_code == 0:
        status = JobStatus.COMPLETED
    elif exit_code == REQUEUE_EXIT_CODE:
        status = JobStatus.INTERRUPTED
    else:
        status = JobStatus.FAILED
    duration = finished - started
    if status is JobStatus.COMPLETED:
        _event(f"[worker {worker}] completed {job.name} exit=0 in {duration:.2f}s")
    elif status is JobStatus.INTERRUPTED:
        _event(
            f"[worker {worker}] interrupted {job.name} exit={exit_code} in {duration:.2f}s "
            f"(checkpoint saved; requeue to continue) log={log_path}"
        )
    else:
        _event(
            f"[worker {worker}] failed {job.name} exit={exit_code} "
            f"in {duration:.2f}s log={log_path}"
        )
    return JobResult(
        index=index,
        job=job,
        status=status,
        worker=worker,
        exit_code=exit_code,
        started=started,
        finished=finished,
        log_path=log_path,
    )
