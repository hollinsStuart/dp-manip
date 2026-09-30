#!/usr/bin/env python3
"""Helpers for the lazy-vs-preload DataLoader benchmark (slurm/bench_dataload.sbatch).

The benchmark trains the same config with ``data.preload`` off and on, in an
interleaved order, and measures a fixed window of optimizer steps after a
warm-up. The trainer itself is not instrumented; these subcommands run beside
it:

  paths      print the train/val HDF5 files of a config (to warm the page cache)
  stamp      copy stdin to stdout, prefixing each line with a Unix timestamp
  monitor    sample GPU utilization and the trainer's process-tree CPU/RAM
  summarize  turn the logs of every run into a per-run and per-mode table

Window metrics come from the timestamped trainer log: the ``[step/total]`` line
of the warm-up step and of the last step bound the window. CPU is the CPU time
of the trainer and its DataLoader workers inside that window; RAM is the summed
PSS (proportional set size), so pages the preloaded arrays share copy-on-write
with forked workers are counted once.

  sbatch --export=ALL,DATA_ROOT=... slurm/bench_dataload.sbatch
  python scripts/dataload_bench.py summarize <bench dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MONITOR_FIELDS = ("time", "gpu_util", "gpu_mem_mb", "cpu_s", "rss_mb", "pss_mb", "processes")
STEP_LINE = re.compile(r"^\[(\d+)/(\d+)\] loss=")
PRELOAD_LINE = re.compile(r"preloaded .* in ([0-9.]+) s \(([0-9.]+) GiB\)")


# ------------------------------------------------------------------ paths


def dataset_paths(args: argparse.Namespace) -> list[Path]:
    from dp_manip import config as config_lib

    cfg = config_lib.load_run(args.config, data_root=args.data_root)
    root = Path(cfg.data.root)
    return [root / cfg.data.train_path, root / cfg.data.val_path]


# ------------------------------------------------------------------ stamp


def stamp() -> None:
    for line in sys.stdin:
        sys.stdout.write(f"{time.time():.3f} {line}")
        sys.stdout.flush()


# ------------------------------------------------------------------ monitor


def process_tree(root_pid: int, proc: Path = Path("/proc")) -> list[int]:
    """``root_pid`` and all of its live descendants, read from ``/proc``."""
    children: dict[int, list[int]] = {}
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
        except OSError:
            continue
        # The command name may contain spaces or parentheses; fields follow the last ")".
        parent = int(stat.rsplit(")", 1)[1].split()[1])
        children.setdefault(parent, []).append(int(entry.name))
    tree, pending = [], [root_pid]
    while pending:
        pid = pending.pop()
        tree.append(pid)
        pending.extend(children.get(pid, ()))
    return tree


def tree_usage(root_pid: int, proc: Path = Path("/proc")) -> dict[str, float] | None:
    """CPU seconds, RSS and PSS summed over the process tree (``None`` without /proc)."""
    if not (proc / str(root_pid)).is_dir():
        return None
    ticks = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
    cpu = rss = pss = 0.0
    alive = 0
    for pid in process_tree(root_pid, proc):
        try:
            fields = (proc / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
        except OSError:
            continue
        alive += 1
        # utime, stime are fields 14 and 15 of /proc/<pid>/stat; after the command
        # name (field 2) is split off they are at offsets 11 and 12.
        cpu += (int(fields[11]) + int(fields[12])) / ticks
        try:
            for line in (proc / str(pid) / "smaps_rollup").read_text().splitlines():
                name, _, rest = line.partition(":")
                if name in ("Rss", "Pss"):
                    kib = float(rest.split()[0])
                    if name == "Rss":
                        rss += kib / 1024
                    else:
                        pss += kib / 1024
        except OSError:
            pass
    return {"cpu_s": cpu, "rss_mb": rss, "pss_mb": pss, "processes": alive}


def gpu_sample() -> tuple[float, float] | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    # A one-GPU job sees only its own card.
    util, memory = output.strip().splitlines()[0].split(",")
    return float(util), float(memory)


def monitor(pid: int, out: Path, interval: float) -> None:
    with out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MONITOR_FIELDS)
        writer.writeheader()
        while True:
            now = time.time()
            usage = tree_usage(pid)
            if usage is None and Path("/proc").is_dir():
                break  # the trainer exited
            gpu = gpu_sample()
            row: dict[str, Any] = {key: "" for key in MONITOR_FIELDS}
            row["time"] = f"{now:.3f}"
            if gpu is not None:
                row["gpu_util"], row["gpu_mem_mb"] = gpu
            if usage is not None:
                row.update(usage)
            writer.writerow(row)
            stream.flush()
            if usage is None:
                # No /proc (e.g. macOS smoke runs): poll the pid instead.
                try:
                    os.kill(pid, 0)
                except OSError:
                    break
            time.sleep(max(0.0, interval - (time.time() - now)))


# ------------------------------------------------------------------ summarize


def _number(value: str) -> float | None:
    return float(value) if value not in ("", None) else None


def parse_log(path: Path) -> dict[str, Any]:
    steps: dict[int, float] = {}
    preload_s = preload_gib = None
    launch = None
    for line in path.read_text(encoding="utf-8").splitlines():
        stamp_text, _, text = line.partition(" ")
        try:
            moment = float(stamp_text)
        except ValueError:
            continue
        launch = moment if launch is None else launch
        match = STEP_LINE.match(text)
        if match:
            steps[int(match.group(1))] = moment
        match = PRELOAD_LINE.search(text)
        if match:
            preload_s, preload_gib = float(match.group(1)), float(match.group(2))
    return {"steps": steps, "preload_s": preload_s, "preload_gib": preload_gib, "first_line": launch}


def read_metrics(path: Path) -> dict[int, dict[str, Any]]:
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        records[int(record["step"])] = record
    return records


def summarize_run(run: dict[str, Any], bench_dir: Path, warmup: int, cpus: int) -> dict[str, Any]:
    name = run["exp"]
    log = parse_log(bench_dir / f"{name}.log")
    metrics = read_metrics(bench_dir / "runs" / name / "metrics.jsonl")
    last = max(step for step in metrics if "elapsed_s" in metrics[step])
    if warmup not in metrics or "elapsed_s" not in metrics[warmup]:
        raise ValueError(f"{name}: warm-up step {warmup} is not a logged step (log_freq must divide it)")
    measured = last - warmup
    wall = metrics[last]["elapsed_s"] - metrics[warmup]["elapsed_s"]
    start, end = log["steps"].get(warmup), log["steps"].get(last)

    rows = []
    with (bench_dir / f"{name}.monitor.csv").open(encoding="utf-8") as stream:
        rows = [{key: _number(value) for key, value in row.items()} for row in csv.DictReader(stream)]
    window = [row for row in rows if start is not None and end is not None and start <= row["time"] <= end]
    gpu = [row["gpu_util"] for row in window if row["gpu_util"] is not None]
    cpu_rows = [row for row in window if row["cpu_s"] is not None]
    cpu_cores = cpu_util = None
    if len(cpu_rows) >= 2:
        first, final = cpu_rows[0], cpu_rows[-1]
        span = final["time"] - first["time"]
        if span > 0:
            cpu_cores = (final["cpu_s"] - first["cpu_s"]) / span
            cpu_util = cpu_cores / cpus
    pss = [row["pss_mb"] for row in rows if row["pss_mb"]]
    rss = [row["rss_mb"] for row in rows if row["rss_mb"]]
    startup = (
        log["steps"][1] - run["launched"]
        if 1 in log["steps"] and run.get("launched") is not None
        else None
    )
    return {
        "exp": name,
        "mode": run["mode"],
        "startup_s": startup,
        "preload_s": log["preload_s"] if run["mode"] == "preload" else 0.0,
        "preload_ram_gb": log["preload_gib"] if run["mode"] == "preload" else 0.0,
        "measured_steps": measured,
        "training_wall_s": wall,
        "steps_per_second": measured / wall if wall > 0 else None,
        "avg_gpu_utilization": statistics.fmean(gpu) if gpu else None,
        "avg_cpu_utilization": cpu_util,
        "cpu_cores_used": cpu_cores,
        "peak_ram_gb": (max(pss) if pss else max(rss) if rss else 0) / 1024 or None,
        "gpu_samples": len(gpu),
        "losses": {step: metrics[step]["train_loss"] for step in sorted(metrics)},
    }


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def summarize(bench_dir: Path) -> dict[str, Any]:
    meta = json.loads((bench_dir / "bench.json").read_text(encoding="utf-8"))
    # One line per finished run, appended by the sbatch script, so a job that
    # stopped early still summarizes the runs it completed.
    records = [
        json.loads(line)
        for line in (bench_dir / "runs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    failed = [record["exp"] for record in records if record.get("status") != 0]
    if failed:
        print(f"skipping runs that did not exit 0: {failed}", file=sys.stderr)
    runs = [
        summarize_run(record, bench_dir, meta["warmup"], meta["cpus"])
        for record in records
        if record.get("status") == 0
    ]
    modes: dict[str, dict[str, Any]] = {}
    for mode in ("lazy", "preload"):
        chosen = [run for run in runs if run["mode"] == mode]
        if not chosen:
            continue
        mean = {}
        for key in ("steps_per_second", "training_wall_s", "avg_gpu_utilization", "avg_cpu_utilization",
                    "cpu_cores_used", "peak_ram_gb", "preload_s", "preload_ram_gb", "startup_s"):
            values = [run[key] for run in chosen if run[key] is not None]
            mean[key] = statistics.fmean(values) if values else None
        modes[mode] = {"runs": len(chosen), **mean}

    # Batches depend only on (seed, step), so both modes must see the same losses
    # up to non-deterministic GPU kernels.
    loss_gap = None
    lazy = [run for run in runs if run["mode"] == "lazy"]
    preload = [run for run in runs if run["mode"] == "preload"]
    if lazy and preload:
        common = set(lazy[0]["losses"]) & set(preload[0]["losses"])
        loss_gap = max(
            (abs(lazy[0]["losses"][step] - preload[0]["losses"][step]) for step in common), default=None
        )
    speedup = None
    if "lazy" in modes and "preload" in modes and modes["lazy"]["steps_per_second"]:
        speedup = modes["preload"]["steps_per_second"] / modes["lazy"]["steps_per_second"]

    summary = {
        "meta": meta,
        "runs": [{key: value for key, value in run.items() if key != "losses"} for run in runs],
        "modes": modes,
        "speedup_steps_per_second": speedup,
        "max_train_loss_gap_first_pair": loss_gap,
    }
    (bench_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    columns = [
        ("run", "exp", 0), ("steps/s", "steps_per_second", 2), ("wall s", "training_wall_s", 1),
        ("GPU %", "avg_gpu_utilization", 1), ("CPU %", "avg_cpu_utilization", 3),
        ("cores", "cpu_cores_used", 2), ("peak RAM GB", "peak_ram_gb", 2),
        ("preload s", "preload_s", 1), ("preload GB", "preload_ram_gb", 2), ("startup s", "startup_s", 1),
    ]
    lines = [
        f"# {meta['task']} n{meta['num_demos']} seed {meta['seed']}: steps {meta['warmup']}-{meta['steps']}, "
        f"{meta['num_workers']} workers, {meta['cpus']} CPUs, job {meta.get('job_id')}",
        "",
        "| " + " | ".join(title for title, _, _ in columns) + " |",
        "|" + "---|" * len(columns),
    ]
    for run in runs:
        cells = []
        for _, key, digits in columns:
            value = run[key]
            if key == "avg_cpu_utilization" and value is not None:
                value = 100 * value
                digits = 1
            cells.append(_fmt(value, digits))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    for mode, values in modes.items():
        cpu = values["avg_cpu_utilization"]
        gpu = values["avg_gpu_utilization"]
        lines.append(
            f"{mode:8} x{values['runs']}: {_fmt(values['steps_per_second'])} steps/s, "
            f"GPU {'n/a' if gpu is None else f'{gpu:.1f}%'}, "
            f"CPU {'n/a' if cpu is None else f'{100 * cpu:.1f}%'}, "
            f"peak RAM {_fmt(values['peak_ram_gb'])} GB"
        )
    lines.append(f"speedup (preload / lazy steps/s): {'n/a' if speedup is None else f'{speedup:.2f}x'}")
    lines.append(f"max |train_loss| gap lazy vs preload (first pair): {_fmt(loss_gap, 6)}")
    report = "\n".join(lines) + "\n"
    (bench_dir / "summary.md").write_text(report, encoding="utf-8")
    print(report, end="")
    return summary


# ------------------------------------------------------------------ cli


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    paths = commands.add_parser("paths", help="print the dataset files of a config")
    paths.add_argument("--config", required=True, type=Path)
    paths.add_argument("--data-root", type=Path)
    commands.add_parser("stamp", help="prefix stdin lines with a Unix timestamp")
    watch = commands.add_parser("monitor", help="sample GPU and process-tree CPU/RAM")
    watch.add_argument("--pid", required=True, type=int)
    watch.add_argument("--out", required=True, type=Path)
    watch.add_argument("--interval", type=float, default=1.0)
    report = commands.add_parser("summarize", help="write summary.json and summary.md")
    report.add_argument("bench_dir", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "paths":
        for path in dataset_paths(args):
            print(path)
    elif args.command == "stamp":
        stamp()
    elif args.command == "monitor":
        monitor(args.pid, args.out, args.interval)
    else:
        summarize(args.bench_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
