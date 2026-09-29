"""scripts/dataload_bench.py: /proc accounting and window summaries (no GPU, no trainer)."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "dataload_bench.py"
    spec = importlib.util.spec_from_file_location("dataload_bench", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bench = load_script()


def fake_process(proc: Path, pid: int, parent: int, ticks: tuple[int, int], rss_kib: int, pss_kib: int) -> None:
    directory = proc / str(pid)
    directory.mkdir(parents=True)
    utime, stime = ticks
    # Fields after the command name: state ppid pgrp session tty tpgid flags
    # minflt cminflt majflt cmajflt utime stime ...
    (directory / "stat").write_text(f"{pid} (pt data (w)) S {parent} 1 1 0 -1 0 0 0 0 0 {utime} {stime} 0 0\n")
    (directory / "smaps_rollup").write_text(f"Rss:  {rss_kib} kB\nPss:  {pss_kib} kB\n")


class ProcAccountingTest(unittest.TestCase):
    def test_tree_sums_descendants_only(self) -> None:
        ticks = bench.os.sysconf("SC_CLK_TCK")
        with tempfile.TemporaryDirectory() as directory:
            proc = Path(directory)
            fake_process(proc, 10, 1, (ticks, ticks), 2048, 1024)  # trainer
            fake_process(proc, 11, 10, (2 * ticks, 0), 1024, 512)  # worker
            fake_process(proc, 12, 11, (ticks, 0), 1024, 512)  # grandchild
            fake_process(proc, 99, 1, (50 * ticks, 0), 9999, 9999)  # unrelated
            (proc / "self").mkdir()

            self.assertEqual(sorted(bench.process_tree(10, proc)), [10, 11, 12])
            usage = bench.tree_usage(10, proc)
            self.assertEqual(usage["cpu_s"], 5.0)
            self.assertEqual(usage["rss_mb"], 4.0)
            self.assertEqual(usage["pss_mb"], 2.0)
            self.assertEqual(usage["processes"], 3)
            self.assertIsNone(bench.tree_usage(12345, proc))


class SummarizeTest(unittest.TestCase):
    def write_run(self, root: Path, exp: str, *, start: float, rate: float, cores: float, gpu: float) -> None:
        run_dir = root / "runs" / exp
        run_dir.mkdir(parents=True)
        steps = [1, 100, 200, 300]
        # elapsed_s counts from the training loop start (just before step 1).
        with (run_dir / "metrics.jsonl").open("w") as stream:
            for step in steps:
                stream.write(json.dumps({"step": step, "train_loss": 1.0 / step, "elapsed_s": step / rate}) + "\n")
        lines = [f"{start:.3f} loading"]
        if exp.startswith("bench_preload"):
            lines.append(f"{start + 1:.3f} preloaded 3 train + 1 val episodes in 0.8 s (0.250 GiB)")
        lines += [f"{start + 2 + step / rate:.3f} [{step:06d}/300] loss=0.1 lr=1e-4 elapsed=0s" for step in steps]
        (root / f"{exp}.log").write_text("\n".join(lines) + "\n")
        with (root / f"{exp}.monitor.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=bench.MONITOR_FIELDS)
            writer.writeheader()
            for second in range(int(2 + 300 / rate) + 2):
                moment = start + second
                busy = start + 2 + 100 / rate <= moment
                writer.writerow(
                    {
                        "time": moment,
                        # Utilization outside the window must not leak into the average.
                        "gpu_util": gpu if busy else 0,
                        "gpu_mem_mb": 1000,
                        "cpu_s": cores * second,
                        "rss_mb": 3000,
                        "pss_mb": 2048 if second > 1 else 100,
                        "processes": 4,
                    }
                )

    def test_window_metrics_and_speedup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meta = {"task": "pegs", "num_demos": 3, "seed": 1, "num_workers": 3,
                    "steps": 300, "warmup": 100, "log_freq": 100, "cpus": 4, "job_id": "1"}
            (root / "bench.json").write_text(json.dumps(meta))
            self.write_run(root, "bench_lazy_r1", start=1000.0, rate=2.0, cores=4.0, gpu=40.0)
            self.write_run(root, "bench_preload_r1", start=2000.0, rate=5.0, cores=1.0, gpu=95.0)
            self.write_run(root, "bench_lazy_r2", start=3000.0, rate=2.0, cores=4.0, gpu=40.0)
            records = [
                {"exp": "bench_lazy_r1", "mode": "lazy", "launched": 999.0, "status": 0},
                {"exp": "bench_preload_r1", "mode": "preload", "launched": 1999.0, "status": 0},
                {"exp": "bench_lazy_r2", "mode": "lazy", "launched": 2999.0, "status": 0},
                {"exp": "bench_preload_r2", "mode": "preload", "launched": 3999.0, "status": 1},
            ]
            (root / "runs.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))

            summary = bench.summarize(root)
            lazy, preload = summary["runs"][0], summary["runs"][1]
            self.assertEqual([run["exp"] for run in summary["runs"]], ["bench_lazy_r1", "bench_preload_r1", "bench_lazy_r2"])
            self.assertEqual(lazy["measured_steps"], 200)
            self.assertAlmostEqual(lazy["training_wall_s"], 100.0)
            self.assertAlmostEqual(lazy["steps_per_second"], 2.0)
            self.assertAlmostEqual(preload["steps_per_second"], 5.0)
            self.assertAlmostEqual(lazy["avg_gpu_utilization"], 40.0)
            self.assertAlmostEqual(preload["avg_gpu_utilization"], 95.0)
            self.assertAlmostEqual(lazy["cpu_cores_used"], 4.0)
            self.assertAlmostEqual(lazy["avg_cpu_utilization"], 1.0)
            self.assertAlmostEqual(preload["avg_cpu_utilization"], 0.25)
            self.assertAlmostEqual(preload["peak_ram_gb"], 2.0)
            self.assertEqual((preload["preload_s"], preload["preload_ram_gb"]), (0.8, 0.25))
            self.assertEqual((lazy["preload_s"], lazy["preload_ram_gb"]), (0.0, 0.0))
            self.assertAlmostEqual(lazy["startup_s"], 1002.5 - 999.0)
            self.assertEqual(summary["modes"]["lazy"]["runs"], 2)
            self.assertAlmostEqual(summary["speedup_steps_per_second"], 2.5)
            self.assertEqual(summary["max_train_loss_gap_first_pair"], 0.0)
            self.assertTrue((root / "summary.md").is_file())

    def test_warmup_must_be_a_logged_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bench.json").write_text(json.dumps({"warmup": 150, "cpus": 4}))
            self.write_run(root, "bench_lazy_r1", start=1000.0, rate=2.0, cores=4.0, gpu=40.0)
            (root / "runs.jsonl").write_text(
                json.dumps({"exp": "bench_lazy_r1", "mode": "lazy", "launched": 999.0, "status": 0}) + "\n"
            )
            with self.assertRaisesRegex(ValueError, "warm-up step 150"):
                bench.summarize(root)


if __name__ == "__main__":
    unittest.main()
