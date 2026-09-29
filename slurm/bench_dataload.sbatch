#!/usr/bin/env bash
# Lazy HDF5 reads vs data.preload: DataLoader throughput benchmark on one GPU.
#
# Trains the same config with data.preload=false ("lazy") and =true
# ("preload") in the order $ORDER (default lazy preload lazy preload), one run
# at a time, and measures optimizer steps $WARMUP..$STEPS after a warm-up.
# Everything else (task, N demos, seed, batch size, workers, GPU) is shared;
# batches depend only on (seed, step), so both modes see identical batches.
# Before every run the dataset files are read once so each run starts from a
# warm page cache. Results land in $BENCH_ROOT/<job id>/: summary.md,
# summary.json, a timestamped log and a monitor CSV per run, and the run
# directories (outside the experiment RUN_ROOT). Submit from the repository root:
#
#   sbatch --export=ALL,DATA_ROOT=/scratch/$USER/dp-data/dataset slurm/bench_dataload.sbatch
#   sbatch --export=ALL,DATA_ROOT=...,TASK=pickcube,ORDER="preload lazy" slurm/bench_dataload.sbatch
#
# Extra arguments are passed to scripts/train_dp.py (e.g. --set policy.backbone='"transformer"').
# Re-summarize later with: python scripts/dataload_bench.py summarize <bench dir>
#SBATCH --job-name=dp-bench-dataload
#SBATCH --gres=gpu:1
# Same per-trainer share as the dual-GPU job (8 CPUs / 2 trainers) and failure_aware.sbatch.
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PYTHON=${PYTHON:-.venv/bin/python}
: "${DATA_ROOT:?set DATA_ROOT to the dataset root (as for --data-root)}"
TASK=${TASK:-peginsertionside}
NUM_DEMOS=${NUM_DEMOS:-100}
SEED=${SEED:-1}
NUM_WORKERS=${NUM_WORKERS:-3}
STEPS=${STEPS:-1200}
WARMUP=${WARMUP:-200}
LOG_FREQ=${LOG_FREQ:-100}
ORDER=${ORDER:-lazy preload lazy preload}
DEVICE=${DEVICE:-cuda}
BENCH_ROOT=${BENCH_ROOT:-bench/dataload}
CPUS=${SLURM_CPUS_PER_TASK:-$(getconf _NPROCESSORS_ONLN)}
BENCH_DIR="$BENCH_ROOT/${SLURM_JOB_ID:-local-$(date +%Y%m%d-%H%M%S)}"
CONFIG="configs/tasks/$TASK.toml"

if (( WARMUP % LOG_FREQ || STEPS % LOG_FREQ || WARMUP < 1 || WARMUP >= STEPS )); then
  echo "need 0 < WARMUP < STEPS, both multiples of LOG_FREQ=$LOG_FREQ" >&2
  exit 2
fi
mkdir -p "$BENCH_DIR"
mapfile -t DATA_FILES < <("$PYTHON" scripts/dataload_bench.py paths --config "$CONFIG" --data-root "$DATA_ROOT") \
  || { echo "could not resolve the dataset files of $CONFIG" >&2; exit 2; }

"$PYTHON" - "$BENCH_DIR/bench.json" "$@" <<PY
import json, subprocess, sys
revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
meta = {
    "task": "$TASK", "num_demos": $NUM_DEMOS, "seed": $SEED, "num_workers": $NUM_WORKERS,
    "steps": $STEPS, "warmup": $WARMUP, "log_freq": $LOG_FREQ, "order": "$ORDER".split(),
    "cpus": $CPUS, "device": "$DEVICE", "job_id": "${SLURM_JOB_ID:-}",
    "node": "$(hostname)", "git": revision, "extra_args": sys.argv[2:],
}
open(sys.argv[1], "w").write(json.dumps(meta, indent=2) + "\n")
PY
: > "$BENCH_DIR/runs.jsonl"
echo "bench dir: $BENCH_DIR; data: ${DATA_FILES[*]}"

declare -A repeat=()
for mode in $ORDER; do
  case "$mode" in
    lazy) preload=false ;;
    preload) preload=true ;;
    *) echo "unknown mode $mode in ORDER (expected lazy or preload)" >&2; exit 2 ;;
  esac
  repeat[$mode]=$(( ${repeat[$mode]:-0} + 1 ))
  exp="bench_${mode}_r${repeat[$mode]}"

  cat "${DATA_FILES[@]}" > /dev/null  # warm page cache: every run reads from RAM-cached files
  launched=$("$PYTHON" -c 'import time; print(f"{time.time():.3f}")')
  echo "== $exp (data.preload=$preload) launched at $launched"
  "$PYTHON" -u scripts/train_dp.py \
    --config "$CONFIG" --data-root "$DATA_ROOT" \
    --output-root "$BENCH_DIR/runs" --exp "$exp" --resume auto \
    --num-demos "$NUM_DEMOS" --seed "$SEED" --device "$DEVICE" \
    --set "data.preload=$preload" \
    --set "train.total_iters=$STEPS" \
    --set "train.log_freq=$LOG_FREQ" \
    --set "train.num_workers=$NUM_WORKERS" \
    --set "train.validation_steps=[]" \
    --set "train.checkpoint_steps=[]" \
    --set "train.resume_freq=100000000" \
    "$@" \
    > >("$PYTHON" scripts/dataload_bench.py stamp > "$BENCH_DIR/$exp.log") 2>&1 &
  trainer=$!
  "$PYTHON" scripts/dataload_bench.py monitor --pid "$trainer" --out "$BENCH_DIR/$exp.monitor.csv" &
  watcher=$!
  wait "$trainer"
  status=$?
  wait "$watcher"
  sleep 1  # let the stamp process flush the last log lines
  printf '{"exp": "%s", "mode": "%s", "launched": %s, "status": %d}\n' \
    "$exp" "$mode" "$launched" "$status" >> "$BENCH_DIR/runs.jsonl"
  echo "== $exp exit $status"
  tail -n 3 "$BENCH_DIR/$exp.log"
done

"$PYTHON" scripts/dataload_bench.py summarize "$BENCH_DIR"
