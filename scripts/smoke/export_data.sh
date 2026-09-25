#!/usr/bin/env bash
# Phase 2 (Mac): turn the replays copied from ubuntu into teammate demo files.
#   rsync -a ubuntu:Coding/dp-manip/demos-smoke0925/ demos-smoke0925/
#   scripts/smoke/export_data.sh [task ...]
# Writes $DST/<task>/<task>_{train,val}.{h5,json} and a <task>_train_preview.png.
set -uo pipefail
cd "$(dirname "$0")/../.."
source scripts/smoke/common.sh

SRC=${SRC:-demos-smoke0925}
DST=${DST:-data/smoke0925}
PY=${PY:-.venv/bin/python}

tasks=("$@")
[ ${#tasks[@]} -gt 0 ] || tasks=("${SMOKE_TASKS[@]}")
for task in "${tasks[@]}"; do
  info=$(task_info "$task") || continue
  read -r env mode _ <<<"$info"
  dir="$SRC/$env/motionplanning"
  for split in train val; do
    step "$task $split: export" "$PY" scripts/export_demos.py \
        --rgb "$dir/smoke_$split.rgb.$mode.physx_cpu.h5" --state "$dir/smoke_$split.state.$mode.physx_cpu.h5" \
        -o "$DST/$task/${task}_$split.h5" --split "$split" || continue
  done
  [ -e "$DST/$task/${task}_train.h5" ] && step "$task: preview" "$PY" scripts/smoke/preview_rgb.py "$DST/$task/${task}_train.h5" "$DST/$task/${task}_train_preview.png"
done
summary
