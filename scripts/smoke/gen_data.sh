#!/usr/bin/env bash
# Phase 1 (ubuntu, expert .venv): short demos for the pipeline check.
# Per task: N_TRAIN pool demos from seed 0 and N_VAL validation demos from seed 4000
# (pd_joint_pos, obs none), then two conversions of them to the plan's control mode,
# one recording rgb and one recording state. CPU physics is deterministic, so both give
# the same actions and env states; scripts/export_demos.py checks that. (A state replay
# with --use-env-states records one-step predictions in mani-skill 3.0.1, not the states.)
# Only writes under $OUT; a task whose output directory already exists is skipped.
#
#   scripts/smoke/gen_data.sh [task ...]     # default: the six tasks of final-plan §1
set -uo pipefail
cd "$(dirname "$0")/../.."
source scripts/smoke/common.sh

OUT=${OUT:-demos-smoke0925}
N_TRAIN=${N_TRAIN:-10}
N_VAL=${N_VAL:-5}
TIMEOUT=${TIMEOUT:-1200}
PY=.venv/bin/python
export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}
mkdir -p logs
exec > >(tee -a "logs/smoke0925_gen.log") 2>&1

tasks=("$@")
[ ${#tasks[@]} -gt 0 ] || tasks=("${SMOKE_TASKS[@]}")
echo "=== gen_data $(date -Is) OUT=$OUT N_TRAIN=$N_TRAIN N_VAL=$N_VAL tasks: ${tasks[*]}"

for task in "${tasks[@]}"; do
  info=$(task_info "$task") || continue
  read -r env mode _ <<<"$info"
  dir="$OUT/$env/motionplanning"
  if [ -e "$dir" ]; then
    RESULTS+=("SKIP  $task: $dir exists")
    continue
  fi
  for split in train val; do
    if [ "$split" = train ]; then start=0; count=$N_TRAIN; else start=4000; count=$N_VAL; fi
    name="smoke_$split"
    step "$task $split: expert" timeout "$TIMEOUT" "$PY" run_cpu.py -e "$env" -b physx_cpu \
        --only-count-success -n "$count" --start-seed "$start" --traj-name "$name" --record-dir "$OUT" || continue
    # Sensor cameras use the "minimal" shader, the default of a plain gym.make env that
    # the teammates' evaluation creates; run_cpu.py records "default" in the JSON.
    step "$task $split: rgb replay" timeout "$TIMEOUT" "$PY" -m mani_skill.trajectory.replay_trajectory \
        --traj-path "$dir/$name.h5" -b physx_cpu --use-first-env-state -c "$mode" -o rgb --shader minimal \
        --save-traj --num-envs 1 || continue
    step "$task $split: state replay" timeout "$TIMEOUT" "$PY" -m mani_skill.trajectory.replay_trajectory \
        --traj-path "$dir/$name.h5" -b physx_cpu --use-first-env-state -c "$mode" -o state \
        --save-traj --num-envs 1 || continue
  done
done
summary
