#!/usr/bin/env bash
# ubuntu, expert .venv: generate demos in stages (docs/0925-smoke.md §二, final-plan §2.2).
#   expert  N_TRAIN pool demos from seed TRAIN_START and N_VAL validation demos from
#           VAL_START (run_cpu.py, pd_joint_pos, obs none, only successes kept)
#   rgb     conversion to the plan's control mode recording rgb (--shader minimal: the
#           default of a plain gym.make env, which the teammates' evaluation creates)
#   state   the same conversion recording state. CPU physics is deterministic, so both give
#           the same actions and env states; scripts/export_demos.py checks that. (A state
#           replay with --use-env-states records one-step predictions in mani-skill 3.0.1.)
#   first   scripts/first_frame_obs.py writes the correct frame 0 of both conversions to
#           *.first_obs.h5 sidecars (3.0.1 records stale contacts and a shifted env state)
#
# A step that succeeds writes <output>.done; a step whose marker exists is skipped, so a
# run can be resumed or run stage by stage. Output without a marker is an interrupted or
# still-running step (RecordEpisode writes the .json while it goes) and fails, never
# overwritten: inspect and remove it by hand. An expert file whose .json already lists the
# requested number of episodes counts as complete and gets its marker.
#
#   scripts/smoke/gen_data.sh [task ...]                          # smoke: 10 + 5 demos per task
#   OUT=demos-final PREFIX= N_TRAIN=440 N_VAL=55 N_TRAIN_POSE=700 N_VAL_POSE=90 \
#     STAGES=expert LOG=logs/final_gen_pickcube.log scripts/smoke/gen_data.sh pickcube
#
# pd_ee_delta_pose tasks lose 20-40% of demos in conversion (docs/0925-smoke.md §五), so
# N_TRAIN_POSE / N_VAL_POSE (default N_TRAIN / N_VAL) generate a surplus; the export takes
# the first N usable demos by seed, which is final-plan's "skip failed seeds" rule.
set -uo pipefail
cd "$(dirname "$0")/../.."
source scripts/smoke/common.sh

OUT=${OUT:-demos-smoke0925}
PREFIX=${PREFIX-smoke_}          # raw demos are $PREFIX{train,val}.h5; set PREFIX= for plain names
N_TRAIN=${N_TRAIN:-10}
N_VAL=${N_VAL:-5}
N_TRAIN_POSE=${N_TRAIN_POSE:-$N_TRAIN}
N_VAL_POSE=${N_VAL_POSE:-$N_VAL}
TRAIN_START=${TRAIN_START:-0}    # final-plan §2.2: pool seeds 0-3999, validation demos 4000-4999
VAL_START=${VAL_START:-4000}
STAGES=${STAGES:-expert rgb state first}
TIMEOUT=${TIMEOUT:-14400}        # per step, seconds
LOG=${LOG:-logs/$(basename "$OUT")_gen.log}
PY=${PY:-.venv/bin/python}
export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

tasks=("$@")
[ ${#tasks[@]} -gt 0 ] || tasks=("${SMOKE_TASKS[@]}")
echo "=== gen_data $(date -Is) at $(git log --oneline -1 | cut -c1-7) OUT=$OUT PREFIX=$PREFIX N_TRAIN=$N_TRAIN" \
     "N_VAL=$N_VAL N_TRAIN_POSE=$N_TRAIN_POSE N_VAL_POSE=$N_VAL_POSE STAGES=[$STAGES] tasks: ${tasks[*]}"

wants() { [[ " $STAGES " == *" $1 "* ]]; }

episode_count() {
  "$PY" -c 'import json, sys; print(len(json.load(open(sys.argv[1]))["episodes"]))' "$1" 2>/dev/null
}

# once LABEL OUTPUT EXPECTED CMD...: run CMD unless OUTPUT is marked complete.
# EXPECTED is the episode count that proves an unmarked OUTPUT complete, or - for none.
once() {
  local label=$1 output=$2 expected=$3; shift 3
  local marker="$output.done"
  if [ -e "$marker" ]; then
    RESULTS+=("SKIP  $label: done ($marker)")
    return 0
  fi
  if [ -e "$output" ] || [ -e "${output%.h5}.json" ]; then
    local found
    found=$(episode_count "${output%.h5}.json")
    if [ "$expected" != - ] && [ "$found" = "$expected" ]; then
      echo "$(date -Is) complete: $found episodes (marked afterwards)" > "$marker"
      RESULTS+=("SKIP  $label: $output has all $found episodes, marked done")
      return 0
    fi
    RESULTS+=("FAIL  $label: $output exists but is not marked done (${found:-?} episodes; interrupted or still running?); inspect and remove it by hand")
    return 1
  fi
  if step "$label" "$@"; then
    echo "$(date -Is) $*" > "$marker"
  else
    return 1
  fi
}

# raw_ready RAW COUNT: the expert file is complete (marker, or all COUNT episodes listed).
raw_ready() {
  [ -e "$1.done" ] && return 0
  [ "$(episode_count "${1%.h5}.json")" = "$2" ] || return 1
  echo "$(date -Is) complete: $2 episodes (marked afterwards)" > "$1.done"
}

for task in "${tasks[@]}"; do
  info=$(task_info "$task") || continue
  read -r env mode _ <<<"$info"
  dir="$OUT/$env/motionplanning"
  for split in train val; do
    if [ "$split" = train ]; then start=$TRAIN_START; count=$N_TRAIN; pose_count=$N_TRAIN_POSE
    else start=$VAL_START; count=$N_VAL; pose_count=$N_VAL_POSE; fi
    [ "$mode" = pd_ee_delta_pose ] && count=$pose_count
    name="$PREFIX$split"
    rgb="$dir/$name.rgb.$mode.physx_cpu.h5"
    state="$dir/$name.state.$mode.physx_cpu.h5"
    if wants expert; then
      once "$task $split: expert ($count from seed $start)" "$dir/$name.h5" "$count" \
          timeout "$TIMEOUT" "$PY" run_cpu.py -e "$env" -b physx_cpu --only-count-success -n "$count" \
          --start-seed "$start" --traj-name "$name" --record-dir "$OUT" || continue
    fi
    if { wants rgb || wants state || wants first; } && ! raw_ready "$dir/$name.h5" "$count"; then
      RESULTS+=("FAIL  $task $split: $dir/$name.h5 is missing or incomplete (not $count episodes); conversions skipped")
      continue
    fi
    if wants rgb; then
      once "$task $split: rgb replay" "$rgb" - timeout "$TIMEOUT" "$PY" -m mani_skill.trajectory.replay_trajectory \
          --traj-path "$dir/$name.h5" -b physx_cpu --use-first-env-state -c "$mode" -o rgb --shader minimal \
          --save-traj --num-envs 1 || continue
    fi
    if wants state; then
      once "$task $split: state replay" "$state" - timeout "$TIMEOUT" "$PY" -m mani_skill.trajectory.replay_trajectory \
          --traj-path "$dir/$name.h5" -b physx_cpu --use-first-env-state -c "$mode" -o state \
          --save-traj --num-envs 1 || continue
    fi
    if wants first; then
      if [ -e "${rgb%.h5}.first_obs.h5" ] && [ -e "${state%.h5}.first_obs.h5" ]; then
        RESULTS+=("SKIP  $task $split: first frame: sidecars exist")
      else
        step "$task $split: first frame" timeout "$TIMEOUT" "$PY" scripts/first_frame_obs.py "$rgb" "$state" || continue
      fi
    fi
  done
done
summary
