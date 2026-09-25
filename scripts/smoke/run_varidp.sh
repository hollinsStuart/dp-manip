#!/usr/bin/env bash
# Phase 4 (wsl): short run of VariDP's own train/train.py and train/eval.py on the exported
# demos, passed with --h5. PickCube trains all three backbones, the other tasks unet only.
#   scripts/smoke/run_varidp.sh [task ...]
#   STEPS=1000 EPISODES=8 VARIDP_BACKEND=cpu scripts/smoke/run_varidp.sh pickcube
# The cpu backend runs one env at a time (ManiSkill cannot vectorize physx_cpu in one
# process). Without an NVIDIA Vulkan ICD (WSL2), eval.py runs through varidp_eval.py,
# which only adds render_backend="none" to gym.make; the ICD-less default render device fails.
set -uo pipefail
export PATH="$HOME/.local/bin:/usr/lib/wsl/lib:$PATH"
unset UV_PROJECT_ENVIRONMENT
REPO=$(cd "$(dirname "$0")/../.." && pwd)
source "$REPO/scripts/smoke/common.sh"

VARIDP=${TEAMMATES_DIR:-$HOME/teammates}/VariDP
DATA=${DATA:-$REPO/data/smoke0925}
STEPS=${STEPS:-1000}
EPISODES=${EPISODES:-8}
BACKEND=${VARIDP_BACKEND:-}  # empty: eval.py's default (auto = gpu when CUDA is available)
if [ "$BACKEND" = cpu ]; then NUM_ENVS=${NUM_ENVS:-1}; else NUM_ENVS=${NUM_ENVS:-$EPISODES}; fi
EVAL=(train/eval.py)
[ -e /usr/share/vulkan/icd.d/nvidia_icd.json ] || EVAL=("$REPO/scripts/smoke/varidp_eval.py" train/eval.py)
mkdir -p "$REPO/logs"
exec > >(tee -a "$REPO/logs/smoke0925_varidp.log") 2>&1
echo "=== run_varidp $(date -Is) at $(git -C "$VARIDP" log --oneline -1) STEPS=$STEPS EPISODES=$EPISODES NUM_ENVS=$NUM_ENVS BACKEND=${BACKEND:-default} EVAL=${EVAL[*]}"
use_lavapipe_if_needed

tasks=("$@")
[ ${#tasks[@]} -gt 0 ] || tasks=("${SMOKE_TASKS[@]}")
cd "$VARIDP"
for task in "${tasks[@]}"; do
  info=$(task_info "$task") || continue
  read -r env mode steps <<<"$info"
  backbones=unet
  [ "$task" = pickcube ] && backbones="mlp unet transformer"
  for backbone in $backbones; do
    name="smoke_${task}_$backbone"
    step "$task $backbone: train" uv run python train/train.py --env-id "$env" --h5 "$(demo_file "$DATA" train "$task")" \
        --backbone "$backbone" --total-iters "$STEPS" --control-mode "$mode" --max-episode-steps "$steps" \
        --exp-name "$name" || continue
    step "$task $backbone: eval" uv run python "${EVAL[@]}" --ckpt "train/runs/$name/best.pt" \
        -n "$EPISODES" --num-envs "$NUM_ENVS" ${BACKEND:+--sim-backend "$BACKEND"}
  done
done
summary
