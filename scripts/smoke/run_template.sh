#!/usr/bin/env bash
# Phase 4 (wsl): short run of 7606-train-template's own commands on the exported demos.
# Per task: plugin tasks/ms_<task> (make_template_tasks.py), then run.py smoke, train and
# evaluate with the template's defaults except the lengths below. PickCube trains all three
# models, the other tasks unet only (smoke already runs every model forward/backward).
#   scripts/smoke/run_template.sh [task ...]
#   STEPS=1000 EPISODES=8 TEMPLATE_BACKEND=physx_cpu scripts/smoke/run_template.sh pickcube
set -uo pipefail
export PATH="$HOME/.local/bin:/usr/lib/wsl/lib:$PATH"
unset UV_PROJECT_ENVIRONMENT
REPO=$(cd "$(dirname "$0")/../.." && pwd)
source "$REPO/scripts/smoke/common.sh"

TEMPLATE=${TEAMMATES_DIR:-$HOME/teammates}/7606-train-template
DATA=${DATA:-$REPO/data/smoke0925}
STEPS=${STEPS:-1000}
EPISODES=${EPISODES:-8}
BACKEND=${TEMPLATE_BACKEND:-}  # empty: the plugin config's eval_backend (physx_cuda)
mkdir -p "$REPO/logs"
exec > >(tee -a "$REPO/logs/smoke0925_template.log") 2>&1
echo "=== run_template $(date -Is) at $(git -C "$TEMPLATE" log --oneline -1) STEPS=$STEPS EPISODES=$EPISODES BACKEND=${BACKEND:-config}"
use_lavapipe_if_needed

tasks=("$@")
[ ${#tasks[@]} -gt 0 ] || tasks=("${SMOKE_TASKS[@]}")
specs=()
for task in "${tasks[@]}"; do
  info=$(task_info "$task") || exit 1
  read -r _ _ steps <<<"$info"
  specs+=("$task=$steps")
done
cd "$TEMPLATE"
step "make plugins" python3 "$REPO/scripts/smoke/make_template_tasks.py" \
    --template-dir . --data-dir "$DATA" "${specs[@]}" || { summary; exit 1; }

for task in "${tasks[@]}"; do
  plugin=ms_$task
  step "$task: smoke" uv run python run.py smoke --task "$plugin" || continue
  models=unet
  [ "$task" = pickcube ] && models="mlp unet transformer"
  for model in $models; do
    step "$task $model: train" uv run python run.py train --task "$plugin" --model "$model" --seed 1 \
        --steps "$STEPS" || continue
    step "$task $model: evaluate" uv run python run.py evaluate --task "$plugin" \
        --checkpoint "runs/$plugin/$model/seed_1/best.pt" --episodes "$EPISODES" --num-envs "$EPISODES" \
        ${BACKEND:+--backend "$BACKEND"}
  done
done
summary
