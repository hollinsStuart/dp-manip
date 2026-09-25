# Shared by the scripts/smoke/*.sh pipeline check (sourced, not run).
# Task table from docs/final-plan.md §1: env id, control mode, eval episode length.
# Works under macOS bash 3.2 (no associative arrays).

SMOKE_TASKS=(pickcube stackcube pushcube pullcube peginsertionside plugcharger)

task_info() {
  case "$1" in
    pickcube)          echo "PickCube-v1 pd_ee_delta_pos 100" ;;
    stackcube)         echo "StackCube-v1 pd_ee_delta_pos 200" ;;
    pushcube)          echo "PushCube-v1 pd_ee_delta_pos 100" ;;
    pullcube)          echo "PullCube-v1 pd_ee_delta_pos 100" ;;
    peginsertionside)  echo "PegInsertionSide-v1 pd_ee_delta_pose 300" ;;
    plugcharger)       echo "PlugCharger-v1 pd_ee_delta_pose 200" ;;
    liftpegupright)    echo "LiftPegUpright-v1 pd_ee_delta_pose 300" ;;  # fallback for PlugCharger
    *) echo "unknown task: $1" >&2; return 1 ;;
  esac
}

# step LABEL CMD...: run CMD, record OK/FAIL with the elapsed time, keep going.
RESULTS=()
step() {
  local label=$1; shift
  echo "=== [$(date +%H:%M:%S)] $label: $*"
  local start=$SECONDS
  if "$@"; then
    RESULTS+=("OK    $label ($((SECONDS - start))s)")
  else
    local code=$?
    RESULTS+=("FAIL  $label (exit $code, $((SECONDS - start))s)")
    return "$code"
  fi
}

summary() {
  echo "=== summary"
  printf '%s\n' "${RESULTS[@]}"
  ! printf '%s\n' "${RESULTS[@]}" | grep -q '^FAIL'
}

# WSL2 has no NVIDIA Vulkan ICD; SAPIEN then needs Mesa's lavapipe to create a renderer
# (same rule as dp_manip/envs.py::ensure_render_icd). No-op where the NVIDIA ICD exists.
use_lavapipe_if_needed() {
  if [ -z "${VK_ICD_FILENAMES:-}" ] && [ ! -e /usr/share/vulkan/icd.d/nvidia_icd.json ] \
      && [ -e /usr/share/vulkan/icd.d/lvp_icd.json ]; then
    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json
    echo "VK_ICD_FILENAMES=$VK_ICD_FILENAMES (lavapipe)"
  fi
}
