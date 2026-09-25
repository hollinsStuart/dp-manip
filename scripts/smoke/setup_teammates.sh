#!/usr/bin/env bash
# Phase 3 (wsl): clone the teammates' repos at pinned commits under $TEAMMATES_DIR and
# build each one's own environment the way its README says. Nothing outside that
# directory is touched; our ~/projects/dp-manip/.venv is not used.
#   7606-train-template: bash setup_linux_uv.sh (uv python 3.11, uv lock, uv sync, import check)
#   VariDP:              uv sync, then the checks of 学院GPU装依赖.md §3 ①-③
set -euo pipefail
export PATH="$HOME/.local/bin:/usr/lib/wsl/lib:$PATH"
unset UV_PROJECT_ENVIRONMENT  # would redirect uv sync into another venv

ROOT=${TEAMMATES_DIR:-$HOME/teammates}
TEMPLATE_REV=db107a9db8e7b3f825f6cbcdafaffe369e6dadb7
VARIDP_REV=28afb871cd18a792267c7ec264bdbf7d49e9acee
REPO=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$ROOT" "$REPO/logs"
exec > >(tee -a "$REPO/logs/smoke0925_setup.log") 2>&1
echo "=== setup_teammates $(date -Is) ROOT=$ROOT"

checkout() {
  local url=$1 dir=$ROOT/$2 rev=$3
  [ -d "$dir/.git" ] || git clone "$url" "$dir"
  git -C "$dir" checkout -q --detach "$rev"
  echo "$2 at $(git -C "$dir" log --oneline -1)"
}
checkout https://github.com/Seanzrui/7606-train-template 7606-train-template "$TEMPLATE_REV"
checkout https://github.com/hryang1130/VariDP VariDP "$VARIDP_REV"

echo "=== 7606-train-template: setup_linux_uv.sh"
(cd "$ROOT/7606-train-template" && bash setup_linux_uv.sh)

echo "=== VariDP: uv sync"
cd "$ROOT/VariDP"
uv sync
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
uv run python -c "import mani_skill, sapien, gymnasium, h5py; print(mani_skill.__version__, sapien.__version__, gymnasium.__version__, h5py.__version__)"
uv run python -c "import dp; print('dp importable')"
echo "=== done $(date -Is)"
