#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${SKATE_BFM_ENV:-skate-sim}"
SIM_ROOT="$(cd "$ROOT/../skate-sim" && pwd)"
cd "$ROOT"


if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda env update --name "$ENV_NAME" --file "$ROOT/environment.yml" --prune
else
  conda env create --name "$ENV_NAME" --file "$ROOT/environment.yml"
fi

conda run --name "$ENV_NAME" python -m pip install -e "$ROOT[motionlib]" -e "$SIM_ROOT[mujoco,mjlab,contact]"
echo "Environment ready. Activate with: conda activate $ENV_NAME"
