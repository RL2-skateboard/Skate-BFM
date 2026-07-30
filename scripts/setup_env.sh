#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${SKATE_BFM_ENV:-skatebfm}"

git -C "$ROOT" submodule update --init --depth 1

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda env update --name "$ENV_NAME" --file "$ROOT/environment.yml" --prune
else
  conda env create --name "$ENV_NAME" --file "$ROOT/environment.yml"
fi

conda run --name "$ENV_NAME" python -m pip install --no-deps -e "$ROOT/husky_sim"
echo "Environment ready. Activate with: conda activate $ENV_NAME"
