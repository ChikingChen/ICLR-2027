#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="/data/czy/ICLR-2027/requirements"
ENVS=("data" "model_download" "model")

mkdir -p "$OUT_DIR"

for env in "${ENVS[@]}"; do
  echo "[INFO] Exporting conda environment: $env"

  conda env export -n "$env" \
    | sed '/^prefix: /d' \
    > "$OUT_DIR/environment-${env}.yml"

  conda run -n "$env" python -m pip freeze \
    > "$OUT_DIR/requirements-${env}.txt"

  if [ "$env" = "model" ]; then
    yml="$OUT_DIR/environment-${env}.yml"

    if ! grep -q -- '--extra-index-url https://download.pytorch.org/whl/cu121' "$yml"; then
      sed -i '/^[[:space:]]*- pip:$/a\      - --extra-index-url https://download.pytorch.org/whl/cu121' "$yml"
    fi
  fi

  echo "[OK] Wrote environment-${env}.yml and requirements-${env}.txt"
done
