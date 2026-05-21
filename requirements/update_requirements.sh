#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="/data/czy/ICLR-2027/requirements"
ENVS=("data" "model_download" "model")

FLASH_ATTN_WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

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
    req="$OUT_DIR/requirements-${env}.txt"
    
    if ! grep -q -- '--extra-index-url https://download.pytorch.org/whl/cu121' "$yml"; then
      sed -i '/^[[:space:]]*- pip:$/a\      - --extra-index-url https://download.pytorch.org/whl/cu121' "$yml"
    fi

    sed -i '/^flash-attn[ =@]/d;/^flash_attn[ =@]/d' "$req"
    echo "flash-attn @ ${FLASH_ATTN_WHEEL_URL}" >> "$req"

    sed -i '/^[[:space:]]*- flash-attn[ =@]/d;/^[[:space:]]*- flash_attn[ =@]/d' "$yml"
    sed -i "/^[[:space:]]*- --extra-index-url https:\/\/download.pytorch.org\/whl\/cu121/a\      - flash-attn @ ${FLASH_ATTN_WHEEL_URL}" "$yml"

    if ! grep -q "flash-attn @ ${FLASH_ATTN_WHEEL_URL}" "$yml"; then
      sed -i "/^[[:space:]]*- pip:$/a\      - flash-attn @ ${FLASH_ATTN_WHEEL_URL}" "$yml"
    fi
  fi

  echo "[OK] Wrote environment-${env}.yml and requirements-${env}.txt"
done