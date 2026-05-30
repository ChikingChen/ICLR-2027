#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="/data/czy/ICLR-2027/requirements"
ENVS=("data" "model_download" "model")

FLASH_ATTN_WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

MEGATRON_CORE_VERSION="0.12.1"
MEGATRON_LM_GIT_URL="git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.12.1"

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

    # ------------------------------------------------------------------
    # requirements-model.txt
    # ------------------------------------------------------------------

    # Remove possibly unstable / auto-generated flash-attn entries
    sed -i \
      -e '/^flash-attn[ =@]/d' \
      -e '/^flash_attn[ =@]/d' \
      "$req"

    # Remove existing Megatron entries, including pip freeze direct-url forms
    sed -i \
      -e '/^megatron-core[= @]/Id' \
      -e '/^megatron-lm[= @]/Id' \
      -e '/^nvidia-megatron-core[= @]/Id' \
      -e '/Megatron-LM\.git/Id' \
      "$req"

    # Add fixed flash-attn wheel
    echo "flash-attn @ ${FLASH_ATTN_WHEEL_URL}" >> "$req"

    # Add fixed Megatron versions
    echo "megatron-core==${MEGATRON_CORE_VERSION}" >> "$req"
    echo "${MEGATRON_LM_GIT_URL}" >> "$req"

    # ------------------------------------------------------------------
    # environment-model.yml pip section
    # ------------------------------------------------------------------

    # Ensure PyTorch CUDA 12.1 extra index exists inside pip section
    if ! grep -q -- '--extra-index-url https://download.pytorch.org/whl/cu121' "$yml"; then
      sed -i '/^[[:space:]]*- pip:$/a\      - --extra-index-url https://download.pytorch.org/whl/cu121' "$yml"
    fi

    # Remove existing flash-attn entries from yml
    sed -i \
      -e '/^[[:space:]]*- flash-attn[ =@]/d' \
      -e '/^[[:space:]]*- flash_attn[ =@]/d' \
      "$yml"

    # Remove existing Megatron entries from yml
    sed -i \
      -e '/^[[:space:]]*- megatron-core[= @]/Id' \
      -e '/^[[:space:]]*- megatron-lm[= @]/Id' \
      -e '/^[[:space:]]*- nvidia-megatron-core[= @]/Id' \
      -e '/Megatron-LM\.git/Id' \
      "$yml"

    # Add fixed flash-attn and Megatron entries after PyTorch extra-index-url
    sed -i "/^[[:space:]]*- --extra-index-url https:\/\/download.pytorch.org\/whl\/cu121/a\      - flash-attn @ ${FLASH_ATTN_WHEEL_URL}" "$yml"
    sed -i "/^[[:space:]]*- flash-attn @ /a\      - megatron-core==${MEGATRON_CORE_VERSION}" "$yml"
    sed -i "/^[[:space:]]*- megatron-core==${MEGATRON_CORE_VERSION}/a\      - ${MEGATRON_LM_GIT_URL}" "$yml"

    # Fallback: if insertion failed for any reason, insert under pip:
    if ! grep -qF "flash-attn @ ${FLASH_ATTN_WHEEL_URL}" "$yml"; then
      sed -i "/^[[:space:]]*- pip:$/a\      - flash-attn @ ${FLASH_ATTN_WHEEL_URL}" "$yml"
    fi

    if ! grep -qF "megatron-core==${MEGATRON_CORE_VERSION}" "$yml"; then
      sed -i "/^[[:space:]]*- pip:$/a\      - megatron-core==${MEGATRON_CORE_VERSION}" "$yml"
    fi

    if ! grep -qF "${MEGATRON_LM_GIT_URL}" "$yml"; then
      sed -i "/^[[:space:]]*- pip:$/a\      - ${MEGATRON_LM_GIT_URL}" "$yml"
    fi

    # ------------------------------------------------------------------
    # Safer migration install script
    # ------------------------------------------------------------------

    cat > "$OUT_DIR/install-model-pip-no-deps.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail

conda activate model

pip install -r "$OUT_DIR/requirements-model.txt" --no-deps

python - <<'PY'
import torch
import megatron
from megatron.core import tensor_parallel

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("Megatron-LM OK")
print("Megatron Core OK")
PY
EOF

    chmod +x "$OUT_DIR/install-model-pip-no-deps.sh"
  fi

  echo "[OK] Wrote environment-${env}.yml and requirements-${env}.txt"
done

echo "[OK] Wrote safe installer: $OUT_DIR/install-model-pip-no-deps.sh"