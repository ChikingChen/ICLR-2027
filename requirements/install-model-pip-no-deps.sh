#!/usr/bin/env bash
set -euo pipefail

conda activate model

pip install -r "/data/czy/ICLR-2027/requirements/requirements-model.txt" --no-deps

python - <<'PY'
import torch
import megatron
from megatron.core import tensor_parallel

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("Megatron-LM OK")
print("Megatron Core OK")
PY
