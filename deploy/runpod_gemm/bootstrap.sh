#!/bin/sh
# Bootstrap one RunPod GPU node for the GEMM v1 E2E.
# Requires: python3.10+, pip; torch with CUDA is expected on RunPod images.
# No credentials are printed or stored by this script.
set -eu

REPO_DIR=${PRISMA_REPO_DIR:-/workspace/Prisma-Chain}
GEMM_PY="$REPO_DIR/compute/gemmv1/python"

if [ ! -d "$GEMM_PY/gemmv1" ]; then
    echo "bootstrap: $GEMM_PY not found; set PRISMA_REPO_DIR or clone the repo" >&2
    exit 1
fi

echo "[bootstrap] python: $(python3 --version 2>&1)"
python3 - <<'EOF'
try:
    import torch
    print(f"[bootstrap] torch {torch.__version__}, cuda={torch.cuda.is_available()}")
except Exception as exc:
    print(f"[bootstrap] torch unavailable: {exc}")
EOF

cd "$GEMM_PY"
python3 - <<'EOF'
from gemmv1.gpu import backend_status
status = backend_status()
print(f"[bootstrap] gemm backend: {status}")
if status["backend"] != "torch_int_mm_cuda":
    print("[bootstrap] WARNING: no bit-exact INT8->INT32 GPU kernel; "
          "the node will report GPU_BACKEND_UNSUPPORTED and may only serve "
          "the CPU reference path. Do NOT present CPU runs as GPU E2E evidence.")
EOF
echo "[bootstrap] done"
