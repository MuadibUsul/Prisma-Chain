#!/bin/sh
# Run the challenger against a worker over the network.
# PRISMA_WORKER_URL must point at the worker's public HTTP endpoint.
# Mode: honest (default) or fraud (development-only output-tile injection).
set -eu

REPO_DIR=${PRISMA_REPO_DIR:-/workspace/Prisma-Chain}
GEMM_PY="$REPO_DIR/compute/gemmv1/python"

WORKER_URL=${PRISMA_WORKER_URL:?set PRISMA_WORKER_URL, e.g. https://<pod-id>-8301.proxy.runpod.net}
M=${PRISMA_GEMM_M:-256}
N=${PRISMA_GEMM_N:-256}
K=${PRISMA_GEMM_K:-256}
SEED=${PRISMA_GEMM_SEED:-42}
FRAUD_TILE=${PRISMA_GEMM_FRAUD_TILE:-0,1}

cd "$GEMM_PY"
if [ "${PRISMA_GEMM_MODE:-honest}" = "fraud" ]; then
    exec python3 -m gemmv1.challenger --worker "$WORKER_URL" \
        --m "$M" --n "$N" --k "$K" --seed "$SEED" \
        --inject-fraud "$FRAUD_TILE"
fi
exec python3 -m gemmv1.challenger --worker "$WORKER_URL" \
    --m "$M" --n "$N" --k "$K" --seed "$SEED"
