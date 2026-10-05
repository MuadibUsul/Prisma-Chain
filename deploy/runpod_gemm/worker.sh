#!/bin/sh
# Start the GEMM worker service on one RunPod node.
# Listens on 0.0.0.0:$PRISMA_WORKER_PORT (default 8301); expose the port to
# the challenger through the RunPod proxy or a firewall rule.
set -eu

REPO_DIR=${PRISMA_REPO_DIR:-/workspace/Prisma-Chain}
GEMM_PY="$REPO_DIR/compute/gemmv1/python"
PORT=${PRISMA_WORKER_PORT:-8301}

cd "$GEMM_PY"
echo "[worker] serving on 0.0.0.0:$PORT"
exec python3 -c "from gemmv1.service import serve; serve('0.0.0.0', $PORT)"
