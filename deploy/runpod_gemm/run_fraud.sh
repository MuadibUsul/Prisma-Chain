#!/bin/sh
# Run one fraud E2E: the worker injects a wrong output tile through its
# DEVELOPMENT-ONLY flag and the challenger must detect, dispute and win.
# PRISMA_GEMM_FRAUD_TILE selects the tile (default 0,1).
set -eu

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
RESULTS_DIR=${PRISMA_RESULTS_DIR:-./gemm-results}
mkdir -p "$RESULTS_DIR"

echo "=== fraud E2E (development-only tile injection) ==="
PRISMA_GEMM_MODE=fraud "$SCRIPT_DIR/challenger.sh" 2>&1 | tee "$RESULTS_DIR/fraud.log"
python3 "$SCRIPT_DIR/collect_results.py" --results-dir "$RESULTS_DIR"
