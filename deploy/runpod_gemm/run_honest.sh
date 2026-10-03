#!/bin/sh
# Run one honest E2E against the worker and save the report.
# PRISMA_WORKER_URL is required; see challenger.sh for the other variables.
set -eu

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
RESULTS_DIR=${PRISMA_RESULTS_DIR:-./gemm-results}
mkdir -p "$RESULTS_DIR"

echo "=== honest E2E ==="
PRISMA_GEMM_MODE=honest "$SCRIPT_DIR/challenger.sh" 2>&1 | tee "$RESULTS_DIR/honest.log"
python3 "$SCRIPT_DIR/collect_results.py" --results-dir "$RESULTS_DIR"
