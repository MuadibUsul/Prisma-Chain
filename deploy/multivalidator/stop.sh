#!/bin/sh
# Stop the four-validator devnet; data directories are preserved.
set -eu
DIR=$(cd "$(dirname "$0")" && pwd)
docker compose -f "$DIR/compose.yaml" down
