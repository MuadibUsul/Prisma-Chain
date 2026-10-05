#!/bin/sh
# Bootstrap (if needed) and start the four-validator devnet.
#   CENSOR_A / CENSOR_B / CENSOR_C / CENSOR_D set the development-only
#   proposer omission harness on individual validators (default off).
set -eu
DIR=$(cd "$(dirname "$0")" && pwd)
COMPOSE="docker compose -f $DIR/compose.yaml"

if [ ! -f "$DIR/data/.mv-ready" ]; then
    echo "[start] initializing multivalidator data..."
    $COMPOSE run --rm init
fi
$COMPOSE up -d validator-a validator-b validator-c validator-d
echo "[start] four validators up; waiting for 10 blocks..."
python "$DIR/status.py" --wait-blocks 10 --timeout 180
