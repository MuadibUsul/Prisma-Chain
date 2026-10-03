#!/bin/sh
# Stop or restart a single validator: restart_one.sh <a|b|c|d> <stop|start>
set -eu
DIR=$(cd "$(dirname "$0")" && pwd)
V=$1
ACTION=$2
case "$ACTION" in
  stop)  docker compose -f "$DIR/compose.yaml" stop "validator-$V" ;;
  start) docker compose -f "$DIR/compose.yaml" start "validator-$V" ;;
  *) echo "usage: restart_one.sh <a|b|c|d> <stop|start>" >&2; exit 2 ;;
esac
