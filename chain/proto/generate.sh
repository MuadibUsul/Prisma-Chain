#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$root/chain"

sdk_proto=$(go list -m -f '{{.Dir}}' github.com/cosmos/cosmos-sdk)/proto
gogo_proto=$(go list -m -f '{{.Dir}}' github.com/cosmos/gogoproto)
output=$(mktemp -d)
trap 'rm -rf -- "$output"' EXIT

protoc -I "$root/chain/proto" -I "$sdk_proto" -I "$gogo_proto" -I /usr/include \
  --gogofaster_out=plugins=grpc,paths=source_relative:"$output" \
  "$root/chain/proto/prisma/compute/v1/tx.proto" \
  "$root/chain/proto/prisma/compute/v1/query.proto"
cp "$output/prisma/compute/v1/tx.pb.go" "$root/chain/x/compute/types/tx.pb.go"
cp "$output/prisma/compute/v1/query.pb.go" "$root/chain/x/compute/types/query.pb.go"
