#!/bin/sh
# Bootstrap four equal-power validators sharing one genesis.
# Runs inside a one-shot container with the repo's chain image:
#
#   docker compose -f deploy/multivalidator/compose.yaml run --rm init
#
# Layout (host bind mounts):
#   deploy/multivalidator/data/validator-{a,b,c,d}/
#
# Every node gets an independent node key, consensus key, home and P2P
# identity. The genesis is generated once and distributed after all four
# gentxs are collected, so all validators start from identical genesis with
# 25% voting power each.
set -eu

CHAIN_ID=${PRISMA_MV_CHAIN_ID:-prisma-mv-1}
DATA=${PRISMA_MV_DATA:-/data}
STAKING=${PRISMA_MV_STAKE:-1000000uprsm}
DENOM=uprsm

NODES="a b c d"

if [ -f "$DATA/.mv-ready" ]; then
    echo "multivalidator data already initialized in $DATA; remove it to re-init" >&2
    exit 1
fi

echo "[init] creating four independent validator homes"
for v in $NODES; do
    H="$DATA/validator-$v"
    rm -rf "$H"
    prismad init "validator-$v" --chain-id "$CHAIN_ID" --home "$H" >/dev/null
    prismad keys add validator --keyring-backend test --keyring-dir "$H" --home "$H" --no-backup >/dev/null
done

echo "[init] funding all four accounts and customizing genesis"
A="$DATA/validator-a"
# Same uprsm-only economics as the single-node devnet.
jq '
  .app_state.staking.params.bond_denom = "uprsm" |
  .app_state.mint.params.mint_denom = "uprsm" |
  .app_state.mint.minter.inflation = "0.000000000000000000" |
  .app_state.mint.params.inflation_rate_change = "0.000000000000000000" |
  .app_state.mint.params.inflation_max = "0.000000000000000000" |
  .app_state.mint.params.inflation_min = "0.000000000000000000" |
  .app_state.gov.params.min_deposit |= (if type == "array" then map(.denom = "uprsm") else . end) |
  .app_state.gov.params.expedited_min_deposit |= (if type == "array" then map(.denom = "uprsm") else . end)
' "$A/config/genesis.json" > "$A/config/genesis.json.tmp"
mv "$A/config/genesis.json.tmp" "$A/config/genesis.json"
for v in $NODES; do
    H="$DATA/validator-$v"
    addr=$(prismad keys show validator -a --keyring-backend test --keyring-dir "$H" --home "$H")
    prismad genesis add-genesis-account "$addr" 100000000000"$DENOM" --home "$A"
    echo "[init] validator-$v account $addr"
done

# Distribute the funded genesis so every gentx signs the same document.
for v in $NODES; do
    H="$DATA/validator-$v"
    if [ "$H" != "$A" ]; then
        cp "$A/config/genesis.json" "$H/config/genesis.json"
    fi
    prismad genesis gentx validator "$STAKING" --chain-id "$CHAIN_ID" \
        --keyring-backend test --keyring-dir "$H" --home "$H" >/dev/null
done

echo "[init] collecting gentxs (equal voting power)"
for v in b c d; do
    cp "$DATA/validator-$v"/config/gentx/*.json "$A/config/gentx/"
done
prismad genesis collect-gentxs --home "$A" >/dev/null
for v in b c d; do
    cp "$A/config/genesis.json" "$DATA/validator-$v/config/genesis.json"
done

echo "[init] wiring persistent peers"
peers=""
for v in $NODES; do
    id=$(prismad comet show-node-id --home "$DATA/validator-$v")
    if [ -n "$peers" ]; then peers="$peers,"; fi
    peers="${peers}${id}@validator-${v}:26656"
done
for v in $NODES; do
    H="$DATA/validator-$v"
    sed -i "s/^persistent_peers = .*/persistent_peers = \"$peers\"/" "$H/config/config.toml"
    sed -i 's/^addr_book_strict = .*/addr_book_strict = false/' "$H/config/config.toml"
    sed -i 's/^keyring-backend = .*/keyring-backend = "test"/' "$H/config/client.toml"
done

touch "$DATA/.mv-ready"
echo "[init] done: $DATA/.mv-ready; peers=$peers"
