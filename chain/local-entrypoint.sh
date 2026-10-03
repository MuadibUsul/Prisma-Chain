#!/bin/sh
set -eu

home_dir=${PRISMA_HOME:-/data}
chain_id=${PRISMA_CHAIN_ID:-prisma-local-1}

if [ ! -f "$home_dir/.prisma-genesis-ready" ]; then
  if [ -f "$home_dir/config/genesis.json" ]; then
    echo "partial Prisma dev genesis in $home_dir; reset this disposable local volume before retrying" >&2
    exit 1
  fi
  prismad init prisma-local --chain-id "$chain_id" --home "$home_dir" >/dev/null
  prismad keys add validator --keyring-backend test --keyring-dir "$home_dir" --home "$home_dir" --no-backup >/dev/null
  # SDK defaults use stake and 13% mint inflation. Testnet issuance is disabled
  # until Prisma's separate, audited base-task issuance schedule is implemented.
  jq '
    .app_state.staking.params.bond_denom = "uprsm" |
    .app_state.mint.params.mint_denom = "uprsm" |
    .app_state.mint.minter.inflation = "0.000000000000000000" |
    .app_state.mint.params.inflation_rate_change = "0.000000000000000000" |
    .app_state.mint.params.inflation_max = "0.000000000000000000" |
    .app_state.mint.params.inflation_min = "0.000000000000000000" |
    .app_state.gov.params.min_deposit |= (if type == "array" then map(.denom = "uprsm") else . end) |
    .app_state.gov.params.expedited_min_deposit |= (if type == "array" then map(.denom = "uprsm") else . end)
  ' "$home_dir/config/genesis.json" > "$home_dir/config/genesis.json.tmp"
  mv "$home_dir/config/genesis.json.tmp" "$home_dir/config/genesis.json"
  validator=$(prismad keys show validator -a --keyring-backend test --keyring-dir "$home_dir" --home "$home_dir")
  prismad genesis add-genesis-account "$validator" 100000000000uprsm --home "$home_dir"
  prismad genesis gentx validator 1000000000uprsm --chain-id "$chain_id" --keyring-backend test --keyring-dir "$home_dir" --home "$home_dir"
  prismad genesis collect-gentxs --home "$home_dir"
  touch "$home_dir/.prisma-genesis-ready"
fi

# The local devnet keeps every key in the test keyring. Autocli-generated
# commands read the keyring backend from client.toml and do not re-read the
# --keyring-backend flag, so pin the backend here or --from fails to find
# local keys.
if [ -f "$home_dir/config/client.toml" ]; then
  sed -i 's/^keyring-backend = .*/keyring-backend = "test"/' "$home_dir/config/client.toml"
fi

# Autocli commands resolve the keyring against the default node home
# (/root/.prisma) unless --home is explicitly re-read; point the default
# home at the volume so every command finds the test keyring.
if [ ! -e /root/.prisma ]; then
  ln -s "$home_dir" /root/.prisma
fi

exec prismad start --home "$home_dir" --minimum-gas-prices 0uprsm --pruning default \
  --rpc.laddr tcp://0.0.0.0:26657 --grpc.address 0.0.0.0:9090
