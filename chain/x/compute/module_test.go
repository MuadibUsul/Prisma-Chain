package compute

import (
	"encoding/json"
	"testing"

	"cosmossdk.io/log"
	"cosmossdk.io/store/metrics"
	"cosmossdk.io/store/rootmulti"
	storetypes "cosmossdk.io/store/types"
	cmtproto "github.com/cometbft/cometbft/proto/tendermint/types"
	dbm "github.com/cosmos/cosmos-db"
	sdk "github.com/cosmos/cosmos-sdk/types"
)

func TestGenesisStoreCanQueryAndReloadEmptyTaskChain(t *testing.T) {
	db := dbm.NewMemDB()
	key := storetypes.NewKVStoreKey(ModuleName)
	newStore := func() *rootmulti.Store {
		store := rootmulti.NewStore(db, log.NewNopLogger(), metrics.NewNoOpMetrics())
		store.MountStoreWithDB(key, storetypes.StoreTypeIAVL, nil)
		return store
	}
	store := newStore()
	if err := store.LoadLatestVersion(); err != nil {
		t.Fatal(err)
	}
	ctx := sdk.NewContext(store, cmtproto.Header{Height: 1}, false, log.NewNopLogger())
	NewAppModule(NewKeeper(key, nil)).InitGenesis(ctx, nil, json.RawMessage(`{}`))
	for height := int64(1); height <= 2; height++ {
		store.SetCommitHeader(cmtproto.Header{Height: height})
		if got := store.Commit().Version; got != height {
			t.Fatalf("height %d committed as %d", height, got)
		}
		queryStore, err := store.CacheMultiStoreWithVersion(height)
		if err != nil {
			t.Fatalf("height %d cannot be queried: %v", height, err)
		}
		if got := queryStore.GetKVStore(key).Get([]byte("schema")); len(got) != 1 || got[0] != 1 {
			t.Fatalf("height %d lacks compute schema marker", height)
		}
	}
	restarted := newStore()
	if err := restarted.LoadLatestVersion(); err != nil {
		t.Fatalf("restart: %v", err)
	}
	if got := restarted.LatestVersion(); got != 2 {
		t.Fatalf("restarted at height %d", got)
	}
	if _, err := restarted.CacheMultiStoreWithVersion(1); err != nil {
		t.Fatalf("historical query after restart: %v", err)
	}
}
