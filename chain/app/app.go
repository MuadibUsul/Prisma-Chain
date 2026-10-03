package app

import (
	"io"

	"cosmossdk.io/depinject"
	"cosmossdk.io/log"
	storetypes "cosmossdk.io/store/types"
	dbm "github.com/cosmos/cosmos-db"

	"github.com/cosmos/cosmos-sdk/baseapp"
	"github.com/cosmos/cosmos-sdk/codec"
	"github.com/cosmos/cosmos-sdk/runtime"
	servertypes "github.com/cosmos/cosmos-sdk/server/types"
	bankkeeper "github.com/cosmos/cosmos-sdk/x/bank/keeper"
	"prismachain/chain/x/compute"
)

// App is a Cosmos SDK application. CometBFT executes only deterministic SDK
// modules; all GPU work and full trace replay happen outside consensus.
type App struct {
	*runtime.App
	ComputeKeeper compute.Keeper
	Codec         codec.Codec
}

func New(logger log.Logger, db dbm.DB, trace io.Writer, loadLatest bool, opts servertypes.AppOptions, baseOptions ...func(*baseapp.BaseApp)) (*App, error) {
	var builder *runtime.AppBuilder
	var bank bankkeeper.BaseKeeper
	var cdc codec.Codec
	if err := depinject.Inject(depinject.Configs(Config, depinject.Supply(opts, logger)), &builder, &bank, &cdc); err != nil {
		return nil, err
	}
	app := &App{App: builder.Build(db, trace, baseOptions...), Codec: cdc}
	key := storetypes.NewKVStoreKey(compute.ModuleName)
	if err := app.RegisterStores(key); err != nil {
		return nil, err
	}
	app.ComputeKeeper = compute.NewKeeper(key, bank)
	if err := app.RegisterModules(compute.NewAppModule(app.ComputeKeeper)); err != nil {
		return nil, err
	}
	if err := app.Load(loadLatest); err != nil {
		return nil, err
	}
	// Development-only proposer censorship harness; no-op unless the
	// PRISMA_DEV_CENSOR_GEMM_CHALLENGES environment variable is set.
	maybeEnableDevCensorship(app)
	return app, nil
}
