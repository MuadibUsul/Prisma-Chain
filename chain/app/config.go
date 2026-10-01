package app

import (
	runtimev1 "cosmossdk.io/api/cosmos/app/runtime/v1alpha1"
	appv1 "cosmossdk.io/api/cosmos/app/v1alpha1"
	authv1 "cosmossdk.io/api/cosmos/auth/module/v1"
	bankv1 "cosmossdk.io/api/cosmos/bank/module/v1"
	consensusv1 "cosmossdk.io/api/cosmos/consensus/module/v1"
	distributionv1 "cosmossdk.io/api/cosmos/distribution/module/v1"
	genutilv1 "cosmossdk.io/api/cosmos/genutil/module/v1"
	govv1 "cosmossdk.io/api/cosmos/gov/module/v1"
	mintv1 "cosmossdk.io/api/cosmos/mint/module/v1"
	slashingv1 "cosmossdk.io/api/cosmos/slashing/module/v1"
	stakingv1 "cosmossdk.io/api/cosmos/staking/module/v1"
	txv1 "cosmossdk.io/api/cosmos/tx/config/v1"
	upgradev1 "cosmossdk.io/api/cosmos/upgrade/module/v1"
	"cosmossdk.io/core/appconfig"
	"cosmossdk.io/depinject"
	_ "cosmossdk.io/x/upgrade"

	upgradetypes "cosmossdk.io/x/upgrade/types"
	"github.com/cosmos/cosmos-sdk/runtime"
	"github.com/cosmos/cosmos-sdk/types/module"
	_ "github.com/cosmos/cosmos-sdk/x/auth"
	_ "github.com/cosmos/cosmos-sdk/x/auth/tx/config"
	authtypes "github.com/cosmos/cosmos-sdk/x/auth/types"
	_ "github.com/cosmos/cosmos-sdk/x/bank"
	banktypes "github.com/cosmos/cosmos-sdk/x/bank/types"
	_ "github.com/cosmos/cosmos-sdk/x/consensus"
	consensustypes "github.com/cosmos/cosmos-sdk/x/consensus/types"
	_ "github.com/cosmos/cosmos-sdk/x/distribution"
	distributiontypes "github.com/cosmos/cosmos-sdk/x/distribution/types"
	"github.com/cosmos/cosmos-sdk/x/genutil"
	genutiltypes "github.com/cosmos/cosmos-sdk/x/genutil/types"
	"github.com/cosmos/cosmos-sdk/x/gov"
	govtypes "github.com/cosmos/cosmos-sdk/x/gov/types"
	_ "github.com/cosmos/cosmos-sdk/x/mint"
	minttypes "github.com/cosmos/cosmos-sdk/x/mint/types"
	_ "github.com/cosmos/cosmos-sdk/x/slashing"
	slashingtypes "github.com/cosmos/cosmos-sdk/x/slashing/types"
	_ "github.com/cosmos/cosmos-sdk/x/staking"
	stakingtypes "github.com/cosmos/cosmos-sdk/x/staking/types"
	"prismachain/chain/x/compute"
)

const (
	Name  = "prismad"
	Denom = "uprsm"
)

var moduleOrder = []string{
	authtypes.ModuleName, banktypes.ModuleName, distributiontypes.ModuleName,
	stakingtypes.ModuleName, slashingtypes.ModuleName, govtypes.ModuleName,
	minttypes.ModuleName, genutiltypes.ModuleName, consensustypes.ModuleName,
	upgradetypes.ModuleName, compute.ModuleName,
}

var Config = depinject.Configs(appconfig.Compose(&appv1.Config{Modules: []*appv1.ModuleConfig{
	{Name: runtime.ModuleName, Config: appconfig.WrapAny(&runtimev1.Module{
		AppName:           Name,
		PreBlockers:       []string{upgradetypes.ModuleName, authtypes.ModuleName},
		BeginBlockers:     []string{minttypes.ModuleName, distributiontypes.ModuleName, slashingtypes.ModuleName, stakingtypes.ModuleName},
		EndBlockers:       []string{govtypes.ModuleName, stakingtypes.ModuleName},
		InitGenesis:       moduleOrder,
		ExportGenesis:     moduleOrder,
		OverrideStoreKeys: []*runtimev1.StoreKeyConfig{{ModuleName: authtypes.ModuleName, KvStoreKey: "acc"}},
		SkipStoreKeys:     []string{"tx"},
	})},
	{Name: authtypes.ModuleName, Config: appconfig.WrapAny(&authv1.Module{
		Bech32Prefix: "prsm",
		ModuleAccountPermissions: []*authv1.ModuleAccountPermission{
			{Account: authtypes.FeeCollectorName},
			{Account: distributiontypes.ModuleName},
			{Account: minttypes.ModuleName, Permissions: []string{authtypes.Minter}},
			{Account: stakingtypes.BondedPoolName, Permissions: []string{authtypes.Burner, stakingtypes.ModuleName}},
			{Account: stakingtypes.NotBondedPoolName, Permissions: []string{authtypes.Burner, stakingtypes.ModuleName}},
			{Account: govtypes.ModuleName, Permissions: []string{authtypes.Burner}},
			{Account: compute.ModuleName, Permissions: []string{authtypes.Burner}},
		},
	})},
	{Name: banktypes.ModuleName, Config: appconfig.WrapAny(&bankv1.Module{
		BlockedModuleAccountsOverride: []string{authtypes.FeeCollectorName, distributiontypes.ModuleName, minttypes.ModuleName, stakingtypes.BondedPoolName, stakingtypes.NotBondedPoolName, compute.ModuleName},
	})},
	{Name: stakingtypes.ModuleName, Config: appconfig.WrapAny(&stakingv1.Module{Bech32PrefixValidator: "prsmvaloper", Bech32PrefixConsensus: "prsmvalcons"})},
	{Name: slashingtypes.ModuleName, Config: appconfig.WrapAny(&slashingv1.Module{})},
	{Name: "tx", Config: appconfig.WrapAny(&txv1.Config{})},
	{Name: genutiltypes.ModuleName, Config: appconfig.WrapAny(&genutilv1.Module{})},
	{Name: upgradetypes.ModuleName, Config: appconfig.WrapAny(&upgradev1.Module{})},
	{Name: distributiontypes.ModuleName, Config: appconfig.WrapAny(&distributionv1.Module{})},
	{Name: minttypes.ModuleName, Config: appconfig.WrapAny(&mintv1.Module{})},
	{Name: govtypes.ModuleName, Config: appconfig.WrapAny(&govv1.Module{})},
	{Name: consensustypes.ModuleName, Config: appconfig.WrapAny(&consensusv1.Module{})},
}}), depinject.Supply(map[string]module.AppModuleBasic{
	genutiltypes.ModuleName: genutil.NewAppModuleBasic(genutiltypes.DefaultMessageValidator),
	govtypes.ModuleName:     gov.NewAppModuleBasic(nil),
}))
