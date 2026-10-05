package cmd

import (
	"errors"
	"io"
	"os"
	"path/filepath"

	autocliv1 "cosmossdk.io/api/cosmos/autocli/v1"
	"cosmossdk.io/client/v2/autocli"
	"cosmossdk.io/core/address"
	"cosmossdk.io/core/appmodule"
	"cosmossdk.io/depinject"
	"cosmossdk.io/log"
	cmtcfg "github.com/cometbft/cometbft/config"
	dbm "github.com/cosmos/cosmos-db"
	"github.com/spf13/cobra"

	"github.com/cosmos/cosmos-sdk/client"
	clientcfg "github.com/cosmos/cosmos-sdk/client/config"
	"github.com/cosmos/cosmos-sdk/client/keys"
	"github.com/cosmos/cosmos-sdk/client/rpc"
	"github.com/cosmos/cosmos-sdk/codec"
	codectypes "github.com/cosmos/cosmos-sdk/codec/types"
	"github.com/cosmos/cosmos-sdk/runtime"
	"github.com/cosmos/cosmos-sdk/server"
	servercfg "github.com/cosmos/cosmos-sdk/server/config"
	servertypes "github.com/cosmos/cosmos-sdk/server/types"
	sdk "github.com/cosmos/cosmos-sdk/types"
	"github.com/cosmos/cosmos-sdk/types/module"
	authcmd "github.com/cosmos/cosmos-sdk/x/auth/client/cli"
	authtypes "github.com/cosmos/cosmos-sdk/x/auth/types"
	genutilcli "github.com/cosmos/cosmos-sdk/x/genutil/client/cli"
	"prismachain/chain/app"
	"prismachain/chain/x/compute"
)

const homeName = ".prisma"

func DefaultNodeHome() string {
	userHome, err := os.UserHomeDir()
	if err != nil {
		panic(err)
	}
	return filepath.Join(userHome, homeName)
}

func NewRootCmd() *cobra.Command {
	home := DefaultNodeHome()
	sdkCfg := sdk.GetConfig()
	sdkCfg.SetBech32PrefixForAccount("prsm", "prsmpub")
	sdkCfg.SetBech32PrefixForValidator("prsmvaloper", "prsmvaloperpub")
	sdkCfg.SetBech32PrefixForConsensusNode("prsmvalcons", "prsmvalconspub")
	sdkCfg.Seal()
	var cdc codec.Codec
	var registry codectypes.InterfaceRegistry
	var txConfig client.TxConfig
	var amino *codec.LegacyAmino
	var basics module.BasicManager
	var modules map[string]appmodule.AppModule
	var accountCodec address.Codec
	var validatorCodec runtime.ValidatorAddressCodec
	var consensusCodec runtime.ConsensusAddressCodec
	if err := depinject.Inject(depinject.Configs(app.Config, depinject.Supply(log.NewNopLogger())),
		&cdc, &registry, &txConfig, &amino, &basics, &modules,
		&accountCodec, &validatorCodec, &consensusCodec); err != nil {
		panic(err)
	}
	computeBasic := compute.NewAppModule(compute.Keeper{})
	computeBasic.RegisterInterfaces(registry)
	basics[compute.ModuleName] = computeBasic
	clientCtx := client.Context{}.
		WithCodec(cdc).
		WithInterfaceRegistry(registry).
		WithTxConfig(txConfig).
		WithLegacyAmino(amino).
		WithInput(os.Stdin).
		WithOutput(os.Stdout).
		WithAccountRetriever(authtypes.AccountRetriever{}).
		WithHomeDir(home).
		WithViper("")
	root := &cobra.Command{Use: "prismad", Short: "Prisma Chain node", SilenceUsage: true,
		PersistentPreRunE: func(cmd *cobra.Command, _ []string) error {
			clientCtx = clientCtx.WithCmdContext(cmd.Context())
			var err error
			clientCtx, err = client.ReadPersistentCommandFlags(clientCtx, cmd.Flags())
			if err != nil {
				return err
			}
			clientCtx, err = clientcfg.ReadFromClientConfig(clientCtx)
			if err != nil {
				return err
			}
			if err := client.SetCmdClientContextHandler(clientCtx, cmd); err != nil {
				return err
			}
			cfg := servercfg.DefaultConfig()
			cfg.MinGasPrices = "0uprsm"
			return server.InterceptConfigsPreRunHandler(cmd, servercfg.DefaultConfigTemplate, cfg, cmtcfg.DefaultConfig())
		},
	}
	server.AddCommandsWithStartCmdOptions(root, home, newApp, exportApp, server.StartCmdOptions{})
	root.AddCommand(genutilcli.InitCmd(basics, home), genutilcli.Commands(txConfig, basics, home), server.StatusCommand(), keys.Commands())
	queryCmd := &cobra.Command{Use: "query", Aliases: []string{"q"}, RunE: client.ValidateCmd}
	queryCmd.AddCommand(rpc.WaitTxCmd(), authcmd.QueryTxCmd(), authcmd.QueryTxsByEventsCmd())
	basics.AddQueryCommands(queryCmd)
	root.AddCommand(queryCmd)
	txCmd := &cobra.Command{Use: "tx", RunE: client.ValidateCmd}
	txCmd.AddCommand(authcmd.GetBroadcastCommand(), authcmd.GetEncodeCommand(), authcmd.GetDecodeCommand())
	basics.AddTxCommands(txCmd)
	root.AddCommand(txCmd)
	autoCliOpts := autocli.AppOptions{
		Modules: modules, ModuleOptions: make(map[string]*autocliv1.ModuleOptions),
		AddressCodec: accountCodec, ValidatorAddressCodec: validatorCodec,
		ConsensusAddressCodec: consensusCodec, ClientCtx: clientCtx,
	}
	autoCliOpts.Modules[compute.ModuleName] = computeBasic
	autoCliOpts.ModuleOptions[compute.ModuleName] = computeBasic.AutoCLIOptions()
	autoCliOpts.ClientCtx = clientCtx
	if err := autoCliOpts.EnhanceRootCommand(root); err != nil {
		panic(err)
	}
	prismaVersion := NewVersionCmd()
	root.AddCommand(prismaVersion, NewHealthCmd(), NewRPCProxyCmd())
	// The SDK registers a generic `version` command; replace it so that
	// `prismad version` reports the Prisma build and frozen-protocol
	// identity (B1-01b) instead of the SDK's build info.
	for _, c := range root.Commands() {
		if c.Name() == "version" && c != prismaVersion {
			root.RemoveCommand(c)
		}
	}
	root.SetOut(os.Stdout)
	root.SetErr(os.Stderr)
	return root
}

func newApp(logger log.Logger, db dbm.DB, trace io.Writer, opts servertypes.AppOptions) servertypes.Application {
	baseOptions := server.DefaultBaseappOptions(opts)
	a, err := app.New(logger, db, trace, true, opts, baseOptions...)
	if err != nil {
		panic(err)
	}
	return a
}

func exportApp(log.Logger, dbm.DB, io.Writer, int64, bool, []string, servertypes.AppOptions, []string) (servertypes.ExportedApp, error) {
	return servertypes.ExportedApp{}, errors.New("state export is not yet implemented")
}
