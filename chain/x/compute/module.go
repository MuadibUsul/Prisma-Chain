package compute

import (
	autocliv1 "cosmossdk.io/api/cosmos/autocli/v1"
	"encoding/json"

	abci "github.com/cometbft/cometbft/abci/types"
	"github.com/cosmos/cosmos-sdk/client"
	"github.com/cosmos/cosmos-sdk/codec"
	codectypes "github.com/cosmos/cosmos-sdk/codec/types"
	sdk "github.com/cosmos/cosmos-sdk/types"
	"github.com/cosmos/cosmos-sdk/types/module"
	"github.com/cosmos/cosmos-sdk/types/msgservice"
	"github.com/grpc-ecosystem/grpc-gateway/runtime"
	"prismachain/chain/x/compute/types"
)

type AppModule struct{ keeper Keeper }

func NewAppModule(keeper Keeper) AppModule { return AppModule{keeper: keeper} }

func (AppModule) IsAppModule()                                                {}
func (AppModule) IsOnePerModuleType()                                         {}
func (AppModule) Name() string                                                { return ModuleName }
func (AppModule) RegisterLegacyAminoCodec(*codec.LegacyAmino)                 {}
func (AppModule) RegisterGRPCGatewayRoutes(client.Context, *runtime.ServeMux) {}
func (AppModule) RegisterInterfaces(registry codectypes.InterfaceRegistry) {
	msgservice.RegisterMsgServiceDesc(registry, &types.Msg_serviceDesc)
}
func (m AppModule) RegisterServices(cfg module.Configurator) {
	types.RegisterMsgServer(cfg.MsgServer(), m.keeper.MsgServer())
	types.RegisterQueryServer(cfg.QueryServer(), m.keeper.QueryServer())
}
func (AppModule) DefaultGenesis(codec.JSONCodec) json.RawMessage { return json.RawMessage(`{}`) }
func (AppModule) ValidateGenesis(codec.JSONCodec, client.TxEncodingConfig, json.RawMessage) error {
	return nil
}

// An empty IAVL tree can be listed in root commit metadata without a loadable
// version. Keep a schema marker from genesis so even a chain with no tasks can
// serve historical queries and restart from its latest block.
func (m AppModule) InitGenesis(ctx sdk.Context, _ codec.JSONCodec, _ json.RawMessage) []abci.ValidatorUpdate {
	m.keeper.store(ctx).Set([]byte("schema"), []byte{1})
	return nil
}
func (AppModule) ExportGenesis(sdk.Context, codec.JSONCodec) json.RawMessage {
	return json.RawMessage(`{}`)
}
func (AppModule) ConsensusVersion() uint64 { return 1 }

func (AppModule) AutoCLIOptions() *autocliv1.ModuleOptions {
	return &autocliv1.ModuleOptions{
		Tx:    &autocliv1.ServiceCommandDescriptor{Service: types.Msg_serviceDesc.ServiceName},
		Query: &autocliv1.ServiceCommandDescriptor{Service: types.Query_serviceDesc.ServiceName},
	}
}
