package app

// Development-only proposer censorship harness.
//
// Phase E needs to prove that one proposer that omits a valid GEMM
// challenge transaction from its proposal cannot permanently suppress it.
// The honest way to model that is proposer omission: the transaction stays
// legal for every validator (CheckTx, ProcessProposal and DeliverTx still
// accept it), but the censoring proposer simply does not include it in the
// block it proposes.
//
// This hook is OFF by default. Without PRISMA_DEV_CENSOR_GEMM_CHALLENGES it
// changes nothing. It is an adversarial devnet test harness, never a
// production feature, and it must never reject or invalidate a transaction.

import (
	"fmt"
	"os"

	abci "github.com/cometbft/cometbft/abci/types"
	"github.com/cosmos/cosmos-sdk/baseapp"
	"github.com/cosmos/cosmos-sdk/client"
	sdk "github.com/cosmos/cosmos-sdk/types"
	authtx "github.com/cosmos/cosmos-sdk/x/auth/tx"

	"prismachain/chain/x/compute/types"
)

// txConfig builds the transaction codec for the harness filter.
func (app *App) txConfig() client.TxConfig {
	return authtx.NewTxConfig(app.Codec, authtx.DefaultSignModes)
}

// maybeEnableDevCensorship installs the devnet-only proposer filter when
// the environment asks for it. mode:
//
//	"1" or "challenge"  -> omit MsgOpenGEMMChallenge
//	"all"               -> omit challenge, trace, midpoint and arbitration
//	"" or "0"           -> disabled (default)
func maybeEnableDevCensorship(app *App) {
	mode := os.Getenv("PRISMA_DEV_CENSOR_GEMM_CHALLENGES")
	if mode == "" || mode == "0" {
		return
	}
	omitTrace := mode == "all"
	app.SetPrepareProposal(func(ctx sdk.Context, req *abci.RequestPrepareProposal) (*abci.ResponsePrepareProposal, error) {
		// Rebuild the SDK default proposal from the live mempool, then drop
		// the censored messages. Validity rules are untouched.
		base := baseapp.NewDefaultProposalHandler(app.Mempool(), app).PrepareProposalHandler()
		resp, err := base(ctx, req)
		if err != nil {
			return nil, err
		}
		txConfig := app.txConfig()
		if txConfig == nil {
			return resp, nil
		}
		filtered := make([][]byte, 0, len(resp.Txs))
		for _, raw := range resp.Txs {
			if devCensorOmits(txConfig, raw, omitTrace) {
				continue
			}
			filtered = append(filtered, raw)
		}
		resp.Txs = filtered
		return resp, nil
	})
	fmt.Fprintf(os.Stderr, "DEVELOPMENT-ONLY proposer censorship harness active (PRISMA_DEV_CENSOR_GEMM_CHALLENGES=%s)\n", mode)
}

// devCensorOmits reports whether the harness would drop this transaction.
func devCensorOmits(txConfig client.TxConfig, raw []byte, omitTrace bool) bool {
	tx, err := txConfig.TxDecoder()(raw)
	if err != nil {
		return false // undecodable transactions keep their normal fate
	}
	for _, msg := range tx.GetMsgs() {
		switch msg.(type) {
		case *types.MsgOpenGEMMChallenge:
			return true
		case *types.MsgCommitGEMMTrace, *types.MsgSubmitGEMMMidState, *types.MsgArbitrateGEMM:
			if omitTrace {
				return true
			}
		}
	}
	return false
}
