package compute

// GEMMGasV1: the deterministic, versioned consensus resource accounting
// schedule for the GEMM settlement path. Cosmos SDK gas does not follow
// the CPU cost of a 512-MAC micro-step by itself, so every GEMM handler
// charges explicit, bounded, shape-derived units BEFORE doing the work.
// These units are consensus resource accounting, not CPU nanoseconds.

import (
	"context"
	"fmt"

	sdk "github.com/cosmos/cosmos-sdk/types"
)

const (
	// GemmGasScheduleVersion names the gas schedule for documentation and
	// tests; parameters must not drift under the same version.
	GemmGasScheduleVersion = "GEMMGasV1"

	GasGEMMTxBase          uint64 = 10_000
	GasPerGEMMHash         uint64 = 30
	GasPerGEMMProofSibling uint64 = 100
	GasPerGEMMStateByte    uint64 = 2
	GasGEMMArbitration     uint64 = 8_000
	GasPerGEMMStoreWrite   uint64 = 500
)

// consumeGEMMGas charges named units on the transaction gas meter.
func (s msgServer) consumeGEMMGas(ctx context.Context, name string, units uint64) {
	sdk.UnwrapSDKContext(ctx).GasMeter().ConsumeGas(units, fmt.Sprintf("%s: %s", GemmGasScheduleVersion, name))
}

// gasForProofs charges for Merkle proof verification before any hashing:
// one hash per sibling plus one leaf/root hash each, bounded by the
// shape-derived sibling count that the callers check first.
func (s msgServer) gasForProofs(ctx context.Context, label string, proofCount int, siblings int) {
	units := GasPerGEMMProofSibling*uint64(siblings) + GasPerGEMMHash*uint64(proofCount*(siblings+2))
	s.consumeGEMMGas(ctx, label, units)
}

// gasForStates charges for canonical state decoding and hashing.
func (s msgServer) gasForStates(ctx context.Context, label string, stateBytes int) {
	s.consumeGEMMGas(ctx, label, GasPerGEMMStateByte*uint64(stateBytes)+GasPerGEMMHash*uint64(stateBytes/256+1))
}
