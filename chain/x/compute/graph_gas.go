package compute

// GraphGasV1: the explicit gas schedule of CANONICAL_GRAPH_V1 settlement.
// Every unit below is charged through the transaction gas meter BEFORE the
// work it prices, so a transaction that exceeds its gas limit aborts
// deterministically. The schedule is versioned and frozen; measured
// devnet numbers live in docs/phase-f-gas-results.json.

import (
	"context"
	"fmt"

	sdk "github.com/cosmos/cosmos-sdk/types"
	"prismachain/compute/canonical"
)

// GraphGasScheduleVersion names the frozen schedule.
const GraphGasScheduleVersion = "GraphGasV1"

const (
	// Transaction base cost of any graph message.
	GasGraphTxBase uint64 = 20_000
	// Descriptor decode + validate per node/input.
	GasGraphNodeDecode  uint64 = 4_000
	GasGraphInputDecode uint64 = 2_000
	// Per canonical work-vector unit above the free allowance. Fusion
	// pricing stays an open economic question (no universal CWU), so the
	// task fee is the agreed flat fee; gas only bounds execution.
	GasGraphWorkUnit      uint64 = 1
	GasGraphFreeWorkUnits uint64 = 200_000
	// Hashing and proof verification.
	GasGraphHash         uint64 = 1_000
	GasGraphProofSibling uint64 = 400
	GasGraphEvidence     uint64 = 800
	GasGraphStateLeaf    uint64 = 600
	// Matching an arbitration recompute (bounded by the operator contract).
	GasGraphArbiterUnit uint64 = 4
	// Receipt derivation, storage and settlement.
	GasGraphReceipt uint64 = 30_000
	GasGraphStore   uint64 = 2_000
)

// consumeGraphGas charges named units on the transaction gas meter.
func (s msgServer) consumeGraphGas(ctx context.Context, name string, units uint64) {
	sdk.UnwrapSDKContext(ctx).GasMeter().ConsumeGas(units, fmt.Sprintf("%s: %s", GraphGasScheduleVersion, name))
}

// graphWorkGas prices a work vector: every unit above the free allowance.
func graphWorkGas(work canonical.WorkVector) uint64 {
	var total uint64
	for _, pair := range work {
		if pair.Value < 0 {
			continue
		}
		total += uint64(pair.Value)
	}
	if total <= GasGraphFreeWorkUnits {
		return 0
	}
	return (total - GasGraphFreeWorkUnits) * GasGraphWorkUnit
}

// GraphGasScheduleJSON renders the frozen schedule for artifacts and docs.
func GraphGasScheduleJSON() map[string]any {
	return map[string]any{
		"version":            GraphGasScheduleVersion,
		"tx_base":            GasGraphTxBase,
		"node_decode":        GasGraphNodeDecode,
		"input_decode":       GasGraphInputDecode,
		"work_unit":          GasGraphWorkUnit,
		"free_work_units":    GasGraphFreeWorkUnits,
		"hash":               GasGraphHash,
		"proof_sibling":      GasGraphProofSibling,
		"evidence_chunk":     GasGraphEvidence,
		"state_leaf":         GasGraphStateLeaf,
		"arbiter_unit":       GasGraphArbiterUnit,
		"receipt":            GasGraphReceipt,
		"store":              GasGraphStore,
	}
}
