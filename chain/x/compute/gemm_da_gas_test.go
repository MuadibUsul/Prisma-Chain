package compute

// DA gas measurement: every DA_REPLICA_V1 handler runs against a real gas
// meter and the consumed units are recorded per phase. The JSON artifact
// is written when PRISMA_DA_GAS_OUT is set. Gas units are consensus
// resource accounting (GEMMGasV1 extension), never CPU nanoseconds.

import (
	"encoding/json"
	"os"
	"testing"
	"time"

	storetypes "cosmossdk.io/store/types"
	"prismachain/chain/x/compute/types"
	"prismachain/compute/gemmv1"
)

func TestDAGasMeasurement(t *testing.T) {
	const m, n, k = 64, 64, 64
	meter := storetypes.NewGasMeter(1 << 62)
	h, usedGas := meteredHarness(t, meter)
	type record struct {
		Phase      string  `json:"phase"`
		Gas        uint64  `json:"gas_units"`
		WallMs     float64 `json:"wall_ms"`
		WitnessLen int     `json:"witness_bytes,omitempty"`
	}
	var records []record
	phase := func(name string, fn func()) {
		before := usedGas()
		start := time.Now()
		fn()
		records = append(records, record{Phase: name, Gas: usedGas() - before,
			WallMs: float64(time.Since(start).Microseconds()) / 1000.0})
	}

	taskID, a, b, c := h.postGEMM(m, n, k, 77)
	h.acceptAndSubmit(taskID, a, b, c, m, n, k, false)
	task, err := h.keeper.GetGEMMTask(h.ctx, taskID)
	if err != nil {
		t.Fatal(err)
	}
	until := daAttestationUntil(task)

	phase("RegisterDAProvider", func() {
		h.registerDAProvider(h.provA, h.provAKeys)
	})
	phase("SubmitDAAttestation", func() {
		h.attestDA(taskID, h.provA, h.provAKeys, until)
	})
	phase("RegisterDAProvider2", func() {
		h.registerDAProvider(h.provB, h.provBKeys)
	})
	phase("SubmitDAAttestation2", func() {
		h.attestDA(taskID, h.provB, h.provBKeys, until)
	})
	var open *types.MsgOpenGEMMDAChallengeResponse
	phase("OpenGEMMDAChallenge", func() {
		open, err = h.msg.OpenGEMMDAChallenge(h.ctx, &types.MsgOpenGEMMDAChallenge{
			Challenger: h.challenger, GemmTaskId: taskID, Provider: h.provA,
			Nonce: []byte("gas-nonce"), Bond: DAPenaltyUprsm,
		})
		if err != nil {
			t.Fatal(err)
		}
	})
	phase("RespondGEMMDAChallenge", func() {
		levels, err := gemmv1.BuildLevels(gemmOutputLeaves(h, taskID, h.lastTiles))
		if err != nil {
			t.Fatal(err)
		}
		rowsC := uint32((task.M + 7) / 8)
		index := open.TileI*rowsC + open.TileJ
		proof, err := gemmv1.ProveLeaf(levels, index)
		if err != nil {
			t.Fatal(err)
		}
		response := &types.MsgRespondGEMMDAChallenge{
			Provider: h.provA, ChallengeId: open.ChallengeId,
			Tile:          h.lastTiles[index].CanonicalBytes(),
			ProofSiblings: proof.Siblings, ProofIndex: proof.Index, ProofCount: proof.Count,
		}
		if _, err := h.msg.RespondGEMMDAChallenge(h.ctx, response); err != nil {
			t.Fatal(err)
		}
	})
	var open2 *types.MsgOpenGEMMDAChallengeResponse
	phase("OpenGEMMDAChallenge2", func() {
		open2, err = h.msg.OpenGEMMDAChallenge(h.ctx, &types.MsgOpenGEMMDAChallenge{
			Challenger: h.requester, GemmTaskId: taskID, Provider: h.provB,
			Nonce: []byte("gas-nonce-2"), Bond: DAPenaltyUprsm,
		})
		if err != nil {
			t.Fatal(err)
		}
	})
	phase("TimeoutGEMMDAChallenge", func() {
		h.advanceHeight(int64(open2.Deadline) + 1)
		if _, err := h.msg.TimeoutGEMMDAChallenge(h.ctx, &types.MsgTimeoutGEMMDAChallenge{
			Actor: h.requester, ChallengeId: open2.ChallengeId,
		}); err != nil {
			t.Fatal(err)
		}
	})
	phase("FailGEMMAvailabilityGate", func() {
		// The task still has a live quorum (provA attestation), so the
		// failure path must refuse; the gas of the refusal is recorded.
		h.advanceHeight(int64(task.ResultSubmittedHeight+task.ChallengeWindow+DAWindowBlocks) + 1)
		if _, err := h.msg.FailGEMMAvailability(h.ctx, &types.MsgFailGEMMAvailability{
			Actor: h.requester, GemmTaskId: taskID,
		}); err == nil {
			t.Fatal("availability failure accepted with a live quorum")
		}
	})

	for _, r := range records {
		t.Logf("%-28s gas=%8d wall=%.2fms", r.Phase, r.Gas, r.WallMs)
	}
	if path := os.Getenv("PRISMA_DA_GAS_OUT"); path != "" {
		payload := map[string]any{
			"schedule":      GemmGasScheduleVersion,
			"shape":         map[string]uint64{"m": m, "n": n, "k": k},
			"records":       records,
			"da_penalty":    DAPenaltyUprsm,
			"da_window":     DAWindowBlocks,
			"challenge_win": DAChallengeBlocks,
			"note": "gas units are consensus resource accounting, not CPU time; wall_ms is reported separately. " +
				"RespondGEMMDAChallenge carries a 256-byte tile plus a bounded Merkle proof.",
		}
		data, err := json.MarshalIndent(payload, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, append(data, '\n'), 0o644); err != nil {
			t.Fatal(err)
		}
		t.Logf("written %s", path)
	}
}
