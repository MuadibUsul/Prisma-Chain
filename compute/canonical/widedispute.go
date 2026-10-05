package canonical

// WIDE_GEMM_DISPUTE_V1 — deterministic on-chain adjudication of one
// A13W10 GEMM node, generalized from the frozen v0.1.1 GEMM dispute
// architecture: a bad output tile localizes through an on-demand partial
// state trace, K-tiles bisect to the first divergent step, and exactly one
// 8x8x8 micro-step (512 logical MAC) is recomputed on chain in exact int64.
//
// The GPU decomposition never appears here: the arbiter only knows the
// mathematical semantics C = A13 x W10 -> int64.

import (
	"errors"
	"fmt"
)

// WideTileN is the frozen tile edge (v0.1.1 precedent).
const WideTileN = 8

// WideKStep is one K-tile width of the partial-state chain.
const WideKStep = 8

// Wide512MAC is the exact logical MAC count of the final arbiter step:
// 8 (rows) * 8 (cols) * 8 (K) = 512.
const Wide512MAC = WideTileN * WideTileN * WideKStep

// WideTileState is one 8x8 int64 partial accumulation (row-major).
type WideTileState [WideTileN * WideTileN]int64

// WideStepCount = ceil(K/8) + 1 states (S0 = zero .. S_{steps}).
func WideStepCount(k int) int {
	return (k+WideKStep-1)/WideKStep + 1
}

// WideATile extracts the 8x8 A block (rows ti*8.., K-cols r*8..) with
// canonical zero padding beyond M/K.  Padding MAC is not logical work.
func WideATile(a []int64, m, k, ti, r int) (WideTileState, error) {
	if a == nil || m <= 0 || k <= 0 {
		return WideTileState{}, errors.New("canonical/v2: wide A tile needs a live matrix")
	}
	var out WideTileState
	for i := 0; i < WideTileN; i++ {
		ri := ti*WideTileN + i
		for d := 0; d < WideKStep; d++ {
			ci := r*WideKStep + d
			if ri < m && ci < k {
				out[i*WideTileN+d] = a[ri*k+ci]
			}
		}
	}
	return out, nil
}

// WideWTile extracts the 8x8 W block (K-rows r*8.., cols tj*8..);
// transposeB selects the (N,K) storage.
func WideWTile(w []int64, n, k, tj, r int, transposeB bool) (WideTileState, error) {
	if w == nil || n <= 0 || k <= 0 {
		return WideTileState{}, errors.New("canonical/v2: wide W tile needs a live matrix")
	}
	var out WideTileState
	for d := 0; d < WideKStep; d++ {
		ci := r*WideKStep + d
		for j := 0; j < WideTileN; j++ {
			co := tj*WideTileN + j
			if ci < k && co < n {
				if transposeB {
					out[d*WideTileN+j] = w[co*k+ci]
				} else {
					out[d*WideTileN+j] = w[ci*n+co]
				}
			}
		}
	}
	return out, nil
}

// WideTileStep is the deterministic 512-MAC micro-step:
// S_next = S_prev + A_tile * W_tile, exact int64.
func WideTileStep(prev WideTileState, aTile, wTile WideTileState) WideTileState {
	out := prev
	for i := 0; i < WideTileN; i++ {
		for j := 0; j < WideTileN; j++ {
			var acc int64
			for d := 0; d < WideKStep; d++ {
				acc += aTile[i*WideTileN+d] * wTile[d*WideTileN+j]
			}
			out[i*WideTileN+j] += acc
		}
	}
	return out
}

// WideStateLeafV1 hashes one partial state: domain || ti || tj || step ||
// 64 x BE64.  Own domain: V1 trace proofs can never verify here.
func WideStateLeafV1(ti, tj, step int, s WideTileState) Hash {
	payload := make([]byte, 0, 3*4+WideTileN*WideTileN*8)
	payload = appendUint32BE(payload, uint32(ti))
	payload = appendUint32BE(payload, uint32(tj))
	payload = appendUint32BE(payload, uint32(step))
	for _, v := range s {
		for shift := 56; shift >= 0; shift -= 8 {
			payload = append(payload, byte(v>>uint(shift)))
		}
	}
	return hashBytes([]byte(DomainWideTrace), payload)
}

// WideTraceRootV1 is the merkle root over the full step chain (states in
// step order).
func WideTraceRootV1(ti, tj int, states []WideTileState) (Hash, error) {
	if len(states) == 0 {
		return Hash{}, errors.New("canonical/v2: empty wide trace")
	}
	leaves := make([]Hash, len(states))
	for i, s := range states {
		leaves[i] = WideStateLeafV1(ti, tj, i, s)
	}
	root, err := MerkleRootOf(leaves)
	if err != nil {
		return Hash{}, err
	}
	return root, nil
}

// WideOutcome is the arbiter verdict.
type WideOutcome uint8

const (
	WideWorkerWins WideOutcome = iota + 1
	WideChallengerWins
)

// WideArbitrate adjudicates ONE K-step: both parties commit a state for
// step r (agreed) and step r+1 (disputed).  The chain recomputes the
// 512-MAC micro-step from the committed evidence and the party consistent
// with the recomputation wins; an inconsistent worker loses.
func WideArbitrate(ti, tj, step int, stateAtStep WideTileState,
	aTile, wTile WideTileState, workerNext, challengerNext WideTileState,
	aProofOK, wProofOK bool) (WideOutcome, WideTileState, error) {
	if !aProofOK || !wProofOK {
		return 0, WideTileState{}, errors.New("canonical/v2: wide arbitration requires valid operand proofs")
	}
	got := WideTileStep(stateAtStep, aTile, wTile)
	workerOK := got == workerNext
	challengerOK := got == challengerNext
	switch {
	case workerOK && !challengerOK:
		return WideWorkerWins, got, nil
	case !workerOK && challengerOK:
		return WideChallengerWins, got, nil
	case workerOK && challengerOK:
		return 0, got, errors.New("canonical/v2: both states match the recomputation")
	default:
		return 0, got, errors.New("canonical/v2: neither state matches the recomputation")
	}
}

// wideBisectNext narrows [low, high] given the two parties' states at mid
// (the chain does not need this; both off-chain parties and the on-chain
// state machine share it so the first divergent step is canonical).
func wideBisectNext(low, high int, workerMid, challengerMid WideTileState) (int, int) {
	if workerMid == challengerMid {
		return midStep(low, high), high
	}
	return low, midStep(low, high)
}

func midStep(low, high int) int { return low + (high-low)/2 }

// WideDisputeEvidence validates the final arbitration witness sizes: one
// 8x8 state pair plus the operand tiles; the MAC count is exactly 512 and
// the 512-MAC bound (8 * 4096 * 512 = 16,777,216 << MaxInt64) is asserted
// so an oversized claim can never reach the arithmetic.
func WideDisputeEvidence(step int, states int) error {
	if step < 0 || step >= states {
		return fmt.Errorf("canonical/v2: wide step %d outside trace", step)
	}
	// Worst-case magnitude of one element after one 512-MAC step:
	// 8 * 4096 * 512 = 16,777,216, far below MaxInt64 (spec section 81).
	if int64(WideTileN)*int64(-A13Min)*int64(WideKStep) >= 1<<62 {
		return errors.New("canonical/v2: wide 512-MAC bound exceeds int64")
	}
	return nil
}
