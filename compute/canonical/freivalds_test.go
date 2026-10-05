package canonical

import (
	"testing"
)

func refWide(t *testing.T, a, w []int64, m, n, k int, tb, wA13 bool) []int64 {
	t.Helper()
	c, err := ReferenceWideGEMM(a, w, m, n, k, tb, wA13)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func TestFreivaldsDetectsTampering(t *testing.T) {
	// k=96, n=64, m=16; deterministic inputs
	a := make([]int64, 16*96)
	w := make([]int64, 96*64)
	for i := range a {
		a[i] = int64((i*37)%4096) - 2048
	}
	for i := range w {
		w[i] = int64((i*53)%1024) - 512
	}
	c := refWide(t, a, w, 16, 64, 96, false, false)
	bits := make([]uint8, 64)
	for j := range bits {
		bits[j] = uint8(j % 2)
	}
	ok, err := FreivaldsWideCheck(a, w, c, 16, 64, 96, false, false, bits)
	if err != nil || !ok {
		t.Fatalf("honest check failed: ok=%v err=%v", ok, err)
	}
	bad := append([]int64(nil), c...)
	bad[5*64+17]++
	detected := 0
	for seed := 0; seed < 40; seed++ {
		for j := range bits {
			bits[j] = uint8((seed>>(j%8) + j) % 2)
		}
		ok, err := FreivaldsWideCheck(a, w, bad, 16, 64, 96, false, false, bits)
		if err != nil {
			t.Fatal(err)
		}
		if !ok {
			detected++
		}
	}
	if detected == 0 {
		t.Fatal("tampering never detected across 40 rounds")
	}
	t.Logf("tampering detected in %d/40 rounds", detected)
}

func TestFreivaldsInt64BoundsForRealShapes(t *testing.T) {
	// every real shape must admit the check with < 62 signed bits
	for _, s := range [][3]int{{16, 128, 1024}, {16, 1024, 128}, {16, 16, 128}, {16, 128, 16}, {16, 3072, 1024}, {16, 1024, 3072}} {
		m, n, k := s[0], s[1], s[2]
		if b := wideFreivaldsBoundBits(13, 10, m, n, k); b > 62 {
			t.Fatalf("shape %dx%dx%d needs %d bits", m, n, k, b)
		}
		if b := wideFreivaldsBoundBits(13, 13, m, n, k); b > 62 {
			t.Fatalf("A13 right shape %dx%dx%d needs %d bits", m, n, k, b)
		}
	}
}

func TestWideDisputeChainAndBisection(t *testing.T) {
	// K = 24 -> 3 steps (+1 initial state)
	m, n, k := 8, 8, 24
	a := make([]int64, m*k)
	w := make([]int64, k*n)
	for i := range a {
		a[i] = int64(i%7) - 3
	}
	for i := range w {
		w[i] = int64(i%5) - 2
	}
	steps := WideStepCount(k)
	states := make([]WideTileState, steps)
	cur := WideTileState{}
	for r := 0; r < steps-1; r++ {
		at, err := WideATile(a, m, k, 0, r)
		if err != nil {
			t.Fatal(err)
		}
		wt, err := WideWTile(w, n, k, 0, r, false)
		if err != nil {
			t.Fatal(err)
		}
		cur = WideTileStep(cur, at, wt)
		states[r+1] = cur
	}
	// the final state equals the direct reference accumulation
	ref := refWide(t, a, w, m, n, k, false, false)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if states[steps-1][i*8+j] != ref[i*n+j] {
				t.Fatalf("trace state mismatch at (%d,%d)", i, j)
			}
		}
	}
	rootW, err := WideTraceRootV1(0, 0, states)
	if err != nil {
		t.Fatal(err)
	}
	// challenger tampers step 2's state; bisection finds the first divergent step
	bad := append([]WideTileState(nil), states...)
	bad[2][3]++
	rootC, err := WideTraceRootV1(0, 0, bad)
	if err != nil {
		t.Fatal(err)
	}
	if rootW == rootC {
		t.Fatal("tampered trace has the same root")
	}
	low, high := 0, steps-1
	for high-low > 1 {
		mid := (low + high) / 2
		if bad[mid] == states[mid] {
			low = mid
		} else {
			high = mid
		}
	}
	if high != 2 {
		t.Fatalf("bisection found step %d, want 2", high)
	}
	at, _ := WideATile(a, m, k, 0, low)
	wt, _ := WideWTile(w, n, k, 0, low, false)
	// the WORKER is the cheater here (its trace is the tampered chain); the
	// challenger holds the honest chain
	out, got, err := WideArbitrate(0, 0, low, states[low], at, wt, bad[high], states[high], true, true)
	if err != nil {
		t.Fatal(err)
	}
	if out != WideChallengerWins {
		t.Fatalf("outcome %v, want challenger wins", out)
	}
	if got != states[high] {
		t.Fatal("recomputed state does not match the honest chain")
	}
	// a false challenge (both honest) yields worker wins when the challenger
	// claims a wrong state
	fakeNext := states[high]
	fakeNext[0] ^= 1
	out2, _, err := WideArbitrate(0, 0, low, states[low], at, wt, states[high], fakeNext, true, true)
	if err != nil || out2 != WideWorkerWins {
		t.Fatalf("false challenge outcome %v err %v, want worker wins", out2, err)
	}
	// missing operand proofs are refused before any arithmetic
	if _, _, err := WideArbitrate(0, 0, low, states[low], at, wt, states[high], bad[high], false, true); err == nil {
		t.Fatal("arbitration accepted missing A proof")
	}
}

func TestWide512Bound(t *testing.T) {
	if err := WideDisputeEvidence(0, 4); err != nil {
		t.Fatal(err)
	}
	if err := WideDisputeEvidence(4, 4); err == nil {
		t.Fatal("out-of-range step accepted")
	}
	if 8*4096*512 >= 1<<62 {
		t.Fatal("512-bound arithmetic broken")
	}
}
