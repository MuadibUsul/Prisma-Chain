package verify

// Exact bad-row -> bad-column -> bad-tile localization. After a Freivalds
// mismatch, the challenger picks one detected bad row, recomputes exactly
// that row (O(KN)), and derives a concrete disputed 8x8 tile. The
// probabilistic evidence only GUIDES this search; the resulting tile plus
// its Merkle proof and the exact expected tile are what enter the v0.1.1
// ChallengeOpen, whose deterministic dispute remains the sole final proof.

import (
	"errors"
)

// ReferenceRowGEMM computes the exact expected output row:
// C[row,:] = A[row,:] x B with int8 inputs and an int32 accumulator, under
// the task admission bound K <= MaxSafeK. Complexity O(KN). It never
// touches the rest of C.
func ReferenceRowGEMM(a []int8, row uint64, b []int8, m, n, k uint64) ([]int32, error) {
	if m == 0 || n == 0 || k == 0 || row >= m {
		return nil, errors.New("verify: row out of range")
	}
	if uint64(len(a)) != m*k || uint64(len(b)) != k*n {
		return nil, errors.New("verify: input shape mismatch")
	}
	out := make([]int32, n)
	ai := a[row*k : (row+1)*k]
	for t := uint64(0); t < k; t++ {
		av := int32(ai[t])
		if av == 0 {
			continue
		}
		bt := b[t*n : (t+1)*n]
		for j := uint64(0); j < n; j++ {
			out[j] += av * int32(bt[j])
		}
	}
	return out, nil
}

// LocateBadColumns returns the columns of one output row where the worker
// matrix differs from the exact recomputation.
func LocateBadColumns(workerRow, expectedRow []int32) []uint64 {
	var cols []uint64
	for j := range workerRow {
		if workerRow[j] != expectedRow[j] {
			cols = append(cols, uint64(j))
		}
	}
	return cols
}

// TileForElement maps one output element to its 8x8 tile coordinates.
func TileForElement(row, col uint64) (tileI, tileJ uint64) {
	return row / 8, col / 8
}

// Localization is the deterministic bridge from a Freivalds mismatch to a
// v0.1.1 ChallengeOpen candidate.
type Localization struct {
	Row         uint64
	Columns     []uint64
	TileI       uint64
	TileJ       uint64
	ExpectedRow []int32
}

// LocalizeFromRows takes one detected bad row, recomputes it exactly and
// returns the first differing column's tile coordinates. found is false
// when the row actually matches (which means the Freivalds round was
// inconsistent with C and the challenger must not open a challenge on it).
func LocalizeFromRows(a, b []int8, c []int32, m, n, k uint64, rows []uint64) (*Localization, bool, error) {
	if len(rows) == 0 {
		return nil, false, errors.New("verify: no residual rows to localize")
	}
	for _, row := range rows {
		expected, err := ReferenceRowGEMM(a, row, b, m, n, k)
		if err != nil {
			return nil, false, err
		}
		workerRow := c[row*n : (row+1)*n]
		cols := LocateBadColumns(workerRow, expected)
		if len(cols) == 0 {
			continue // residual row was a false pointer; try the next one
		}
		tileI, tileJ := TileForElement(row, cols[0])
		return &Localization{
			Row:         row,
			Columns:     cols,
			TileI:       tileI,
			TileJ:       tileJ,
			ExpectedRow: expected,
		}, true, nil
	}
	return nil, false, nil
}
