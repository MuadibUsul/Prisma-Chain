package gemmv1

// Tensor layout, tiling (including boundary padding) and matrix
// commitments.

import (
	"bytes"
	"testing"
)

func TestCanonicalInt32LayoutPlatformIndependent(t *testing.T) {
	// Row-major big-endian two's complement: known byte patterns.
	vals := []int32{1, -2, 0, 2147483647, -2147483648}
	want := []byte{
		0x00, 0x00, 0x00, 0x01,
		0xff, 0xff, 0xff, 0xfe,
		0x00, 0x00, 0x00, 0x00,
		0x7f, 0xff, 0xff, 0xff,
		0x80, 0x00, 0x00, 0x00,
	}
	if got := Int32sToCanonical(vals); !bytes.Equal(got, want) {
		t.Fatalf("got %x want %x", got, want)
	}
	back, err := CanonicalToInt32s(want)
	if err != nil {
		t.Fatal(err)
	}
	for i := range vals {
		if back[i] != vals[i] {
			t.Fatalf("roundtrip mismatch at %d", i)
		}
	}
}

func TestStateZeroAndRoundtrip(t *testing.T) {
	var s State
	if !s.IsZero() {
		t.Fatal("default state must be zero")
	}
	s[0] = -1
	if s.IsZero() {
		t.Fatal("modified state is not zero")
	}
	back, err := StateFromCanonical(s.CanonicalBytes())
	if err != nil {
		t.Fatal(err)
	}
	if *back != s {
		t.Fatal("state roundtrip mismatch")
	}
	if _, err := StateFromCanonical([]byte{0x01}); err == nil {
		t.Fatal("short state must be rejected")
	}
}

// H: boundary tiles must be zero padded and the canonical MAC count must
// stay M x N x K.
func TestBoundaryTilesAndPadding(t *testing.T) {
	sizes := []struct{ m, n, k uint64 }{
		{5, 4, 12},
		{17, 9, 23},
		{8, 8, 8},
		{1, 1, 1},
	}
	for _, s := range sizes {
		a := GenTestMatrix('A', uint32(s.m*100+s.k), s.m*s.k)
		b := GenTestMatrix('B', uint32(s.n+7), s.k*s.n)
		counts := TileCountsFor(s.m, s.n, s.k)
		if counts.ColsA != RSteps(s.k) {
			t.Fatalf("R mismatch for %v", s)
		}
		for i := uint64(0); i < counts.RowsA; i++ {
			for r := uint64(0); r < counts.ColsA; r++ {
				tile := ExtractATile(a, s.m, s.k, i, r)
				for row := 0; row < TileSize; row++ {
					for col := 0; col < TileSize; col++ {
						gi, gk := i*TileSize+uint64(row), r*TileSize+uint64(col)
						want := int8(0)
						if gi < s.m && gk < s.k {
							want = a[gi*s.k+gk]
						}
						if tile[row*TileSize+col] != want {
							t.Fatalf("A tile padding wrong at %v (%d,%d,%d,%d)", s, i, r, row, col)
						}
					}
				}
			}
		}
		for r := uint64(0); r < counts.RowsB; r++ {
			for j := uint64(0); j < counts.ColsB; j++ {
				tile := ExtractBTile(b, s.k, s.n, r, j)
				for row := 0; row < TileSize; row++ {
					for col := 0; col < TileSize; col++ {
						gk, gn := r*TileSize+uint64(row), j*TileSize+uint64(col)
						want := int8(0)
						if gk < s.k && gn < s.n {
							want = b[gk*s.n+gn]
						}
						if tile[row*TileSize+col] != want {
							t.Fatalf("B tile padding wrong at %v", s)
						}
					}
				}
			}
		}
	}
}

func TestMicroStepAgainstDirectGEMM(t *testing.T) {
	a := GenTestMatrix('A', 7, 64)
	b := GenTestMatrix('B', 8, 64)
	var at, bt [int8TileSize]int8
	copy(at[:], a)
	copy(bt[:], b)
	next := MicroStep(&State{}, at, bt)
	// Direct 8x8x8 accumulation over the same inputs.
	for i := 0; i < TileSize; i++ {
		for j := 0; j < TileSize; j++ {
			var acc int32
			for r := 0; r < TileSize; r++ {
				acc += int32(a[i*8+r]) * int32(b[r*8+j])
			}
			if next[i*8+j] != acc {
				t.Fatalf("micro step mismatch at (%d,%d)", i, j)
			}
		}
	}
}

func TestMatrixRootDeterminismAndTamper(t *testing.T) {
	const m, n, k = 12, 10, 20
	a := GenTestMatrix('A', 99, m*k)
	b := GenTestMatrix('B', 100, k*n)

	ra1, rb1, err := BuildMatrixRoots(a, b, m, n, k)
	if err != nil {
		t.Fatal(err)
	}
	ra2, rb2, err := BuildMatrixRoots(a, b, m, n, k)
	if err != nil {
		t.Fatal(err)
	}
	if ra1 != ra2 || rb1 != rb2 {
		t.Fatal("matrix roots are not deterministic")
	}

	// Flip one input byte: every root must change.
	aTampered := append([]int8(nil), a...)
	if aTampered[3] == 1 {
		aTampered[3] = 2
	} else {
		aTampered[3] = 1
	}
	ra3, rb3, err := BuildMatrixRoots(aTampered, b, m, n, k)
	if err != nil {
		t.Fatal(err)
	}
	if ra3 == ra1 {
		t.Fatal("A root unchanged after tampering one byte")
	}
	if rb3 != rb1 {
		t.Fatal("B root changed although B was untouched")
	}
}

func TestInputTileProofs(t *testing.T) {
	const m, n, k = 9, 11, 18
	a := GenTestMatrix('A', 5, m*k)
	b := GenTestMatrix('B', 6, k*n)
	rootA, rootB, err := BuildMatrixRoots(a, b, m, n, k)
	if err != nil {
		t.Fatal(err)
	}
	counts := TileCountsFor(m, n, k)
	for i := uint64(0); i < counts.RowsA; i++ {
		for r := uint64(0); r < counts.ColsA; r++ {
			p, err := ProveInputTile(MatrixIDA, a, b, m, n, k, i, r)
			if err != nil {
				t.Fatal(err)
			}
			tile := Int8TileBytes(ExtractATile(a, m, k, i, r))
			if !VerifyInputTileProof(rootA, MatrixIDA, uint32(i), uint32(r), tile, p) {
				t.Fatalf("A tile proof (%d,%d) rejected", i, r)
			}
			if !VerifyInputTileProof(rootA, MatrixIDA, uint32(i), uint32(r), tile, p) {
				t.Fatalf("B tile proof under A root accepted")
			}
			// Wrong tile content must be rejected.
			bad := append([]byte(nil), tile...)
			bad[0] ^= 0xff
			if VerifyInputTileProof(rootA, MatrixIDA, uint32(i), uint32(r), bad, p) {
				t.Fatal("tampered tile accepted")
			}
		}
	}
	for r := uint64(0); r < counts.RowsB; r++ {
		for j := uint64(0); j < counts.ColsB; j++ {
			p, err := ProveInputTile(MatrixIDB, a, b, m, n, k, r, j)
			if err != nil {
				t.Fatal(err)
			}
			tile := Int8TileBytes(ExtractBTile(b, k, n, r, j))
			if !VerifyInputTileProof(rootB, MatrixIDB, uint32(r), uint32(j), tile, p) {
				t.Fatalf("B tile proof (%d,%d) rejected", r, j)
			}
		}
	}
}
