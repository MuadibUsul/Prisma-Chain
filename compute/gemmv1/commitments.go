package gemmv1

import "errors"

// Matrix commitment: the Merkle roots over the 8x8 zero-padded int8 tile
// grids of A and B. A and B use distinct matrix identifiers in their leaves.

// BuildMatrixRoots commits A (M x K) and B (K x N). Tiles are enumerated in
// row-major tile order; boundary tiles are zero padded.
func BuildMatrixRoots(matrixA, matrixB []int8, m, n, k uint64) (Hash, Hash, error) {
	if err := CheckInputSizes(m, n, k, matrixA, matrixB); err != nil {
		return Hash{}, Hash{}, err
	}
	counts := TileCountsFor(m, n, k)

	leavesA := make([]Hash, counts.RowsA*counts.ColsA)
	for i := uint64(0); i < counts.RowsA; i++ {
		for r := uint64(0); r < counts.ColsA; r++ {
			leavesA[i*counts.ColsA+r] = LeafInputTile(MatrixIDA, uint32(i), uint32(r), Int8TileBytes(ExtractATile(matrixA, m, k, i, r)))
		}
	}
	leavesB := make([]Hash, counts.RowsB*counts.ColsB)
	for r := uint64(0); r < counts.RowsB; r++ {
		for j := uint64(0); j < counts.ColsB; j++ {
			leavesB[r*counts.ColsB+j] = LeafInputTile(MatrixIDB, uint32(r), uint32(j), Int8TileBytes(ExtractBTile(matrixB, k, n, r, j)))
		}
	}
	rootA, err := MerkleRoot(leavesA)
	if err != nil {
		return Hash{}, Hash{}, err
	}
	rootB, err := MerkleRoot(leavesB)
	if err != nil {
		return Hash{}, Hash{}, err
	}
	return rootA, rootB, nil
}

// VerifyMatrixRoots recomputes both matrix roots of t from raw matrices and
// compares them with the committed roots in the descriptor.
func VerifyMatrixRoots(t *TaskDescriptor, matrixA, matrixB []int8) error {
	rootA, rootB, err := BuildMatrixRoots(matrixA, matrixB, t.M, t.N, t.K)
	if err != nil {
		return err
	}
	if !equalBytes(rootA[:], t.MatrixARoot) {
		return errors.New("gemmv1: matrix A does not match matrix_a_root")
	}
	if !equalBytes(rootB[:], t.MatrixBRoot) {
		return errors.New("gemmv1: matrix B does not match matrix_b_root")
	}
	return nil
}

// ProveInputTile builds an inclusion proof for one A or B tile against the
// recomputed tile tree. matrixID selects A or B.
func ProveInputTile(matrixID byte, matrixA, matrixB []int8, m, n, k uint64, row, col uint64) (InputTileProof, error) {
	var (
		leaves []Hash
		counts TileCounts
	)
	switch matrixID {
	case MatrixIDA:
		counts = TileCountsFor(m, n, k)
		leaves = make([]Hash, counts.RowsA*counts.ColsA)
		for i := uint64(0); i < counts.RowsA; i++ {
			for r := uint64(0); r < counts.ColsA; r++ {
				leaves[i*counts.ColsA+r] = LeafInputTile(MatrixIDA, uint32(i), uint32(r), Int8TileBytes(ExtractATile(matrixA, m, k, i, r)))
			}
		}
	case MatrixIDB:
		counts = TileCountsFor(m, n, k)
		leaves = make([]Hash, counts.RowsB*counts.ColsB)
		for r := uint64(0); r < counts.RowsB; r++ {
			for j := uint64(0); j < counts.ColsB; j++ {
				leaves[r*counts.ColsB+j] = LeafInputTile(MatrixIDB, uint32(r), uint32(j), Int8TileBytes(ExtractBTile(matrixB, k, n, r, j)))
			}
		}
	default:
		return InputTileProof{}, errors.New("gemmv1: unknown matrix id")
	}
	levels, err := buildLevels(leaves)
	if err != nil {
		return InputTileProof{}, err
	}
	proof, err := ProveLeaf(levels, uint32(row*colsOf(counts, matrixID)+col))
	if err != nil {
		return InputTileProof{}, err
	}
	return InputTileProof{
		MatrixID: uint32(matrixID),
		TileRow:  uint32(row),
		TileCol:  uint32(col),
		TileCols: uint32(colsOf(counts, matrixID)),
		Count:    proof.Count,
		Siblings: proof.Siblings,
	}, nil
}

func colsOf(counts TileCounts, matrixID byte) uint64 {
	if matrixID == MatrixIDA {
		return counts.ColsA
	}
	return counts.ColsB
}

// VerifyInputTileProof checks a tile against the committed matrix root.
func VerifyInputTileProof(root Hash, matrixID byte, tileRow, tileCol uint32, tileBytes []byte, p InputTileProof) bool {
	if uint32(matrixID) != p.MatrixID || p.TileRow != tileRow || p.TileCol != tileCol {
		return false
	}
	if p.TileCols == 0 || uint64(p.TileRow)*uint64(p.TileCols)+uint64(p.TileCol) >= uint64(p.Count) {
		return false
	}
	leaf := LeafInputTile(matrixID, tileRow, tileCol, tileBytes)
	proof := MerkleProof{Index: p.TileRow*p.TileCols + p.TileCol, Count: p.Count, Siblings: p.Siblings}
	return VerifyLeafInclusion(root, leaf, proof)
}
