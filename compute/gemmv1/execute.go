package gemmv1

// CPU reference execution and the normal worker path: Input -> GEMM ->
// output tiles -> output Merkle root -> ResultCommit. The reference is the
// protocol oracle: it is intentionally not fast, it is exact.

import (
	"crypto/ed25519"
	"errors"
)

// ReferenceGEMM computes C = A x B with int8 inputs and a signed int32
// accumulator. Admission (Validate) bounds K <= MaxSafeK, so no partial sum
// in any loop order can overflow and the result is bit-exact everywhere.
// The implementation is sequential and deterministic; it is the protocol
// truth, the test oracle and the arbiter cross-check, not a performance
// path.
func ReferenceGEMM(a, b []int8, m, n, k uint64) []int32 {
	c := make([]int32, m*n)
	for i := uint64(0); i < m; i++ {
		ai := a[i*k : (i+1)*k]
		ci := c[i*n : (i+1)*n]
		for t := uint64(0); t < k; t++ {
			av := int32(ai[t])
			if av == 0 {
				continue
			}
			bt := b[t*n : (t+1)*n]
			for j := uint64(0); j < n; j++ {
				ci[j] += av * int32(bt[j])
			}
		}
	}
	return c
}

// OutputTiles slices C into zero-padded row-major 8x8 int32 tiles.
func OutputTiles(c []int32, m, n uint64) []State {
	counts := TileCountsFor(m, n, 8)
	tiles := make([]State, counts.RowsC*counts.ColsC)
	for i := uint64(0); i < counts.RowsC; i++ {
		for j := uint64(0); j < counts.ColsC; j++ {
			var tile State
			for row := uint64(0); row < TileSize; row++ {
				gi := i*TileSize + row
				if gi >= m {
					break
				}
				for col := uint64(0); col < TileSize; col++ {
					gj := j*TileSize + col
					if gj >= n {
						break
					}
					tile[row*TileSize+col] = c[gi*n+gj]
				}
			}
			tiles[i*counts.ColsC+j] = tile
		}
	}
	return tiles
}

// OutputLeaves computes the output tree leaves for one task/assignment.
func OutputLeaves(taskID, assignmentID []byte, tiles []State, colsC uint32) []Hash {
	leaves := make([]Hash, len(tiles))
	for idx, tile := range tiles {
		leaves[idx] = LeafOutputTile(taskID, assignmentID, uint32(idx)/colsC, uint32(idx)%colsC, tile.CanonicalBytes())
	}
	return leaves
}

// WorkerResult is the worker's local execution product before submission.
type WorkerResult struct {
	OutputRoot   Hash
	Tiles        []State
	ResultCommit ResultCommit
}

// ExecuteTask runs the full worker normal path: validate, verify input
// commitments, GEMM, tile, commit. CompletedEpoch stamps the ResultCommit.
func ExecuteTask(t *TaskDescriptor, a *Assignment, matrixA, matrixB []int8, completedEpoch uint64) (*WorkerResult, error) {
	if err := t.Validate(); err != nil {
		return nil, err
	}
	if err := CheckInputSizes(t.M, t.N, t.K, matrixA, matrixB); err != nil {
		return nil, err
	}
	taskID, err := t.TaskID()
	if err != nil {
		return nil, err
	}
	assignmentID, err := a.AssignmentID()
	if err != nil {
		return nil, err
	}
	if err := VerifyMatrixRoots(t, matrixA, matrixB); err != nil {
		return nil, err
	}
	if !equalBytes(a.TaskID, taskID) {
		return nil, errors.New("gemmv1: assignment is not for this task")
	}

	c := ReferenceGEMM(matrixA, matrixB, t.M, t.N, t.K)
	counts := TileCountsFor(t.M, t.N, t.K)
	tiles := OutputTiles(c, t.M, t.N)
	leaves := OutputLeaves(taskID, assignmentID, tiles, uint32(counts.ColsC))
	root, err := MerkleRoot(leaves)
	if err != nil {
		return nil, err
	}

	rc := ResultCommit{
		ProtocolVersion:   ProtocolVersion,
		TaskID:            taskID,
		AssignmentID:      assignmentID,
		WorkerPubKey:      append([]byte(nil), a.WorkerPubKey...),
		OutputRoot:        append([]byte(nil), root[:]...),
		CanonicalMACCount: t.CanonicalMACCount(),
		CompletedEpoch:    completedEpoch,
	}
	return &WorkerResult{OutputRoot: root, Tiles: tiles, ResultCommit: rc}, nil
}

// SignResultCommit fills the worker signature of a ResultCommit.
func SignResultCommit(rc *ResultCommit, key ed25519.PrivateKey) error {
	if err := checkPubKey(rc.WorkerPubKey); err != nil {
		return err
	}
	sigBytes, err := SignedBytes(rc)
	if err != nil {
		return err
	}
	rc.WorkerSignature = SignObject(key, sigBytes)
	return nil
}

// VerifyResultCommitSignature checks a ResultCommit against its worker key.
func VerifyResultCommitSignature(rc *ResultCommit) bool {
	if err := checkPubKey(rc.WorkerPubKey); err != nil {
		return false
	}
	if len(rc.WorkerSignature) != ed25519.SignatureSize {
		return false
	}
	unsigned := *rc
	unsigned.WorkerSignature = nil
	sigBytes, err := SignedBytes(&unsigned)
	if err != nil {
		return false
	}
	return VerifyObject(ed25519.PublicKey(rc.WorkerPubKey), sigBytes, rc.WorkerSignature)
}
