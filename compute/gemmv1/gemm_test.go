package gemmv1

// Shared test fixture: keys, deterministic matrices, task, assignment and a
// signed honest worker result.

import (
	"crypto/ed25519"
	"testing"
)

type fixture struct {
	reqPub       ed25519.PublicKey
	reqPriv      ed25519.PrivateKey
	workerPub    ed25519.PublicKey
	workerPriv   ed25519.PrivateKey
	challPub     ed25519.PublicKey
	challPriv    ed25519.PrivateKey
	matrixA      []int8
	matrixB      []int8
	rootA        Hash
	rootB        Hash
	task         *TaskDescriptor
	taskID       []byte
	assignment   *Assignment
	assignmentID []byte
	result       *WorkerResult
	colsC        uint32
}

func newFixture(t *testing.T, m, n, k uint64, seed uint32, withRandomKeys bool) *fixture {
	t.Helper()
	f := &fixture{}
	f.matrixA = GenTestMatrix('A', seed, m*k)
	f.matrixB = GenTestMatrix('B', seed, k*n)

	if withRandomKeys {
		var err error
		if f.reqPub, f.reqPriv, err = GenerateKeyPair(); err != nil {
			t.Fatal(err)
		}
		if f.workerPub, f.workerPriv, err = GenerateKeyPair(); err != nil {
			t.Fatal(err)
		}
		if f.challPub, f.challPriv, err = GenerateKeyPair(); err != nil {
			t.Fatal(err)
		}
	} else {
		f.reqPub = ed25519.PublicKey(Int8ToBytes(GenTestMatrix('Q', seed, 32)))
		f.workerPub = ed25519.PublicKey(Int8ToBytes(GenTestMatrix('W', seed, 32)))
		f.challPub = ed25519.PublicKey(Int8ToBytes(GenTestMatrix('C', seed, 32)))
		f.reqPriv = ed25519.PrivateKey(append(append([]byte(nil), f.reqPub...), Int8ToBytes(GenTestMatrix('q', seed, 32))...))
		f.workerPriv = ed25519.PrivateKey(append(append([]byte(nil), f.workerPub...), Int8ToBytes(GenTestMatrix('w', seed, 32))...))
		f.challPriv = ed25519.PrivateKey(append(append([]byte(nil), f.challPub...), Int8ToBytes(GenTestMatrix('c', seed, 32))...))
	}

	var err error
	f.rootA, f.rootB, err = BuildMatrixRoots(f.matrixA, f.matrixB, m, n, k)
	if err != nil {
		t.Fatal(err)
	}
	f.task, err = NewTaskDescriptor(f.reqPub, Int8ToBytes(GenTestMatrix('N', seed, 16)), 1000, m, n, k, f.rootA[:], f.rootB[:], 100, 1000)
	if err != nil {
		t.Fatal(err)
	}
	f.taskID, err = f.task.TaskID()
	if err != nil {
		t.Fatal(err)
	}
	f.assignment = &Assignment{
		TaskID:          append([]byte(nil), f.taskID...),
		WorkerPubKey:    append([]byte(nil), f.workerPub...),
		AssignmentNonce: Int8ToBytes(GenTestMatrix('M', seed, 16)),
		AcceptedEpoch:   1001,
	}
	f.assignmentID, err = f.assignment.AssignmentID()
	if err != nil {
		t.Fatal(err)
	}
	f.result, err = ExecuteTask(f.task, f.assignment, f.matrixA, f.matrixB, 1002)
	if err != nil {
		t.Fatal(err)
	}
	if err := SignResultCommit(&f.result.ResultCommit, f.workerPriv); err != nil {
		t.Fatal(err)
	}
	f.colsC = uint32(tileCols(n))
	return f
}

// challengerHonestTiles recomputes the canonical output tiles independently.
func (f *fixture) challengerHonestTiles() []State {
	c := ReferenceGEMM(f.matrixA, f.matrixB, f.task.M, f.task.N, f.task.K)
	return OutputTiles(c, f.task.M, f.task.N)
}
