// Command prisma-gemm drives the GEMM_INT8_V1 protocol locally: task
// creation, worker execution, independent verification, dispute runs and
// benchmarks. It is a development CLI; matrices are generated
// deterministically from a seed so every artifact is reproducible without
// transporting multi-gigabyte inputs.
package main

import (
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"time"

	"prismachain/compute/gemmv1"
)

const (
	devWorkerSeedDomain     = "PRISMA_GEMM_DEV_WORKER_SEED_V1\x00"
	devChallengerSeedDomain = "PRISMA_GEMM_DEV_CHALLENGER_SEED_V1\x00"
	devRequesterSeedDomain  = "PRISMA_GEMM_DEV_REQUESTER_SEED_V1\x00"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	var err error
	switch os.Args[1] {
	case "create-task":
		err = cmdCreateTask(os.Args[2:])
	case "execute":
		err = cmdExecute(os.Args[2:])
	case "verify":
		err = cmdVerify(os.Args[2:])
	case "challenge":
		err = cmdChallenge(os.Args[2:], false)
	case "prove-step":
		err = cmdChallenge(os.Args[2:], true)
	case "trace-tile":
		err = cmdTraceTile(os.Args[2:])
	case "benchmark":
		err = cmdBenchmark(os.Args[2:])
	default:
		usage()
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "prisma-gemm: %v\n", err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprint(os.Stderr, `usage:
  prisma-gemm create-task --m 512 --n 512 --k 512 --seed 42 [--out task.json]
  prisma-gemm execute [--dev-inject-output-fraud I,J] [--out result.json] task.json
  prisma-gemm verify task.json result.json
  prisma-gemm challenge task.json result.json
  prisma-gemm prove-step task.json result.json   (verbose arbitration detail)
  prisma-gemm trace-tile task.json result.json --i 0 --j 0
  prisma-gemm benchmark --m 512 --n 512 --k 512 | --sizes 128,256,512
`)
}

func devKey(domain string, seed uint32) ed25519.PrivateKey {
	h := sha256.New()
	h.Write([]byte(domain))
	var b [4]byte
	binary.BigEndian.PutUint32(b[:], seed)
	h.Write(b[:])
	return ed25519.NewKeyFromSeed(h.Sum(nil))
}

// ---- JSON artifacts (development transport; hashing is always canonical
// CBOR inside the library) ----

type matrixSpec struct {
	Generator string `json:"generator"`
	Seed      uint32 `json:"seed"`
}

type outputTilesJSON struct {
	ColsC uint32    `json:"cols_c"`
	Tiles [][]int32 `json:"tiles"`
}

type taskJSON struct {
	Task       *gemmv1.TaskDescriptor `json:"task"`
	MatrixSpec matrixSpec             `json:"matrix_spec"`
}

type resultJSON struct {
	ResultCommit gemmv1.ResultCommit `json:"result_commit"`
	OutputTiles  outputTilesJSON     `json:"output_tiles"`
}

func writeJSON(path string, v any) error {
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), 0o600)
}

func cmdCreateTask(args []string) error {
	fs := flag.NewFlagSet("create-task", flag.ExitOnError)
	var m, n, k uint64
	var seed64 uint64
	var out string
	fs.Uint64Var(&m, "m", 512, "rows of A")
	fs.Uint64Var(&n, "n", 512, "cols of B")
	fs.Uint64Var(&k, "k", 512, "inner dimension")
	fs.Uint64Var(&seed64, "seed", 42, "deterministic matrix seed")
	fs.StringVar(&out, "out", "task.json", "output task file")
	if err := fs.Parse(args); err != nil {
		return err
	}

	started := time.Now()
	seed := uint32(seed64)
	requester := devKey(devRequesterSeedDomain, seed)
	matrixA := gemmv1.GenTestMatrix('A', seed, m*k)
	matrixB := gemmv1.GenTestMatrix('B', seed, k*n)
	rootA, rootB, err := gemmv1.BuildMatrixRoots(matrixA, matrixB, m, n, k)
	if err != nil {
		return err
	}
	task, err := gemmv1.NewTaskDescriptor(requester.Public().(ed25519.PublicKey), gemmv1.Int8ToBytes(gemmv1.GenTestMatrix('N', seed, 16)), uint64(started.Unix()), m, n, k, rootA[:], rootB[:], 100, 1000)
	if err != nil {
		return err
	}
	taskID, err := task.TaskID()
	if err != nil {
		return err
	}
	if err := writeJSON(out, taskJSON{Task: task, MatrixSpec: matrixSpec{Generator: "prisma_testgen_v1", Seed: seed}}); err != nil {
		return err
	}
	fmt.Printf("task_id: %x\nmatrix_a_root: %x\nmatrix_b_root: %x\ncanonical_mac_count: %d\nCWU: %.3f\ncommitment_time: %s\nwritten: %s\n",
		taskID, rootA[:], rootB[:], task.CanonicalMACCount(), gemmv1.CWUnits(task.CanonicalMACCount()), time.Since(started).Round(time.Millisecond), out)
	return nil
}

func loadTask(path string) (*gemmv1.TaskDescriptor, matrixSpec, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, matrixSpec{}, err
	}
	var tf taskJSON
	if err := json.Unmarshal(data, &tf); err != nil {
		return nil, matrixSpec{}, err
	}
	if tf.MatrixSpec.Generator != "prisma_testgen_v1" {
		return nil, matrixSpec{}, fmt.Errorf("unsupported matrix generator %q", tf.MatrixSpec.Generator)
	}
	if err := tf.Task.Validate(); err != nil {
		return nil, matrixSpec{}, err
	}
	return tf.Task, tf.MatrixSpec, nil
}

func matricesFor(t *gemmv1.TaskDescriptor, spec matrixSpec) ([]int8, []int8, error) {
	matrixA := gemmv1.GenTestMatrix('A', spec.Seed, t.M*t.K)
	matrixB := gemmv1.GenTestMatrix('B', spec.Seed, t.K*t.N)
	if err := gemmv1.VerifyMatrixRoots(t, matrixA, matrixB); err != nil {
		return nil, nil, err
	}
	return matrixA, matrixB, nil
}

func cmdExecute(args []string) error {
	fs := flag.NewFlagSet("execute", flag.ExitOnError)
	var out, fraud string
	fs.StringVar(&out, "out", "result.json", "output result file")
	fs.StringVar(&fraud, "dev-inject-output-fraud", "", "DEVELOPMENT ONLY: corrupt output tile (i,j)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 1 {
		return fmt.Errorf("execute needs exactly one task file")
	}
	task, spec, err := loadTask(fs.Arg(0))
	if err != nil {
		return err
	}
	matrixA, matrixB, err := matricesFor(task, spec)
	if err != nil {
		return err
	}

	started := time.Now()
	seed := spec.Seed
	workerPriv := devKey(devWorkerSeedDomain, seed)
	assignment := &gemmv1.Assignment{
		TaskID:          mustTaskID(task),
		WorkerPubKey:    workerPriv.Public().(ed25519.PublicKey),
		AssignmentNonce: gemmv1.Int8ToBytes(gemmv1.GenTestMatrix('M', seed, 16)),
		AcceptedEpoch:   task.IssuedEpoch + 1,
	}
	result, err := gemmv1.ExecuteTask(task, assignment, matrixA, matrixB, task.IssuedEpoch+2)
	if err != nil {
		return err
	}

	tiles := result.Tiles
	if fraud != "" {
		var i, j int
		if _, err := fmt.Sscanf(fraud, "%d,%d", &i, &j); err != nil {
			return fmt.Errorf("--dev-inject-output-fraud wants TILE_I,TILE_J: %w", err)
		}
		colsC := tileColsOf(task.N)
		idx := i*int(colsC) + j
		if idx < 0 || idx >= len(tiles) {
			return fmt.Errorf("fraud tile (%d,%d) out of range", i, j)
		}
		fmt.Printf("WARNING: development fraud injection into output tile (%d,%d); this result must be rejected by the challenger\n", i, j)
		tiles[idx][0] += 1
		// The fraudulent worker re-commits over the corrupted tiles.
		taskID := mustTaskID(task)
		assignmentID := mustAssignmentID(assignment)
		leaves := gemmv1.OutputLeaves(taskID, assignmentID, tiles, colsC)
		root, err := gemmv1.MerkleRoot(leaves)
		if err != nil {
			return err
		}
		result.OutputRoot = root
		result.ResultCommit.OutputRoot = append([]byte(nil), root[:]...)
	}
	if err := gemmv1.SignResultCommit(&result.ResultCommit, workerPriv); err != nil {
		return err
	}
	elapsed := time.Since(started)

	flat := make([][]int32, len(tiles))
	for idx, tile := range tiles {
		flat[idx] = append([]int32(nil), tile[:]...)
	}
	if err := writeJSON(out, resultJSON{ResultCommit: result.ResultCommit, OutputTiles: outputTilesJSON{ColsC: tileColsOf(task.N), Tiles: flat}}); err != nil {
		return err
	}
	fmt.Printf("output_root: %x\ncanonical_mac_count: %d\nCWU: %.3f\ngemm_and_commitment_time: %s\nwritten: %s\n",
		result.OutputRoot[:], result.ResultCommit.CanonicalMACCount, gemmv1.CWUnits(result.ResultCommit.CanonicalMACCount), elapsed.Round(time.Millisecond), out)
	return nil
}

func mustTaskID(t *gemmv1.TaskDescriptor) []byte {
	id, err := t.TaskID()
	if err != nil {
		panic(err)
	}
	return id
}

func mustAssignmentID(a *gemmv1.Assignment) []byte {
	id, err := a.AssignmentID()
	if err != nil {
		panic(err)
	}
	return id
}

func tileColsOf(dim uint64) uint32 {
	return uint32((dim + gemmv1.TileSize - 1) / gemmv1.TileSize)
}
