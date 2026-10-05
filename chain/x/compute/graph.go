package compute

// CANONICAL_GRAPH_V1 on-chain settlement state. The graph path is a
// separate storage namespace and status machine that REUSES the GEMM
// settlement mechanics (escrow, worker bond reservation, challenge window,
// persisted dispute, permissionless bounded arbitration, feeSplit
// settlement) without touching the bounded-VM or lightweight paths. Every
// canonical value (graph id, work vector, state roots, verdicts) is
// derived through prismachain/compute/canonical, never accepted raw.

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"

	"prismachain/compute/canonical"
)

// GraphProtocolVersion is the wire protocol string of the settlement path.
const GraphProtocolVersion = "CANONICAL_GRAPH_V1/1.0.0"

// GraphRequesterKeyBindingDomain separates requester key-binding signatures.
const GraphRequesterKeyBindingDomain = "prisma:graph-requester-key-binding:v1\n"

// Admission bounds (DoS): a graph task is a small static block, never an
// arbitrary program.
const (
	// Bounds raised for the Phase F.1 real model block (Qwen3 layer 0 is a
	// 308-node static graph); they are DoS policy, not protocol semail.
	MaxGraphJSONBytes      = 1 << 20
	MaxGraphNodes          = 512
	MaxGraphInputs         = 256
	MaxGraphEvidenceChunks = 512
	MaxGraphOutputs        = 64
	// One active challenge per task (the Phase E one-open-challenge bound);
	// further challenges are refused until this one resolves.
	MaxGraphChallengeBondMultiple = 16
)

// Graph task statuses: posted -> assigned -> result_submitted ->
// finalized, with challenged disputes in between and refunded / fraud as
// terminal failure states.
const (
	GraphStatusPosted          = "posted"
	GraphStatusAssigned        = "assigned"
	GraphStatusResultSubmitted = "result_submitted"
	GraphStatusChallenged      = "challenged"
	GraphStatusFinalized       = "finalized"
	GraphStatusRefunded        = "refunded"
	GraphStatusFraud           = "fraud"
)

// Graph dispute sub-phase statuses.
const (
	GraphDisputeClaiming  = "claiming"
	GraphDisputeBisection = "bisection"
	GraphDisputeArbReady  = "arb_ready"
)

// GraphTrailClaimData persists one party's locked trail claim.
type GraphTrailClaimData struct {
	Root         []byte   `json:"root"`
	InitialRoot  []byte   `json:"initial_root"`
	InitialProof [][]byte `json:"initial_proof"`
	FinalRoot    []byte   `json:"final_root"`
	FinalProof   [][]byte `json:"final_proof"`
}

// GraphTask is the on-chain settlement record of one graph task.
type GraphTask struct {
	ID uint64 `json:"id"`

	GraphID   []byte `json:"graph_id"`
	Spec      string `json:"spec"`
	GraphJSON []byte `json:"graph_json"`

	Requester               string `json:"requester"`
	RequesterProtocolPubKey []byte `json:"requester_protocol_pub_key"`
	InputDataRef            string `json:"input_data_ref"`
	ChallengeWindow         uint64 `json:"challenge_window"`
	MaxPricePerCWU          uint64 `json:"max_price_per_cwu"`
	MaxFee                  uint64 `json:"max_fee"`
	IssuedHeight            uint64 `json:"issued_height"`

	Worker               string `json:"worker,omitempty"`
	WorkerProtocolPubKey []byte `json:"worker_protocol_pub_key,omitempty"`
	AssignmentNonce      []byte `json:"assignment_nonce,omitempty"`
	AssignmentRef        []byte `json:"assignment_ref,omitempty"`
	AcceptedHeight       uint64 `json:"accepted_height,omitempty"`
	ReservedBond         uint64 `json:"reserved_bond,omitempty"`

	// ProtocolVersion is the descriptor's own version string ("" = V1,
	// "CANONICAL_GRAPH_V2/1.0.0" = the wide-integer family).  Every later
	// phase dispatches on it; V1 tasks never see the V2 route.
	ProtocolVersion string `json:"protocol_version,omitempty"`

	// CommitVersion selects the settlement receipt: "" / "1" = V1,
	// "2" = GraphResultCommitV2 (node-level manifest), "3" =
	// GraphResultCommitV3 (wide-integer Graph V2 manifest).
	CommitVersion          string `json:"commit_version,omitempty"`
	NodeOutputManifestRoot []byte `json:"node_output_manifest_root,omitempty"`

	FinalOutputRoot       []byte   `json:"final_output_root,omitempty"`
	OutputRoots           [][]byte `json:"output_roots,omitempty"`
	ResultSignature       []byte   `json:"result_signature,omitempty"`
	CompletedEpoch        uint64   `json:"completed_epoch,omitempty"`
	ResultSubmittedHeight uint64   `json:"result_submitted_height,omitempty"`
	ChallengeEnd          uint64   `json:"challenge_end,omitempty"`

	Status string `json:"status"`

	ReceiptID               []byte `json:"receipt_id,omitempty"`
	SurvivedChallenge       bool   `json:"survived_challenge,omitempty"`
	DisputeTranscriptDigest []byte `json:"dispute_transcript_digest,omitempty"`
}

// GraphDisputeRecord persists one challenge and its dispute. The dispute
// core lives in the canonical GDS1 snapshot; everything around it is chain
// identity and economics.
type GraphDisputeRecord struct {
	TaskID            uint64              `json:"task_id"`
	Challenger        string              `json:"challenger"`
	ChallengerPubKey  []byte              `json:"challenger_pub_key"`
	Bond              uint64              `json:"bond"`
	ChallengerOutputs [][]byte            `json:"challenger_outputs"`
	WorkerClaim       GraphTrailClaimData `json:"worker_claim,omitempty"`
	ChallengerClaim   GraphTrailClaimData `json:"challenger_claim,omitempty"`
	Snapshot          []byte              `json:"snapshot,omitempty"`
	ClaimDeadline     uint64              `json:"claim_deadline"`
	Status            string              `json:"status"`
	Outcome           string              `json:"outcome,omitempty"`
	TranscriptDigest  []byte              `json:"transcript_digest"`
}

func graphTaskKey(id uint64) []byte {
	var key [9]byte
	key[0] = 'y'
	binary.BigEndian.PutUint64(key[1:], id)
	return key[:]
}

func graphDisputeKey(taskID uint64) []byte {
	var key [9]byte
	key[0] = 'x'
	binary.BigEndian.PutUint64(key[1:], taskID)
	return key[:]
}

func graphReceiptKey(receiptID []byte) []byte {
	return append([]byte{'z'}, receiptID...)
}

func (k Keeper) GetGraphTask(ctx context.Context, id uint64) (GraphTask, error) {
	data := k.store(ctx).Get(graphTaskKey(id))
	if len(data) == 0 {
		return GraphTask{}, errGraphTaskNotFound
	}
	var task GraphTask
	if err := json.Unmarshal(data, &task); err != nil {
		return GraphTask{}, err
	}
	return task, nil
}

func (k Keeper) setGraphTask(ctx context.Context, task GraphTask) error {
	data, err := json.Marshal(task)
	if err != nil {
		return err
	}
	k.store(ctx).Set(graphTaskKey(task.ID), data)
	return nil
}

func (k Keeper) GetGraphDispute(ctx context.Context, taskID uint64) (GraphDisputeRecord, error) {
	data := k.store(ctx).Get(graphDisputeKey(taskID))
	if len(data) == 0 {
		return GraphDisputeRecord{}, errGraphDisputeNotFound
	}
	var record GraphDisputeRecord
	if err := json.Unmarshal(data, &record); err != nil {
		return GraphDisputeRecord{}, err
	}
	return record, nil
}

func (k Keeper) setGraphDispute(ctx context.Context, record GraphDisputeRecord) error {
	data, err := json.Marshal(record)
	if err != nil {
		return err
	}
	k.store(ctx).Set(graphDisputeKey(record.TaskID), data)
	return nil
}

func (k Keeper) deleteGraphDispute(ctx context.Context, taskID uint64) {
	k.store(ctx).Delete(graphDisputeKey(taskID))
}

var (
	errGraphTaskNotFound    = errors.New("graph task not found")
	errGraphDisputeNotFound = errors.New("graph dispute not found")
	errGraphWrongPhase      = errors.New("graph task is not in the expected phase")
)

// graphDescriptor re-decodes and re-validates the stored descriptor; the
// stored graph id must match the derived one.
func graphDescriptor(task *GraphTask) (*canonical.GraphDescriptor, error) {
	var graph canonical.GraphDescriptor
	if err := json.Unmarshal(task.GraphJSON, &graph); err != nil {
		return nil, fmt.Errorf("graph descriptor decode: %w", err)
	}
	graphID, err := graph.GraphID()
	if err != nil {
		return nil, err
	}
	if !equalBytes32(graphID[:], task.GraphID) {
		return nil, errors.New("graph descriptor does not match the committed graph id")
	}
	return &graph, nil
}

func equalBytes32(a, b []byte) bool {
	if len(a) != 32 || len(b) != 32 {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func root32Of(raw []byte) (canonical.Hash, error) {
	var out canonical.Hash
	if len(raw) != 32 {
		return out, errors.New("expected a 32-byte root")
	}
	copy(out[:], raw)
	return out, nil
}

// graphInitialStateRoot derives the committed state root before node 0:
// the Merkle root over every input tensor leaf. It is a pure function of
// the descriptor, so the chain never trusts a claimed initial state.
func graphInitialStateRoot(g *canonical.GraphDescriptor) (canonical.Hash, error) {
	leaves := make([]canonical.Hash, 0, len(g.Inputs))
	for idx, in := range g.Inputs {
		root, err := root32Of(in.Root)
		if err != nil {
			return canonical.Hash{}, fmt.Errorf("input %d: %w", idx, err)
		}
		leaves = append(leaves, canonical.StateLeaf(0, uint32(idx), root))
	}
	if len(leaves) == 0 {
		return canonical.Hash{}, errors.New("graph has no inputs")
	}
	return canonical.MerkleRootOf(leaves)
}

// graphStateGeometry returns the leaf index and count of one tensor in
// the committed state tree before `beforeNode` nodes were executed.
// Inputs occupy the first slots; then every node output in id order.
func graphStateGeometry(g *canonical.GraphDescriptor, beforeNode uint32, ref canonical.TensorRef) (index, count uint32, err error) {
	k := uint32(len(g.Inputs))
	if ref.Kind == 0 {
		if ref.Index >= k {
			return 0, 0, errors.New("evidence references an unknown input")
		}
		return ref.Index, k + beforeNode, nil
	}
	if ref.Kind == 1 {
		if ref.Index >= beforeNode {
			return 0, 0, errors.New("evidence references a tensor that does not exist in the committed state")
		}
		return k + ref.Index, k + beforeNode, nil
	}
	return 0, 0, errors.New("unknown tensor ref kind")
}

// verifyGraphEvidenceState checks that one evidence tensor root is a leaf
// of the committed input state tree.
func verifyGraphEvidenceState(g *canonical.GraphDescriptor, beforeNode uint32, stateRoot canonical.Hash,
	ref canonical.TensorRef, root canonical.Hash, stateProof [][]byte) error {
	index, count, err := graphStateGeometry(g, beforeNode, ref)
	if err != nil {
		return err
	}
	siblings := make([]canonical.Hash, len(stateProof))
	for i, raw := range stateProof {
		h, err := root32Of(raw)
		if err != nil {
			return err
		}
		siblings[i] = h
	}
	leaf := canonical.StateLeaf(ref.Kind, ref.Index, root)
	if !canonical.VerifyLeafInclusion(stateRoot, leaf, canonical.MerkleProof{Index: index, Count: count, Siblings: siblings}) {
		return errors.New("evidence tensor is not part of the committed input state")
	}
	return nil
}

// graphTranscriptStep chains one dispute event into the transcript digest.
func graphTranscriptStep(prev []byte, tag string, event any) ([]byte, error) {
	payload, err := canonicalJSON(map[string]any{"tag": tag, "event": event})
	if err != nil {
		return nil, err
	}
	h := sha256.New()
	h.Write([]byte("prisma:graph-transcript:v1\n"))
	h.Write(prev)
	h.Write(payload)
	return h.Sum(nil), nil
}
