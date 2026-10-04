package compute

// CANONICAL_GRAPH_V2 on-chain path (Phase F.5C).  Additive: the task
// envelope (`MsgPostGraphTask.graph_json`) carries its own explicit
// `protocol_version`, and the chain dispatches on it.  V1 transactions
// keep their exact behaviour; nothing is silently reinterpreted.
//
// Scope of the V2 route in this file: post (V2 admission with typed
// descriptor validation and arithmetic/policy binding), submit
// (GraphResultCommitV3 validation), finalize (V3 receipt).  The dispute
// and DA extensions build on the same dispatch.

import (
	"context"
	"crypto/ed25519"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"

	sdk "github.com/cosmos/cosmos-sdk/types"

	"prismachain/chain/x/compute/types"
	"prismachain/compute/canonical"
)

// graphProtocolVersionOf extracts the descriptor's own protocol_version.
func graphProtocolVersionOf(graphJSON []byte) (string, error) {
	var probe struct {
		ProtocolVersion string `json:"protocol_version"`
	}
	if err := json.Unmarshal(graphJSON, &probe); err != nil {
		return "", err
	}
	return probe.ProtocolVersion, nil
}

// graphTaskProtocolV2 reports whether a stored task is a V2 task.
func graphTaskProtocolV2(task *GraphTask) bool {
	return task.ProtocolVersion == canonical.ProtocolVersionGraphV2
}

// graphDescriptorV2 decodes, re-validates and re-hashes the stored V2
// descriptor; the stored graph id must match the derived GraphIDV2.
func graphDescriptorV2(task *GraphTask) (*canonical.GraphDescriptorV2, error) {
	var g canonical.GraphDescriptorV2
	if err := json.Unmarshal(task.GraphJSON, &g); err != nil {
		return nil, fmt.Errorf("graph V2 descriptor decode: %w", err)
	}
	if err := g.ValidateV2(); err != nil {
		return nil, err
	}
	graphID, err := g.GraphIDV2()
	if err != nil {
		return nil, err
	}
	if !equalBytes32(graphID[:], task.GraphID) {
		return nil, errors.New("graph V2 descriptor does not match the committed graph id")
	}
	return &g, nil
}

// postGraphTaskV2 is the admission path for a canonical GRAPH_V2 task:
// bounds first, then full typed validation (dtypes, ranges, operator
// versions, arithmetic profile, MaxSafeK64 per GEMM), then escrow and
// storage.  The task records its protocol version so every later phase
// dispatches deterministically.
func (s msgServer) postGraphTaskV2(ctx context.Context, msg *types.MsgPostGraphTask,
	height uint64, requester sdk.AccAddress) (*types.MsgPostGraphTaskResponse, error) {
	var graph canonical.GraphDescriptorV2
	if err := json.Unmarshal(msg.GraphJson, &graph); err != nil {
		return nil, fmt.Errorf("graph V2 descriptor decode: %w", err)
	}
	if len(graph.Nodes) > MaxGraphNodes || len(graph.Inputs) > MaxGraphInputs ||
		len(graph.Outputs) > MaxGraphOutputs {
		return nil, errors.New("graph V2 exceeds the admission bounds")
	}
	if err := graph.ValidateV2(); err != nil {
		return nil, err
	}
	s.consumeGraphGas(ctx, "graph V2 decode",
		GasGraphNodeDecode*uint64(len(graph.Nodes))+GasGraphInputDecode*uint64(len(graph.Inputs)))
	work, err := canonical.GraphWorkVectorV2(&graph)
	if err != nil {
		return nil, err
	}
	s.consumeGraphGas(ctx, "graph V2 work", graphWorkGas(work))
	graphID, err := graph.GraphIDV2()
	if err != nil {
		return nil, err
	}
	if err := s.bank.SendCoinsFromAccountToModule(ctx, requester, ModuleName, amount(msg.MaxFee)); err != nil {
		return nil, err
	}
	store := s.store(ctx)
	idBytes := store.Get([]byte("graph:next"))
	var id uint64 = 1
	if len(idBytes) == 8 {
		id = binary.BigEndian.Uint64(idBytes)
	}
	if id == ^uint64(0) {
		return nil, errors.New("graph task id exhausted")
	}
	var next [8]byte
	binary.BigEndian.PutUint64(next[:], id+1)
	store.Set([]byte("graph:next"), next[:])
	task := GraphTask{
		ID: id, GraphID: graphID[:], Spec: graph.Spec, GraphJSON: append([]byte(nil), msg.GraphJson...),
		ProtocolVersion:        canonical.ProtocolVersionGraphV2,
		Requester:              msg.Requester,
		RequesterProtocolPubKey: append([]byte(nil), msg.RequesterProtocolPubkey...),
		InputDataRef:           msg.InputDataRef, ChallengeWindow: msg.ChallengeWindow,
		MaxPricePerCWU: msg.MaxPricePerCwu, MaxFee: msg.MaxFee, IssuedHeight: height,
		Status: GraphStatusPosted,
	}
	if err := s.setGraphTask(ctx, task); err != nil {
		return nil, err
	}
	s.consumeGraphGas(ctx, "graph V2 store",
		GasGraphStore+GasGraphHash*uint64(len(msg.GraphJson)/256+1))
	return &types.MsgPostGraphTaskResponse{GraphTaskId: id, GraphId: graphID[:]}, nil
}

// submitGraphResultV3 validates a GraphResultCommitV3 submission for a V2
// task and locks the manifest root before any verification randomness.
func (s msgServer) submitGraphResultV3(ctx context.Context, task *GraphTask,
	graph *canonical.GraphDescriptorV2, msg *types.MsgSubmitGraphResultV2,
	height uint64) (*types.MsgSubmitGraphResultV2Response, error) {
	rc, err := buildAndValidateV3(task, graph, msg)
	if err != nil {
		return nil, err
	}
	task.CommitVersion = "3"
	task.NodeOutputManifestRoot = append([]byte(nil), rc.NodeOutputManifestRootV2...)
	task.FinalOutputRoot = append([]byte(nil), rc.FinalOutputRoot...)
	task.OutputRoots = cloneBytes2D(rc.OutputRoots)
	task.ResultSignature = append([]byte(nil), rc.Signature...)
	task.CompletedEpoch = msg.CompletedEpoch
	task.ResultSubmittedHeight = height
	task.ChallengeEnd = height + task.ChallengeWindow
	task.Status = GraphStatusResultSubmitted
	if err := s.setGraphTask(ctx, *task); err != nil {
		return nil, err
	}
	return &types.MsgSubmitGraphResultV2Response{ChallengeEnd: task.ChallengeEnd}, nil
}

func buildAndValidateV3(task *GraphTask, graph *canonical.GraphDescriptorV2,
	msg *types.MsgSubmitGraphResultV2) (*canonical.GraphResultCommitV3, error) {
	if len(msg.NodeOutputManifestRoot) != 32 {
		return nil, errors.New("graph V3 manifest root must be 32 bytes")
	}
	if len(msg.OutputRoots) != len(graph.Outputs) || len(msg.WorkerSignature) != ed25519.SignatureSize {
		return nil, errors.New("graph V3 commitment shape is invalid")
	}
	var taskRef [8]byte
	binary.BigEndian.PutUint64(taskRef[:], task.ID)
	rc := &canonical.GraphResultCommitV3{
		ProtocolVersion:          canonical.GraphResultCommitV3Version,
		GraphID:                  append([]byte(nil), task.GraphID...),
		ArithmeticID:             graph.Arithmetic.ID,
		PolicyID:                 graph.Arithmetic.PolicyID,
		TaskRef:                  taskRef[:],
		AssignmentRef:            append([]byte(nil), task.AssignmentRef...),
		WorkerPubKey:             append([]byte(nil), task.WorkerProtocolPubKey...),
		NodeOutputManifestRootV2: append([]byte(nil), msg.NodeOutputManifestRoot...),
		FinalOutputRoot:          append([]byte(nil), msg.FinalOutputRoot...),
		OutputRoots:              cloneBytes2D(msg.OutputRoots),
		CompletedEpoch:           msg.CompletedEpoch,
		Signature:                append([]byte(nil), msg.WorkerSignature...),
	}
	if err := canonical.ValidateGraphResultCommitV3(graph, rc, nil); err != nil {
		return nil, err
	}
	return rc, nil
}
