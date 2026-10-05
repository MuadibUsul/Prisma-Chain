package compute

// F.5C A5: aggregate Graph V2 read surface.  External tools, the CLI and
// the watcher recover everything they need from ONE formal query - task
// identity and phase, DA availability, dispute/wide progress and the V3
// receipt - instead of reading the raw ABCI store.

import (
	"context"
	"encoding/json"
	"sort"

	sdk "github.com/cosmos/cosmos-sdk/types"

	"prismachain/chain/x/compute/types"
)

func (q queryServer) GraphV2Status(ctx context.Context, req *types.QueryGraphV2StatusRequest) (*types.QueryGraphV2StatusResponse, error) {
	_ = sdk.UnwrapSDKContext(ctx)
	task, err := q.GetGraphTask(ctx, req.GraphTaskId)
	if err != nil {
		return nil, err
	}
	doc := map[string]any{
		"task": map[string]any{
			"id":                        task.ID,
			"protocol_version":          task.ProtocolVersion,
			"graph_id":                  task.GraphID,
			"spec":                      task.Spec,
			"status":                    task.Status,
			"requester":                 task.Requester,
			"worker":                    task.Worker,
			"issued_height":             task.IssuedHeight,
			"accepted_height":           task.AcceptedHeight,
			"challenge_window":          task.ChallengeWindow,
			"challenge_end":             task.ChallengeEnd,
			"result_submitted_height":   task.ResultSubmittedHeight,
			"completed_epoch":           task.CompletedEpoch,
			"commit_version":            task.CommitVersion,
			"node_output_manifest_root": task.NodeOutputManifestRoot,
			"final_output_root":         task.FinalOutputRoot,
			"output_roots":              task.OutputRoots,
			"receipt_id":                task.ReceiptID,
			"survived_challenge":        task.SurvivedChallenge,
		},
	}
	if rec, err := q.GetGraphDARecord(ctx, task.ID); err == nil {
		attesters := make([]string, 0, len(rec.Attesters))
		for k := range rec.Attesters {
			attesters = append(attesters, k)
		}
		sort.Strings(attesters)
		doc["da"] = map[string]any{
			"status":          rec.Status,
			"attesters":       attesters,
			"open_challenges": rec.OpenChallenges,
		}
	}
	if rec, err := q.GetGraphDispute(ctx, task.ID); err == nil {
		doc["dispute"] = map[string]any{
			"status":            rec.Status,
			"challenger":        rec.Challenger,
			"bond":              rec.Bond,
			"claim_deadline":    rec.ClaimDeadline,
			"has_snapshot":      len(rec.Snapshot) > 0,
			"worker_claim_root": rec.WorkerClaim.Root,
			"challenger_root":   rec.ChallengerClaim.Root,
			"outcome":           rec.Outcome,
		}
	}
	if wd, err := q.GetWideDispute(ctx, task.ID); err == nil {
		doc["wide"] = map[string]any{
			"status": wd.Status, "node_id": wd.NodeID,
			"tile_i": wd.TileI, "tile_j": wd.TileJ,
			"m": wd.M, "n": wd.N, "k": wd.K, "transpose_b": wd.TransposeB,
			"claim_deadline": wd.ClaimDeadline, "outcome": wd.Outcome,
		}
	}
	if len(task.ReceiptID) == 32 {
		if raw := q.store(ctx).Get(graphReceiptKey(task.ReceiptID)); len(raw) != 0 {
			doc["receipt_json"] = json.RawMessage(raw)
		}
	}
	body, err := json.Marshal(doc)
	if err != nil {
		return nil, err
	}
	return &types.QueryGraphV2StatusResponse{StatusJson: body}, nil
}
