package compute

// Read-only GEMM queries. Queries must not change state.

import (
	"context"
	"encoding/json"
	"errors"

	"prismachain/chain/x/compute/types"
	"prismachain/compute/gemmv1"
)

func (q queryServer) GEMMTask(ctx context.Context, req *types.QueryGEMMTaskRequest) (*types.QueryGEMMTaskResponse, error) {
	task, err := q.GetGEMMTask(ctx, req.GemmTaskId)
	if err != nil {
		return nil, err
	}
	taskJSON, err := json.Marshal(task)
	if err != nil {
		return nil, err
	}
	var disputeJSON []byte
	if record, err := q.GetGEMMDispute(ctx, req.GemmTaskId); err == nil {
		disputeJSON, err = json.Marshal(record)
		if err != nil {
			return nil, err
		}
	}
	return &types.QueryGEMMTaskResponse{TaskJson: taskJSON, DisputeJson: disputeJSON}, nil
}

func (q queryServer) GEMMDispute(ctx context.Context, req *types.QueryGEMMDisputeRequest) (*types.QueryGEMMDisputeResponse, error) {
	record, err := q.GetGEMMDispute(ctx, req.GemmTaskId)
	if err != nil {
		return nil, err
	}
	data, err := json.Marshal(record)
	if err != nil {
		return nil, err
	}
	return &types.QueryGEMMDisputeResponse{DisputeJson: data}, nil
}

func (q queryServer) GEMMReceipt(ctx context.Context, req *types.QueryGEMMReceiptRequest) (*types.QueryGEMMReceiptResponse, error) {
	if len(req.ReceiptId) != 32 {
		return nil, errors.New("receipt id must be 32 bytes")
	}
	data := q.store(ctx).Get(gemmReceiptKey(req.ReceiptId))
	if len(data) == 0 {
		return nil, errors.New("gemm receipt not found")
	}
	return &types.QueryGEMMReceiptResponse{ReceiptJson: data}, nil
}

func (q queryServer) GEMMWorkerWork(ctx context.Context, req *types.QueryGEMMWorkerWorkRequest) (*types.QueryGEMMWorkerWorkResponse, error) {
	if _, err := addr(req.Worker); err != nil {
		return nil, err
	}
	return &types.QueryGEMMWorkerWorkResponse{TotalCanonicalMac: q.GetGEMMVerifiedWork(ctx, req.Worker)}, nil
}

// GEMMDAStatus reports the availability accounting of one GEMM task.
func (q queryServer) GEMMDAStatus(ctx context.Context, req *types.QueryGEMMDAStatusRequest) (*types.QueryGEMMDAStatusResponse, error) {
	task, err := q.GetGEMMTask(ctx, req.GemmTaskId)
	if err != nil {
		return nil, err
	}
	server := msgServer{q.Keeper}
	status, valid, open := server.gemmDAStatus(ctx, task)
	var attestationsJSON []byte
	if record, err := q.GetDARecord(ctx, req.GemmTaskId); err == nil {
		attestationsJSON, err = json.Marshal(record.Attesters)
		if err != nil {
			return nil, err
		}
	}
	return &types.QueryGEMMDAStatusResponse{
		RequiredReplicas: gemmv1.DARequiredReplicas, ValidReplicas: valid,
		OpenChallenges: open, AvailabilityStatus: status, AttestationsJson: attestationsJSON,
	}, nil
}

// DAProvider reports one registered availability provider.
func (q queryServer) DAProvider(ctx context.Context, req *types.QueryDAProviderRequest) (*types.QueryDAProviderResponse, error) {
	provider, err := q.GetDAProvider(ctx, req.Provider)
	if err != nil {
		return nil, err
	}
	data, err := json.Marshal(provider)
	if err != nil {
		return nil, err
	}
	return &types.QueryDAProviderResponse{ProviderJson: data}, nil
}
