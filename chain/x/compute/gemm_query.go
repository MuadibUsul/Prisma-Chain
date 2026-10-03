package compute

// Read-only GEMM queries. Queries must not change state.

import (
	"context"
	"encoding/json"
	"errors"

	"prismachain/chain/x/compute/types"
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
