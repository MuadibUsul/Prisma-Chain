package compute

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strconv"
)

const MaxLightReceiptBytes = 8192

type lightAttestation struct {
	TaskID           string `json:"task_id"`
	AttemptID        string `json:"attempt_id"`
	WorkerNodeID     string `json:"worker_node_id"`
	GatewayNodeID    string `json:"gateway_node_id"`
	GroupID          string `json:"group_id"`
	LeaseEpoch       uint64 `json:"lease_epoch"`
	ModelID          string `json:"model_id"`
	ModelDigest      string `json:"model_digest"`
	SpecVersion      string `json:"spec_version"`
	InputCommitment  string `json:"input_commitment"`
	OutputCommitment string `json:"output_commitment"`
	OutputTokens     uint64 `json:"output_tokens"`
	CompletedAtMS    uint64 `json:"completed_at_ms"`
	Status           string `json:"status"`
}

type lightReceipt struct {
	lightAttestation
	Mode              string           `json:"mode"`
	WorkerAttestation lightAttestation `json:"worker_attestation"`
	WorkerSignature   string           `json:"worker_signature"`
	GatewaySignature  string           `json:"gateway_signature"`
}

// canonicalReceipt accepts exactly the JSON bytes signed by the Python v1
// gateway: sorted compact keys, UTF-8 text, and integer numeric fields.
func canonicalReceipt(data []byte) (map[string]any, error) {
	decode := json.NewDecoder(bytes.NewReader(data))
	decode.UseNumber()
	var value map[string]any
	if err := decode.Decode(&value); err != nil {
		return nil, err
	}
	var extra any
	if decode.Decode(&extra) != io.EOF {
		return nil, errors.New("trailing receipt JSON")
	}
	encoded, err := canonicalJSON(value)
	if err != nil || !bytes.Equal(encoded, data) {
		return nil, errors.New("receipt JSON is not v1 canonical")
	}
	return value, nil
}

func canonicalJSON(value any) ([]byte, error) {
	var buffer bytes.Buffer
	encoder := json.NewEncoder(&buffer)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(value); err != nil {
		return nil, err
	}
	return bytes.TrimSuffix(buffer.Bytes(), []byte{'\n'}), nil
}

func signedBy(publicKey []byte, domain string, payload any, encodedSignature string) bool {
	signature, err := base64.StdEncoding.DecodeString(encodedSignature)
	if err != nil || len(signature) != ed25519.SignatureSize {
		return false
	}
	canonical, err := canonicalJSON(payload)
	if err != nil {
		return false
	}
	message := append([]byte(domain+"\n"), canonical...)
	return ed25519.Verify(publicKey, message, signature)
}

func (s msgServer) bondedNodeKey(ctx context.Context, nodeID string) (string, []byte, error) {
	node, err := hex.DecodeString(nodeID)
	if err != nil || len(node) != sha256.Size || hex.EncodeToString(node) != nodeID {
		return "", nil, errors.New("invalid receipt node ID")
	}
	var id [sha256.Size]byte
	copy(id[:], node)
	account := string(s.store(ctx).Get(nodeKey(id)))
	key := s.store(ctx).Get(networkKey(account))
	if account == "" || len(key) != ed25519.PublicKeySize ||
		len(s.store(ctx).Get(networkProofKey(account))) == 0 || s.GetBond(ctx, account) < MinBond ||
		sha256.Sum256(key) != id {
		return "", nil, errors.New("receipt node is not bonded and registered")
	}
	return account, key, nil
}

func (s msgServer) verifyLightReceipt(ctx context.Context, task Task, raw []byte,
	outputDigest, receiptDigest []byte, outputTokens uint64) (string, error) {
	if len(raw) == 0 || len(raw) > MaxLightReceiptBytes {
		return "", errors.New("invalid lightweight receipt size")
	}
	value, err := canonicalReceipt(raw)
	if err != nil {
		return "", err
	}
	var receipt lightReceipt
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&receipt); err != nil {
		return "", err
	}
	model, err := s.GetModel(ctx, task.ModelID, task.SpecVersion)
	if err != nil {
		return "", err
	}
	pinned, err := canonicalJSON(map[string]any{
		"model_id": model.ID, "runtime_digest": hex.EncodeToString(model.ImageDigest),
		"spec_version": model.Version, "tokenizer_digest": hex.EncodeToString(model.TokenizerDigest),
		"weights_digest": hex.EncodeToString(model.WeightsDigest),
	})
	if err != nil {
		return "", err
	}
	modelDigest := sha256.Sum256(pinned)
	att := receipt.WorkerAttestation
	if receipt.Mode != "lightweight" || receipt.Status != "delivered" ||
		receipt.TaskID != strconv.FormatUint(task.ID, 10) || receipt.ModelID != task.ModelID ||
		receipt.SpecVersion != task.SpecVersion || receipt.ModelDigest != hex.EncodeToString(modelDigest[:]) ||
		receipt.InputCommitment != hex.EncodeToString(task.InputCommitment) ||
		receipt.OutputCommitment != hex.EncodeToString(outputDigest) ||
		receipt.OutputTokens != outputTokens || outputTokens == 0 || receipt.LeaseEpoch == 0 ||
		receipt.CompletedAtMS == 0 || !validLabel(receipt.AttemptID, false) ||
		!validLabel(receipt.GroupID, false) || att != receipt.lightAttestation {
		return "", errors.New("lightweight receipt does not match task or worker attestation")
	}
	workerAccount, workerKey, err := s.bondedNodeKey(ctx, receipt.WorkerNodeID)
	if err != nil || workerAccount != task.Worker {
		return "", errors.New("receipt worker does not match accepted task")
	}
	gatewayAccount, gatewayKey, err := s.bondedNodeKey(ctx, receipt.GatewayNodeID)
	if err != nil || gatewayAccount == workerAccount {
		return "", errors.New("receipt gateway must be a distinct bonded node")
	}
	workerPayload, ok := value["worker_attestation"]
	if !ok || !signedBy(workerKey, "prisma:worker-receipt:v1", workerPayload, receipt.WorkerSignature) {
		return "", errors.New("invalid worker receipt signature")
	}
	delete(value, "gateway_signature")
	if !signedBy(gatewayKey, "prisma:gateway-receipt:v1", value, receipt.GatewaySignature) {
		return "", errors.New("invalid gateway receipt signature")
	}
	hash := sha256.Sum256(raw)
	if !bytes.Equal(hash[:], receiptDigest) {
		return "", fmt.Errorf("receipt digest does not match signed bytes")
	}
	return gatewayAccount, nil
}
