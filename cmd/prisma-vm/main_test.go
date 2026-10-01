package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"prismachain/vm"
)

func TestLoadJob(t *testing.T) {
	program, input, err := loadJob(strings.NewReader(`{"input":[3],"program":[{"op":"input","dst":1},{"op":"set","dst":2,"imm":2},{"op":"mul","dst":0,"a":1,"b":2}]}`))
	if err != nil {
		t.Fatal(err)
	}
	execution, err := vm.Execute(program, input)
	if err != nil || execution.Output() != 6 {
		t.Fatalf("output=%d, err=%v", execution.Output(), err)
	}
	if _, _, err := loadJob(strings.NewReader(`{"program":[{"op":"hidden"}]}`)); err == nil {
		t.Fatal("unknown opcode accepted")
	}
}

func TestExportAndIndependentReview(t *testing.T) {
	job := filepath.Join("..", "..", "examples", "public-classifier.json")
	var exported bytes.Buffer
	if err := runWithOptions(&exported, job, -1, true, ""); err != nil {
		t.Fatal(err)
	}
	claimFile := filepath.Join(t.TempDir(), "worker-result.json")
	if err := os.WriteFile(claimFile, exported.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
	var reviewed bytes.Buffer
	if err := runWithOptions(&reviewed, job, -1, false, claimFile); err != nil {
		t.Fatal(err)
	}
	var result struct {
		Action vm.ReviewAction `json:"review_action"`
	}
	if err := json.Unmarshal(reviewed.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result.Action != vm.Attest {
		t.Fatalf("full replay did not attest valid claim: %s", result.Action)
	}
}
