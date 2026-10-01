// prisma-vm runs a public bounded integer task and prints an auditable trace
// commitment. It is a local development tool, not a consensus node.
package main

import (
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"

	"prismachain/vm"
)

type instructionJSON struct {
	Op  string `json:"op"`
	Dst uint8  `json:"dst"`
	A   uint8  `json:"a"`
	B   uint8  `json:"b"`
	C   uint8  `json:"c"`
	Imm int64  `json:"imm"`
}

type jobJSON struct {
	Input   []int64           `json:"input"`
	Program []instructionJSON `json:"program"`
}

var opNames = map[string]vm.Op{
	"set": vm.OpSet, "input": vm.OpInput, "add": vm.OpAdd,
	"sub": vm.OpSub, "mul": vm.OpMul, "div": vm.OpDiv,
	"muldiv": vm.OpMulDiv, "max": vm.OpMax, "gt": vm.OpGT,
	"select": vm.OpSelect,
}

func loadJob(r io.Reader) (vm.Program, []int64, error) {
	var job jobJSON
	decoder := json.NewDecoder(io.LimitReader(r, 1<<20))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&job); err != nil {
		return vm.Program{}, nil, err
	}
	var extra interface{}
	if err := decoder.Decode(&extra); err != io.EOF {
		return vm.Program{}, nil, errors.New("job must contain exactly one JSON object")
	}
	code := make([]vm.Instruction, len(job.Program))
	for i, item := range job.Program {
		op, ok := opNames[item.Op]
		if !ok {
			return vm.Program{}, nil, fmt.Errorf("program[%d]: unknown opcode %q", i, item.Op)
		}
		code[i] = vm.Instruction{Op: op, Dst: item.Dst, A: item.A, B: item.B, C: item.C, Imm: item.Imm}
	}
	program, err := vm.NewProgram(code)
	return program, job.Input, err
}

func run(w io.Writer, jobFile string, proofIndex int) error {
	return runWithOptions(w, jobFile, proofIndex, false, "")
}

func runWithOptions(w io.Writer, jobFile string, proofIndex int, exportClaim bool, reviewFile string) error {
	file, err := os.Open(jobFile)
	if err != nil {
		return err
	}
	defer file.Close()
	program, input, err := loadJob(file)
	if err != nil {
		return err
	}
	execution, err := vm.Execute(program, input)
	if err != nil {
		return err
	}
	programDigest := program.Digest()
	inputDigest := vm.InputDigest(input)
	traceRoot := execution.Root()
	response := map[string]interface{}{
		"spec_version":   vm.SpecVersion,
		"program_digest": hex.EncodeToString(programDigest[:]),
		"input_digest":   hex.EncodeToString(inputDigest[:]),
		"trace_root":     hex.EncodeToString(traceRoot[:]),
		"steps":          execution.Steps(),
		"output":         execution.Output(),
	}
	if exportClaim {
		_, initialProof, err := execution.Proof(0)
		if err != nil {
			return err
		}
		final, finalProof, err := execution.Proof(execution.Steps())
		if err != nil {
			return err
		}
		response["trace_claim"] = vm.TraceClaim{
			Root: traceRoot, InitialProof: initialProof,
			Final: final, FinalProof: finalProof,
		}
	}
	if reviewFile != "" {
		claimFile, err := os.Open(reviewFile)
		if err != nil {
			return err
		}
		defer claimFile.Close()
		decoder := json.NewDecoder(io.LimitReader(claimFile, 1<<20))
		var envelope map[string]json.RawMessage
		if err := decoder.Decode(&envelope); err != nil {
			return err
		}
		var extra interface{}
		if err := decoder.Decode(&extra); err != io.EOF {
			return errors.New("review file must contain exactly one JSON object")
		}
		claimBytes, ok := envelope["trace_claim"]
		if !ok {
			return errors.New("review file has no trace_claim")
		}
		var workerClaim vm.TraceClaim
		if err := json.Unmarshal(claimBytes, &workerClaim); err != nil {
			return err
		}
		review, err := vm.ReviewClaim(program, input, workerClaim)
		if err != nil {
			return err
		}
		response["review_action"] = review.Action
		if review.Action == vm.Challenge {
			response["challenger_claim"] = review.Canonical
		}
	}
	if proofIndex >= 0 {
		if proofIndex > int(execution.Steps()) {
			return fmt.Errorf("proof index %d exceeds final state %d", proofIndex, execution.Steps())
		}
		state, proof, err := execution.Proof(uint32(proofIndex))
		if err != nil {
			return err
		}
		siblings := make([]string, len(proof.Siblings))
		for i, digest := range proof.Siblings {
			siblings[i] = hex.EncodeToString(digest[:])
		}
		response["proof"] = map[string]interface{}{
			"index":    proofIndex,
			"state":    state,
			"siblings": siblings,
			"valid":    vm.VerifyProof(execution.Root(), state, uint32(proofIndex), execution.Steps()+1, proof),
		}
	}
	encoder := json.NewEncoder(w)
	encoder.SetIndent("", "  ")
	return encoder.Encode(response)
}

func main() {
	job := flag.String("job", "", "public JSON task file")
	proof := flag.Int("proof", -1, "include Merkle proof for trace state at this index")
	claim := flag.Bool("claim", false, "export endpoint claim for independent monitoring")
	review := flag.String("review", "", "review a worker claim exported by prisma-vm")
	flag.Parse()
	if *job == "" {
		fmt.Fprintln(os.Stderr, "usage: prisma-vm -job public-task.json [-proof state-index] [-claim] [-review worker-result.json]")
		os.Exit(2)
	}
	if err := runWithOptions(os.Stdout, *job, *proof, *claim, *review); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
