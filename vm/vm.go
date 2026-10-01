// Package vm defines Prisma's bounded, deterministic integer execution format.
// It deliberately has no jumps, clock, floating point, I/O, or host calls.
package vm

import (
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"math/big"
)

const (
	SpecVersion     = 1
	RegisterCount   = 16
	MaxInputValues  = 64
	MaxInstructions = 4096
)

// Hash is a SHA-256 digest. All VM hashes use distinct domain prefixes.
type Hash [32]byte

type Op uint8

const (
	OpSet    Op = iota + 1 // dst = imm
	OpInput                // dst = input[imm]
	OpAdd                  // dst = a + b; overflow is an execution error
	OpSub                  // dst = a - b; overflow is an execution error
	OpMul                  // dst = a * b; overflow is an execution error
	OpDiv                  // dst = a / b; truncates toward zero
	OpMulDiv               // dst = trunc((a * b) / imm); 128-bit intermediate, imm > 0
	OpMax                  // dst = max(a, b)
	OpGT                   // dst = 1 if a > b, else 0
	OpSelect               // dst = b if a != 0, else c
)

// Instruction is the canonical v1 instruction encoding. Unused fields must be
// zero. Dst, A, B, and C are register indexes; Imm is a signed operand.
type Instruction struct {
	Op           Op
	Dst, A, B, C uint8
	Imm          int64
}

// Program owns a private copy of its code. A versioned hash commits every
// instruction, including its operands, before work can be assigned.
type Program struct {
	code   []Instruction
	digest Hash
	inputs int
}

func NewProgram(code []Instruction) (Program, error) {
	if len(code) == 0 || len(code) > MaxInstructions {
		return Program{}, fmt.Errorf("instruction count must be 1..%d", MaxInstructions)
	}
	p := Program{code: append([]Instruction(nil), code...)}
	h := sha256.New()
	h.Write([]byte("prismavm:program:v1\x00"))
	var count [4]byte
	binary.BigEndian.PutUint32(count[:], uint32(len(code)))
	h.Write(count[:])
	for i, ins := range p.code {
		if err := validateInstruction(ins); err != nil {
			return Program{}, fmt.Errorf("instruction %d: %w", i, err)
		}
		if ins.Op == OpInput && int(ins.Imm)+1 > p.inputs {
			p.inputs = int(ins.Imm) + 1
		}
		var enc [13]byte
		enc[0], enc[1], enc[2], enc[3], enc[4] = byte(ins.Op), ins.Dst, ins.A, ins.B, ins.C
		binary.BigEndian.PutUint64(enc[5:], uint64(ins.Imm))
		h.Write(enc[:])
	}
	copy(p.digest[:], h.Sum(nil))
	return p, nil
}

func validateInstruction(i Instruction) error {
	if i.Dst >= RegisterCount {
		return errors.New("destination register out of range")
	}
	switch i.Op {
	case OpSet:
		if i.A != 0 || i.B != 0 || i.C != 0 {
			return errors.New("nonzero unused operand")
		}
	case OpInput:
		if i.A != 0 || i.B != 0 || i.C != 0 || i.Imm < 0 || i.Imm >= MaxInputValues {
			return errors.New("invalid input instruction")
		}
	case OpAdd, OpSub, OpMul, OpDiv, OpMax, OpGT, OpMulDiv:
		if i.A >= RegisterCount || i.B >= RegisterCount || i.C != 0 {
			return errors.New("invalid register operand")
		}
		if i.Op == OpMulDiv {
			if i.Imm <= 0 {
				return errors.New("muldiv divisor must be positive")
			}
		} else if i.Imm != 0 {
			return errors.New("nonzero unused immediate")
		}
	case OpSelect:
		if i.A >= RegisterCount || i.B >= RegisterCount || i.C >= RegisterCount || i.Imm != 0 {
			return errors.New("invalid select operand")
		}
	default:
		return errors.New("unknown opcode")
	}
	return nil
}

func (p Program) Len() int            { return len(p.code) }
func (p Program) Digest() Hash        { return p.digest }
func (p Program) RequiredInputs() int { return p.inputs }
func (p Program) Instruction(n int) (Instruction, bool) {
	if n < 0 || n >= len(p.code) {
		return Instruction{}, false
	}
	return p.code[n], true
}

// State contains the entire bounded VM memory. Register 0 is the final output.
// The state at trace index n must have PC == n.
type State struct {
	PC  uint32
	Reg [RegisterCount]int64
}

func validateInput(p Program, input []int64) error {
	if len(p.code) == 0 {
		return errors.New("uninitialized program")
	}
	if len(input) != p.inputs {
		return fmt.Errorf("expected %d input values, got %d", p.inputs, len(input))
	}
	return nil
}

// Step evaluates exactly one instruction. Its return value is a new state;
// overflow, division by zero, and invalid inputs cannot be accepted as steps.
func Step(p Program, input []int64, before State) (State, error) {
	if err := validateInput(p, input); err != nil {
		return State{}, err
	}
	if before.PC >= uint32(len(p.code)) {
		return State{}, errors.New("program already complete")
	}
	i := p.code[before.PC]
	a, b := before.Reg[i.A], before.Reg[i.B]
	var value int64
	switch i.Op {
	case OpSet:
		value = i.Imm
	case OpInput:
		value = input[i.Imm]
	case OpAdd:
		if (b > 0 && a > math.MaxInt64-b) || (b < 0 && a < math.MinInt64-b) {
			return State{}, errors.New("addition overflow")
		}
		value = a + b
	case OpSub:
		if (b < 0 && a > math.MaxInt64+b) || (b > 0 && a < math.MinInt64+b) {
			return State{}, errors.New("subtraction overflow")
		}
		value = a - b
	case OpMul:
		if a == math.MinInt64 && b == -1 || b == math.MinInt64 && a == -1 {
			return State{}, errors.New("multiplication overflow")
		}
		value = a * b
		if a != 0 && value/a != b {
			return State{}, errors.New("multiplication overflow")
		}
	case OpDiv:
		if b == 0 {
			return State{}, errors.New("division by zero")
		}
		if a == math.MinInt64 && b == -1 {
			return State{}, errors.New("division overflow")
		}
		value = a / b
	case OpMulDiv:
		var x, y, divisor big.Int
		x.SetInt64(a)
		y.SetInt64(b)
		divisor.SetInt64(i.Imm)
		x.Mul(&x, &y).Quo(&x, &divisor)
		if !x.IsInt64() {
			return State{}, errors.New("muldiv overflow")
		}
		value = x.Int64()
	case OpMax:
		value = a
		if b > a {
			value = b
		}
	case OpGT:
		if a > b {
			value = 1
		}
	case OpSelect:
		value = before.Reg[i.C]
		if a != 0 {
			value = b
		}
	default:
		return State{}, errors.New("invalid opcode")
	}
	after := before
	after.Reg[i.Dst] = value
	after.PC++
	return after, nil
}

// VerifyStep checks a claimed transition using only one bounded VM step.
func VerifyStep(p Program, input []int64, before, claimedAfter State) error {
	after, err := Step(p, input, before)
	if err != nil {
		return err
	}
	if after != claimedAfter {
		return errors.New("claimed state does not match instruction result")
	}
	return nil
}
