package vm

import (
	"crypto/sha256"
	"encoding/binary"
	"errors"
)

// InputDigest commits the ordered, public int64 input vector. The caller must
// make the values available to challengers for the whole dispute window.
func InputDigest(input []int64) Hash {
	h := sha256.New()
	h.Write([]byte("prismavm:input:v1\x00"))
	var enc [8]byte
	binary.BigEndian.PutUint32(enc[:4], uint32(len(input)))
	h.Write(enc[:4])
	for _, value := range input {
		binary.BigEndian.PutUint64(enc[:], uint64(value))
		h.Write(enc[:])
	}
	var digest Hash
	copy(digest[:], h.Sum(nil))
	return digest
}

func leafHash(state State) Hash {
	var enc [1 + 4 + RegisterCount*8]byte
	enc[0] = 0
	binary.BigEndian.PutUint32(enc[1:5], state.PC)
	for i, value := range state.Reg {
		binary.BigEndian.PutUint64(enc[5+i*8:], uint64(value))
	}
	return sha256.Sum256(enc[:])
}

func paddingHash(index uint32) Hash {
	var enc [5]byte
	enc[0] = 2
	binary.BigEndian.PutUint32(enc[1:], index)
	return sha256.Sum256(enc[:])
}

func nodeHash(left, right Hash) Hash {
	var enc [65]byte
	enc[0] = 1
	copy(enc[1:33], left[:])
	copy(enc[33:], right[:])
	return sha256.Sum256(enc[:])
}

// Proof authenticates a state at a trace index. The tree shape is determined
// by the program length: states 0..N, padded to the next power of two.
type Proof struct {
	Siblings []Hash
}

// Execution owns one complete replay and the Merkle tree over all N+1 states.
type Execution struct {
	states []State
	levels [][]Hash
}

func Execute(p Program, input []int64) (Execution, error) {
	if err := validateInput(p, input); err != nil {
		return Execution{}, err
	}
	states := make([]State, len(p.code)+1)
	for i := range p.code {
		state, err := Step(p, input, states[i])
		if err != nil {
			return Execution{}, err
		}
		states[i+1] = state
	}
	return newExecution(states), nil
}

func newExecution(states []State) Execution {
	count := 1
	for count < len(states) {
		count <<= 1
	}
	leaves := make([]Hash, count)
	for i := range leaves {
		if i < len(states) {
			leaves[i] = leafHash(states[i])
		} else {
			leaves[i] = paddingHash(uint32(i))
		}
	}
	levels := [][]Hash{leaves}
	for len(levels[len(levels)-1]) > 1 {
		current := levels[len(levels)-1]
		parent := make([]Hash, len(current)/2)
		for i := range parent {
			parent[i] = nodeHash(current[2*i], current[2*i+1])
		}
		levels = append(levels, parent)
	}
	return Execution{states: append([]State(nil), states...), levels: levels}
}

func (e Execution) Root() Hash {
	if len(e.levels) == 0 {
		return Hash{}
	}
	return e.levels[len(e.levels)-1][0]
}

func (e Execution) Steps() uint32 {
	if len(e.states) == 0 {
		return 0
	}
	return uint32(len(e.states) - 1)
}

func (e Execution) Final() State {
	if len(e.states) == 0 {
		return State{}
	}
	return e.states[len(e.states)-1]
}

func (e Execution) Output() int64 { return e.Final().Reg[0] }

func (e Execution) Proof(index uint32) (State, Proof, error) {
	if index >= uint32(len(e.states)) {
		return State{}, Proof{}, errors.New("trace index out of range")
	}
	position := int(index)
	proof := Proof{Siblings: make([]Hash, 0, len(e.levels)-1)}
	for level := 0; level+1 < len(e.levels); level++ {
		proof.Siblings = append(proof.Siblings, e.levels[level][position^1])
		position >>= 1
	}
	return e.states[index], proof, nil
}

func VerifyProof(root Hash, state State, index, stateCount uint32, proof Proof) bool {
	if stateCount == 0 || stateCount > MaxInstructions+1 || index >= stateCount || state.PC != index {
		return false
	}
	width, depth := uint32(1), 0
	for width < stateCount {
		width <<= 1
		depth++
	}
	if len(proof.Siblings) != depth {
		return false
	}
	digest := leafHash(state)
	position := index
	for level, sibling := range proof.Siblings {
		// The proof path for the final state exposes the whole padded suffix.
		// Enforce its canonical hashes so one trace has one commitment root.
		start := ((index >> level) ^ 1) << level
		if start >= stateCount && sibling != paddingSubtreeRoot(start, uint32(1)<<level) {
			return false
		}
		if position&1 == 0 {
			digest = nodeHash(digest, sibling)
		} else {
			digest = nodeHash(sibling, digest)
		}
		position >>= 1
	}
	return digest == root
}

func paddingSubtreeRoot(start, width uint32) Hash {
	if width == 1 {
		return paddingHash(start)
	}
	half := width / 2
	return nodeHash(paddingSubtreeRoot(start, half), paddingSubtreeRoot(start+half, half))
}
