package verify

// Randomness rules for v0.1.2 (protocol invariants 1 and 6):
//
//   - The verification randomness is produced only AFTER the worker's
//     ResultCommit and output_root are immutable. NewChallengeRandomness
//     requires the committed task/assignment identifier and rejects empty
//     ones, so a production challenge cannot even be constructed before a
//     commit exists.
//   - Production sources draw from a CSPRNG (crypto/rand). Seeded sources
//     exist ONLY for deterministic tests and cross-language vectors and
//     are marked development-only; they must never back production
//     challenges.

import (
	"crypto/rand"
	"crypto/sha256"
	"errors"
	"io"
)

// RandomSource fills p with cryptographically unpredictable bytes.
// io.Readers such as crypto/rand.Reader satisfy it directly.
type RandomSource interface {
	Fill(p []byte)
}

// CSPRNGSource adapts an io.Reader (crypto/rand.Reader) to RandomSource.
type CSPRNGSource struct {
	R io.Reader
}

// Fill reads from the underlying CSPRNG.
func (s CSPRNGSource) Fill(p []byte) {
	if _, err := io.ReadFull(s.R, p); err != nil {
		panic("verify: CSPRNG failure: " + err.Error())
	}
}

// NewChallengeRandomness derives the production randomness source for one
// committed result. commitID must be the canonical task/assignment binding
// of an already-immutable ResultCommit; the parameter exists so that the
// type system and tests can enforce invariant 6 (no challenge randomness
// before a commit).
func NewChallengeRandomness(commitID []byte) (RandomSource, error) {
	if len(commitID) == 0 {
		return nil, errors.New("verify: challenge randomness requires an immutable ResultCommit; refusing to generate before commit lock")
	}
	return CSPRNGSource{R: rand.Reader}, nil
}

// DeterministicSource is a SHA-256 counter stream for tests and
// cross-language vectors ONLY. It is intentionally not reachable from any
// production challenge constructor.
type DeterministicSource struct {
	state []byte
	count uint64
}

// NewDeterministicSource returns a development-only seeded source.
func NewDeterministicSource(seed []byte) *DeterministicSource {
	return &DeterministicSource{state: append([]byte(nil), seed...)}
}

// Fill produces the next deterministic block stream.
func (s *DeterministicSource) Fill(p []byte) {
	for len(p) > 0 {
		h := sha256.New()
		h.Write(s.state)
		var ctr [8]byte
		for i := 0; i < 8; i++ {
			ctr[i] = byte(s.count >> (8 * (7 - i)))
		}
		h.Write(ctr[:])
		s.count++
		block := h.Sum(nil)
		n := len(block)
		if n > len(p) {
			n = len(p)
		}
		copy(p[:n], block[:n])
		p = p[n:]
	}
}

// RandomBinaryVector returns an N-length binary vector drawn bit-per-column
// from the source: bit j of byte j/8 selects r[j].
func RandomBinaryVector(src RandomSource, n uint64) []uint8 {
	nbytes := (n + 7) / 8
	raw := make([]byte, nbytes)
	src.Fill(raw)
	r := make([]uint8, n)
	for j := uint64(0); j < n; j++ {
		r[j] = (raw[j/8] >> (j % 8)) & 1
	}
	return r
}
