# REQUANTIZE_WIDE_V1 — exact wide requantization

```
v = clamp( round_ties_even(input * mult >> shift), clamp_lo, clamp_hi )
```

- `input` is INT64_ACCUM (the GEMM accumulator), Q12.20 or A13; the
  output dtype is declared by the node descriptor (Q12.20 or A13).
- Rounding is `RShiftRoundEven64`: arithmetic shift with round-to-nearest,
  ties-to-even; identical bits in Go, Python and the GPU kernel.
- `mult` and `shift` are descriptor parameters with admission bounds
  (`0 < mult <= 2^62`, `0 <= shift <= 62`, `clamp_lo <= clamp_hi`); an
  intermediate product that would exceed signed int64 is an ADMISSION
  FAILURE (the graph is rejected), never a silent widening / big.Int
  fallback.
- For the frozen A13W10 profile every real requant link is provably
  int64-safe: worst product 55 signed bits (F.5B theoretical bound,
  recomputed per node), well inside the 62-bit guard asserted by the
  reference and GPU implementations.
- The clamp bounds must lie inside the target dtype's logical range
  (A13 requant outputs cannot exceed [-4096, 4095]).
- Real block usage: 126 requant nodes = 83 GEMM-output requants (int64
  -> Q12.20 or A13) + 43 quantizer requants (Q12.20 -> A13), all with
  ties-to-even rounding; no saturation is ever relied on for correctness.
- The GPU execution of the requant path is exact integer arithmetic
  (validated in the F.5B.2 full-block run: all 126 nodes bit-identical
  to the CPU manifest on two architectures).

Adversarial vectors (ties, saturation, overflow rejection, min/max
extremes) live in `testdata/f5b2_adversarial_vectors.json` and the Go/Python
suites.
