# F.5C Wide GEMM dispute wire (roadmap A2-01)

Decision: **additive proto messages** in the existing Msg service
(`chain/proto/prisma/compute/v1/tx.proto`), no new service, no envelope
overloading.  Generated with protoc 29.3 + protoc-gogofaster; because the
toolchain version differs from the original generation, `tx.pb.go` is
rewritten in full — the wire-compatibility check is mechanical: every
pre-existing `protobuf:"..."` tag is byte-identical (227 old tags, diff
adds only), and the full chain test suite (all V1 messages) stays green.

Messages (all bind task + node + geometry so replay across
tasks/nodes/epochs is impossible):

| message | binds | notes |
|---|---|---|
| `MsgOpenWideGEMMDispute` | task, node_id, tile_i/j, bond | node_id must equal the graph bisection's first divergent node and its operator must be `GEMM_A13W10_I64_V1`; the descriptor fixes M/N/K/transpose_b. |
| `MsgWideTraceClaim` | party, trace_root, endpoint states (BE64) + proofs | S0 must be the zero state; the two final states must differ; trace length = ceil(K/8)+1 derived from the descriptor. |
| `MsgWideMidPoint` | party, state (BE64), proof, epoch | midpoint step is derived from the locked interval; epoch is the block-height round clock. |
| `MsgArbitrateWide512` | both claimed next states + proofs, 8+8 operand row chunks (`GraphChunkEvidence`) | the chain extracts the 8x8 A/W tiles from the *proven* chunks (never from caller-supplied tiles), then recomputes exactly 512 MAC. |

Replay protection: every handler requires the wide dispute record bound to
the task in the right phase; claims are one-shot per party; midpoints are
bounded by the interval-derived index and the round window; the arbiter
requires the arb-ready state.  Old GEMM v0.1.1 messages can never enter
this path (different task namespace and operator checks).
