# Canonical graph v1 (CANONICAL_GRAPH_V1)

A canonical graph is a static, hashable DAG of canonical operators. It has
no loops, branches, host callbacks, randomness or I/O; every tensor shape,
parameter and commitment is fixed before execution. The graph is the unit
of verification: normal-path workers commit output roots only, and the
per-node state trail exists only on demand inside a dispute.

Implementation: `compute/canonical/graph.go`, `graphdispute.go`,
`graphsnapshot.go`, `graphcommit.go`, `graphreceipt.go`; Python mirror in
`compute/canonical/python/canonical_ref.py`.

## Objects and identities

* `GraphDescriptor { protocol_version, spec, inputs[], nodes[], outputs[] }`
  with `GraphInput { name, desc, root }`, `GraphNode { node_id,
  operator_id, operator_version, inputs[], output, params[] }`.
* `GraphID = SHA256("PRISMA_CANONICAL_GRAPH_V1\0" || CanonicalCBOR(descriptor))`
  — binds operators, versions, weight roots, shapes, constants and
  parameters. The chain validates a posted descriptor structurally
  (densely ordered nodes, registered operators, versions, recomputed
  output descriptors — declared outputs are never trusted), then derives
  the same id.
* Tensor commitments: `TensorRoot = SHA256(domain || CBOR(desc) || merkle)`
  over 64-element chunks of big-endian int32 values, zero padding, odd
  nodes self-pairing. The descriptor binds dtype/layout/shape, so bytes
  can never be reinterpreted.
* `GraphStateRoot`: Merkle root over the sorted live `(kind,index) →
  TensorRoot` leaves (domain-separated). The state after node k contains
  all inputs plus every node output up to k; blocks branch, so the state
  covers the whole live set, not just the last tensor.
* `GraphResultCommit` (output-first): graph id, task/assignment
  references, worker key, per-output roots, final output root,
  completed epoch, and the worker's domain-separated Ed25519 signature.
  No trace of any kind leaves the worker on the normal path.
* `VerifiedGraphWorkReceiptV1`: settlement artifact of a finalized task
  (graph id, spec, references, worker key, output roots, final root,
  derived work vector, verification mode, finalized epoch, settlement
  reference, optional dispute transcript digest). It carries no signature
  of its own: the chain derives it from the validated commitment, exactly
  like the frozen gemmv1 receipt. The old `gemmv1.VerifiedWorkReceipt` is
  untouched.

## Execution and trail

`ExecuteGraph` verifies every input against its committed root, runs the
registry operators in order, and produces the per-node state trail
(`len(nodes)+1` roots). The trail is NEVER committed on chain; both
parties regenerate it on demand from the same inputs and serialize only
the trail Merkle root plus endpoint proofs (`GraphTrailClaim`).

## Dispute

1. `GraphDispute` locks both trail claims (both trails must start in the
   same state — the chain additionally requires that this state equals
   the descriptor-derived input state, so a party cannot mislabel it —
   and must end in different states).
2. Bisection submits midpoint state roots (chunk-proof style inclusion
   against the party's own trail root) until the interval collapses to
   the FIRST divergent node.
3. `ArbitrateNodeChunk` dispatches to the operator arbiter
   (docs/canonical-operators-v1.md). Evidence must be leaves of the
   committed input state; on chain the keeper verifies the tensor-leaf
   membership against the dispute's low state root with the leaf index
   derived from the graph geometry, and re-derives the RoPE table from
   its committed input root before any use.
4. Verdicts settle through the same economics as the GEMM path:
   `ChallengerWins` refunds the requester and slashes the worker, a
   failed challenge burns the challenger bond, and the worker keeps
   settling after surviving (`challenged_worker_won`).

Persistence: `GDS1` — one fixed binary snapshot (locked trail roots,
interval, midpoint submissions, round clock) restored by re-supplying the
descriptor and round period; the stored graph id is validated. A restored
session continues to the same first-divergent node (test:
`TestGraphSnapshotRoundTripAndContinuation`).

## Chain integration

Eight messages in their own storage namespace and status machine, reusing
the GEMM escrow/bond/window/queue mechanics: `PostGraphTask`,
`AcceptGraphTask`, `SubmitGraphResult`, `OpenGraphChallenge`,
`GraphTrailClaim`, `GraphMidPoint`, `ArbitrateGraphNode`,
`FinalizeGraphTask`. Admission bounds are checked BEFORE structural
validation; exactly one active challenge per task; midpoints are clocked
by block height and bounded by the chain clock.

Measured on the four-validator devnet (`docs/phase-f-e2e-results.json`,
`docs/phase-f-gas-results.json`): a 90-node / 23-input mini block settled
honestly with a receipt; a corrupted ADD output was localized to node 89
in seven bisection rounds and adjudicated permissionlessly to
`challenger_wins` with the challenger's ledger matching the expected
accounting to the unit and no receipt for the worker; a false challenge
was refused at admission; all four validators converged on one app hash.
Measured gas: post 4.52M, accept 1.28M, submit 1.39M, challenge 1.32M,
trail claim 0.23-0.27M, midpoint 0.258M, arbitration 1.38M, finalize
1.38M (GraphGasV1 schedule in `chain/x/compute/graph_gas.go`).

## Honest limits

* Bisection and arbitration are exercised on the mini block; the medium
  block and long-horizon multi-operator runs are NOT TESTED on chain.
* No graph-task query protos exist yet (the E2E reads state through the
  ABCI store query); operator arbitration of ROPE nodes is implemented
  and unit-tested but not driven end to end on the devnet.
* One active challenge per task (the Phase E one-open-challenge bound)
  replaces the GEMM challenge queue for graphs.
