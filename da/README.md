# prisma-da

Prisma Chain DA provider daemon (DA_REPLICA_V1). A DA provider stores the
canonical full output blob of an assignment, recomputes its `output_root`
**before** attesting anything, serves bulk retrieval and per-tile proofs, and
answers the chain's sampling challenges.

```text
prisma-da init                 # provider identity (same rules as the worker) + config
prisma-da register             # register/bond the provider against the chain
prisma-da run                  # serve the DA surface
prisma-da status               # local index + health of a running daemon
```

The frozen attestation/tile formats are **loaded from the frozen Python
libraries** (`compute/canonical/python`, `compute/gemmv1/python`) — never
re-implemented. `PRISMA_FROZEN_PYTHON` (colon-separated directories) points at
them when they are not in a repository checkout; the release ships them.
