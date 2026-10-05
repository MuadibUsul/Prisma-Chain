# B0-04 — Development Environment Cleanup

```text
Task ID     B0-04
Status      DONE (2026-10-05)
Branch      product/testnet-alpha
Scope       local development environment only; no repository content changes
Stop policy deploy/multivalidator/stop.sh (data preserved), per roadmap B0-04
```

Every action below was re-queried live at execution time (nothing is
copied from earlier notes).

## 1. Inventory (as found, 2026-10-05)

```text
docker compose project prisma-multivalidator   RUNNING (4 validators):
  prisma-multivalidator-validator-a-1   up ~57m   RPC 127.0.0.1:26661
  prisma-multivalidator-validator-b-1   up ~1h    RPC 127.0.0.1:26662
  prisma-multivalidator-validator-c-1   up ~1h    RPC 127.0.0.1:26663
  prisma-multivalidator-validator-d-1   up ~1h    RPC 127.0.0.1:26664
  prisma-multivalidator-init-1          Exited(1) leftover init container
live chain state before stop:
  chain_id prisma-mv-1, height 2120, catching_up false
DA replica ports 8401-8403   no listeners (stale replicas already gone)
stale prisma python drivers  none
temporary git worktrees      none (verification worktrees already removed)
```

Out of scope and deliberately NOT touched (other projects / unrelated):

```text
ancs-dev-redis, ancs-dev-db        (compose project "infra", E:\Codes\ANCS)
zhangji-db, gubugu-postgres, aeo-redis   (stopped, unrelated)
node.exe / python.exe / docker desktop helpers  (unrelated processes)
RunPod GPU pods                    user-managed; no API key exists here
```

## 2. Actions taken

```text
1. recorded live chain state: chain_id prisma-mv-1, height 2120, catching_up false
2. sh deploy/multivalidator/stop.sh
     -> docker compose -f deploy/multivalidator/compose.yaml down
     -> 4 validator containers removed, init-1 leftover removed with the project,
        network prisma-multivalidator_default removed
3. verified after stop:
     docker ps -a | grep prisma   -> empty
     netstat listeners 26661-26664 -> empty (ports released)
```

Bind-mounted data directories are **preserved** (`down` without `-v`; no
volume was deleted). The devnet was a development artifact; Phase C will
create its own clean topology.

## 3. Retained on purpose (not cleaned)

```text
prisma-chain:phase-f docker image           (frozen-lineage devnet image; rebuildable)
devnet data directories (bind mounts)       (F.5C devnet state; not evidence, kept anyway)
phase worktrees E:\Codes\Prisma-Chain-phase-* (hold freeze evidence: E2E JSONs, reports)
docs/phase-f5c-e2e-*.json, phase-f-freeze.json, reports, testdata/  (protocol evidence —
   never cleaned, per B0-04 forbidden changes)
```

## 4. Findings recorded (hygiene, not blockers)

1. **Test mutates a tracked artifact.** `chain/x/compute/graph_gas_measure_test.go`
   rewrites `docs/phase-f5c-gas-results.json` on every `go test ./...` run
   (deterministic content, but the rewrite churns line endings and dirties
   the tree mid-test-run; observed and reverted during B0-02).  Proposed
   owner: B1-01 test/build hygiene — make the evidence write opt-in
   (e.g. `PRISMA_WRITE_EVIDENCE=1`) or write to a temp path, without
   changing any measured value.  Not fixed here (B0-04 is environment-only).
2. **`prisma-multivalidator-init-1` Exited (1)** — expected: the init
   container runs once and exits non-zero when the devnet already exists;
   it is removed with `compose down`.  No action needed.
3. **`f5c_replay.exe` is a tracked Windows binary** in the frozen tree
   (root).  Not temporary (part of the tag), so not removed here; product
   packaging may decide to untrack it later (B1-01 scope).

## 5. Definition of Done

```text
environment state documented            yes (this file)
nothing referenced by committed reports/evidence deleted   yes (nothing deleted)
retained state written down for the next session           yes (section 3)
```
