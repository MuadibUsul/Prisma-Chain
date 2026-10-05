# B0-01 — Repository Reconciliation

```text
Task ID            B0-01
Status             COMPLETE (one item surfaced as NEEDS_REVIEW, parked; see §7)
Executed           2026-10-05
Working branch     protocol/gemm-v0.1.2-freivalds
Base commit        4d4a4189708ab3644d8a800fa620f455c3fb4583
Transition branch  feature/public-node-admission @ 520abb158119c9d44cdcd907a50ed360836de73d
Roadmap authority  docs/roadmap/Prisma_Chain_AI_Engineering_Roadmap_v1.1.md (§B0-01)
```

Goal (roadmap v1.1): inventory, classify and reconcile every pre-existing
working-tree change so the productization baseline starts from a clean,
fully-understood tree.  Nothing was deleted; nothing was edited; no `git
add .` was used at any point.

---

## 1. Mechanical inventory (B0-01a)

All commands run in the main worktree `E:\Codes\prisma` on 2026-10-05.

```text
$ git status --porcelain=v1 -b
## protocol/gemm-v0.1.2-freivalds...origin/protocol/gemm-v0.1.2-freivalds [ahead 1]
 M network/README.md
 M network/prisma_network/core.py
 M network/prisma_network/server.py
?? LICENSE
?? Prisma_Chain_AI_Engineering_Roadmap_F5C_to_Public_Testnet_Alpha.md
?? network/prisma_network/node_http.py
?? network/tests/test_public_nodes.py

$ git diff --stat
 network/README.md                | 15 ++++++++++++---
 network/prisma_network/core.py   | 27 +++++++++++++++++++++++++-
 network/prisma_network/server.py | 41 +++++++++++++++++++++++++++++++++-------
 3 files changed, 72 insertions(+), 11 deletions(-)

$ git ls-files --others --exclude-standard
LICENSE
Prisma_Chain_AI_Engineering_Roadmap_F5C_to_Public_Testnet_Alpha.md
network/prisma_network/node_http.py
network/tests/test_public_nodes.py

$ git rev-parse HEAD        -> 4d4a4189708ab3644d8a800fa620f455c3fb4583
$ git stash list            -> (empty)
$ git branch --show-current -> protocol/gemm-v0.1.2-freivalds
```

`git fetch --all --tags` was **attempted and failed**:

```text
fatal: unable to access 'https://github.com/MuadibUsul/Prisma-Chain.git/':
Could not resolve host: github.com
```

Consequence: remote refs were **NOT re-verified against the remote this
session**.  All ref facts below come from local objects and local
remote-tracking refs, which were last synchronised by the previous
session (the freeze tag was pushed there).  Re-run the fetch before
relying on remote state.

Four entries is the complete set of untracked files; three modified files;
no staged changes; no stashes.  This matches the known starting point
recorded in roadmap v1.1 §B0-01 (branch and HEAD confirmed as
`protocol/gemm-v0.1.2-freivalds`; HEAD is now `4d4a418` instead of the
roadmap's noted `e550cbb`, because `4d4a418` — the roadmap migration
itself — was committed on top after that text was written).

The main worktree is one of 13 linked worktrees (phase worktrees under
`E:\Codes\Prisma-Chain-phase-*`).  Only the main worktree was in B0-01
scope; the others are freeze/phase evidence locations and were untouched.

---

## 2. Classification (B0-01b)

Class definitions used here — the class is the verdict on the *content's*
fate; `disposition` records where the bytes live now:

```text
KEEP          content is current and belongs in the productization
              baseline unchanged
MIGRATE       valuable content whose home is a product line, not the
              protocol branch it currently sits on
OBSOLETE      superseded; content preserved elsewhere
UNRELATED     not protocol and not product; must not enter the baseline
NEEDS_REVIEW  genuinely ambiguous; question surfaced, item parked
```

| # | path | pre-state | class | disposition | reason |
|---|---|---|---|---|---|
| 1 | `network/README.md` | M (+12/−3) | MIGRATE | commit `57be2af` on `feature/public-node-admission` | Documents the opt-in public node admission operator contract (`PRISMA_PUBLIC_NODE_ADMISSION`, HTTPS-only, funded tasks). Product-line content, not protocol. |
| 2 | `network/prisma_network/core.py` | M (+26/−1) | MIGRATE | commit `57be2af` | `ControlPlane(public_node_admission=…)`: rejects non-global IP literals, validates public hostnames, requires `/v1/` paths, forbids dev-HTTP combination. |
| 3 | `network/prisma_network/server.py` | M (+34/−7) | MIGRATE | commit `57be2af` | Gateway wiring: env gate, protected transport + funded-task coupling, `PublicNodeHTTP` for probes/worker calls/receipts. |
| 4 | `network/prisma_network/node_http.py` | ?? (new, 52 lines) | MIGRATE | commit `57be2af` | `PublicNodeHTTP`: per-request DNS pinning to the checked global address, TLS SNI/Host preserved, proxy env and redirects disabled, response size bounded. |
| 5 | `network/tests/test_public_nodes.py` | ?? (new, 81 lines) | MIGRATE | commit `57be2af` | 7 tests covering private-literal/dev-HTTP rejection, DNS pinning + SNI, multi-answer private rejection, response bound. All pass (§8). |
| 6 | `LICENSE` | ?? (new, 202 lines) | KEEP | commit `0f8ebd2` on `feature/public-node-admission` (parked) | Apache License 2.0 at repo root. Repo-level hygiene, no feature content. Parked because B0-01 forbids content commits outside `docs/productization/**` on the protocol branch; B0-02 must carry it into the product baseline. |
| 7 | `Prisma_Chain_AI_Engineering_Roadmap_F5C_to_Public_Testnet_Alpha.md` | ?? (root, 2114 lines) | NEEDS_REVIEW | commit `520abb1` on `feature/public-node-admission` (parked) | Redundant root duplicate of the tracked historical copy; deleting vs keeping is the user's call — not guessed. Details in §7. |

Every entry of the mechanical inventory is classified; entry counts match
(3 modified + 4 untracked = 7 rows).

Items 1–5 are one coherent feature (public node admission), read in full
before classification; item 6 is unrelated to that feature; item 7 is a
duplicate artifact.

---

## 3. Conflict check (B0-01c)

Each item checked against the three references named by the roadmap:

**a. Frozen protocol (roadmap §6 Protocol Freeze Boundary).**  No entry is
in the frozen object list (CANONICAL_TENSOR_V2, CANONICAL_GRAPH_V2,
GEMM_A13W10_I64_V1, REQUANTIZE_WIDE_V1, FREIVALDS_A13W10_I64_V1,
GraphDisputeV2, WIDE_GEMM_DISPUTE_V1, CommitV3, VWR V3, GraphIDV2,
PolicyID, V1 protocols).  `network/` is explicitly *product software*
under roadmap §7 (code layering: "worker / watcher / da / api / scheduler
/ cli / ops — product software; free to evolve as long as frozen
semantics are consumed as-is").

Mechanical checks:

```text
$ git diff --stat 64742af..89547fa -- network/     -> (empty; the frozen
    lineage never touched network/)
$ git log --all --oneline -- network/prisma_network/node_http.py \
                              network/tests/test_public_nodes.py
                                                   -> (empty; no branch
    anywhere carries this WIP)
$ git log 64742af..89547fa -- LICENSE              -> (empty)
```

Verdict: **NO CONFLICT** for items 1–7.

**b. Phase B roadmap (v1.1).**  The public-node WIP is not yet owned by
any B/C/D task.  Roadmap §0 recorded it only as uncommitted
public-client WIP to be reconciled; nothing in v1.1 supersedes or forbids
it.  Verdict: no conflict; its integration point is an open decision
(§9).

**c. Current public network architecture.**  The WIP *is* the transport
admission boundary described in `network/README.md`; it is self-contained
(no other module depends on it in this tree) and its tests are green.
Verdict: no conflict.

**d. Roadmap bookkeeping documents.**  Item 7's content asserts
`PHASE_F = PARTIAL` as current, which roadmap v1.1 declares obsolete.  It
is out of every working tree after this task (parked in history only), so
it cannot be read as current state where it stands; if the user chooses
"keep", it must not be restored as an un-bannered file at the repo root
(§7).  Verdict: NEEDS_REVIEW, parked.

---

## 4. Repository topology (recorded for B0-02)

```text
merge-base(protocol/gemm-v0.1.2-freivalds, freeze commit) = 64742af8
commits unique to the current branch: e550cbb, 4d4a418
freeze lineage is 98 commits ahead of the merge-base
freeze commit is NOT an ancestor of the current branch
```

Local ref verification against roadmap §0 (remote not reachable, see §1):

```text
git rev-parse transformer-phase-f-wide-integer          -> 330d27f5fc92d009aca9599cc1e0bb13a496c766  MATCH (tag object)
git rev-parse transformer-phase-f-wide-integer^{commit} -> 89547fa97b62d72a4cd336a5dfd3b000b03eac60  MATCH
git rev-parse protocol/transformer-phase-f5c-wide-integer -> 40b4fe353df7d2203d1609cceb10d24380abed7a MATCH
origin/main (local remote-tracking)                     -> a01c5ed (freeze not merged, as §0 states)
```

The single-command enumeration of all post-tag material for B0-02:

```text
$ git log --oneline 89547fa..feature/public-node-admission
520abb1 transition: park redundant root copy of historical v1.0 roadmap (B0-01)
0f8ebd2 transition: park Apache-2.0 LICENSE (B0-01)
57be2af transition: park public-node admission WIP (B0-01)
4d4a418 docs: migrate engineering roadmap to productization v1.1
e550cbb Fix network CI by declaring tokenizers in the test extra
```

`main` was not touched.  Reconciling `main` with the frozen lineage
remains the explicit B0-02 decision per §0.

---

## 5. Actions taken

All actions were confined to git ref/commit mechanics plus
`docs/productization/**` (the B0-01 allowed set).  Exact sequence:

```text
1. git checkout -b feature/public-node-admission          # from 4d4a418
2. git add <the 5 network files, listed path-by-path>     # never `git add .`
   git commit  -> 57be2af  "transition: park public-node admission WIP (B0-01)"
3. git add LICENSE
   git commit  -> 0f8ebd2  "transition: park Apache-2.0 LICENSE (B0-01)"
4. git add Prisma_Chain_..._Alpha.md
   git commit  -> 520abb1  "transition: park redundant root copy of historical v1.0 roadmap (B0-01)"
5. git checkout protocol/gemm-v0.1.2-freivalds
```

Before each commit the staged diff was inspected and matched the
working-tree diff exactly (per-file insertions/deletions: `network/README.md`
+12/−3, `core.py` +26/−1, `server.py` +34/−7, `node_http.py` +52/−0, tests
+81/−0; total +205/−11) — no
whole-file diffs from line-ending conversion, no accidental content
changes.  `core.autocrlf=true` is set locally; it did not alter any
content (see §7 for the byte-level proof on the parked file).

After step 5: `git status --porcelain` empty,
`git ls-files --others --exclude-standard` empty,
`git diff --stat` empty.  The five network paths, `LICENSE` and the root
roadmap file are absent from the working tree and present in
`feature/public-node-admission`.

Nothing was pushed.  No roadmap file was modified (roadmap state
bookkeeping is B0-03's charter; B0-01's allowed changes exclude
`docs/roadmap/**`).  The running local devnet was not touched (B0-04).

---

## 6. Transition commits (B0-01e)

Branch `feature/public-node-admission`, base `4d4a418`, tip `520abb1`:

| commit | content | blob-level facts |
|---|---|---|
| `57be2af` | 5 network files (public-node admission WIP) | staged diff +205/−11; byte content unchanged from the working tree |
| `0f8ebd2` | `LICENSE` (Apache-2.0, 202 lines) | blob `d645695673349e3947e8e5ae42332d0ac3164cd7` |
| `520abb1` | root v1.0 roadmap duplicate (parked, NEEDS_REVIEW) | blob `a8aeb3d6b62ea53f87605b805aadcec33d270f3f`; `git show` equals the original file's sha256 (§7) |

Commit messages state what each item is and where it belongs, per
B0-01e.  The branch name follows the existing `feature/*` convention
(`feature/chain-bonded-admission` already exists); it is derived from the
dominant content (5 of 7 files).

---

## 7. NEEDS_REVIEW item surfaced (not guessed)

**Item:** `Prisma_Chain_AI_Engineering_Roadmap_F5C_to_Public_Testnet_Alpha.md` (repo root).

Mechanical facts:

```text
sha256 (original working-tree file)  = c318f309e60b36bc8d10657f2172a6a86961c7873c1570ab852d4ba5f690db47
sha256 (parked commit blob, via git show)  = c318f309... (identical, byte-for-byte)
diff against docs/roadmap/Prisma_Chain_AI_Engineering_Roadmap_v1.0_historical.md:
    identical except (a) the tracked copy prepends a 5-line
    "HISTORICAL / superseded" banner, and (b) line endings in the working
    tree (root = LF; docs copy = CRLF via core.autocrlf checkout).
```

Roadmap v1.1 §12 designates the committed `docs/roadmap/..._v1.0_historical.md`
as the v1.0 preservation point, which makes this root copy redundant; its
`PHASE_F = PARTIAL` text is obsolete.  But §12 also says v1.0 is
"preserved un-deleted", and the previous session deliberately left this
file untouched — so deleting it is the user's call, not the AI's.  Per
the B0-01 failure clause it was marked NEEDS_REVIEW, parked in a
transition commit with the item explicitly listed, and the working tree
kept clean around it.

**The question:** delete this root duplicate (recommended — the tracked
historical copy preserves the content and carries the superseded banner),
or keep the exact bytes recoverable/reinstated?

Until answered, recover with:

```text
git show feature/public-node-admission:Prisma_Chain_AI_Engineering_Roadmap_F5C_to_Public_Testnet_Alpha.md
```

---

## 8. Validation

| validation line (roadmap §B0-01) | evidence | verdict |
|---|---|---|
| every pre-existing modification classified | §2 table covers all 7 inventory entries; counts match exactly | TRUE |
| no accidental overwrite | staged-vs-working diffstat identity before every commit; byte-level sha256 proof for the parked file (§7); modified network files reverted to `4d4a418` state on checkout | TRUE |
| no unrelated WIP lost | all bytes preserved in `feature/public-node-admission`; fresh worktree checkout of `520abb1` runs the WIP's test suite green | TRUE |
| working tree clean | `git status --porcelain` empty; `git ls-files --others --exclude-standard` empty; `git diff` empty | TRUE |

Tests run:

```text
python -m pytest tests/test_public_nodes.py -q          # in network/, on the pre-commit working tree
  -> 7 passed in 1.03s
git worktree add --detach /tmp/b0-01-verify feature/public-node-admission   # fresh checkout of the parked tip
  -> 7 passed in 1.33s        (temporary worktree removed afterwards)
```

The only output besides passes was a `pytest-asyncio` deprecation warning
about an unset fixture loop scope; not a failure, pre-existing.

Deferred on purpose:

- Roadmap status bookkeeping (B0-01 → DONE, active_task → B0-02) is
  **not** done here: B0-03 owns it ("the bookkeeping completes after
  B0-01/B0-02 so it can record their result"), and B0-01's allowed
  changes exclude `docs/roadmap/**`.
- Push of `4d4a418` + this report commit: open user decision from the
  previous session; not performed.

---

## 9. Recommendations for B0-02 (recorded, non-binding)

1. Baseline from the freeze tag `89547fa` as planned; do not use this
   branch or `main` as ancestry.
2. Enumerate and carry post-tag material with recorded reasons, one item
   at a time: `e550cbb` (network CI test-extra fix; check it still
   applies), `4d4a418` (roadmap v1.1 + migration JSON), this report,
   `0f8ebd2` (LICENSE — carry to the product baseline and the published
   default branch unchanged), `57be2af` (public-node admission WIP — carry
   as its own commit and wire its tests into the baseline CI; the
   clean-venv CI install is the known trap that `e550cbb` fixed).
3. Resolve the `main` ↔ frozen-lineage decision (§0) explicitly and
   record the verdict; `origin/main` does not contain the freeze commit.
4. Give the public-node admission WIP an owning task on the product line
   so it does not linger unowned; it is the transport admission boundary
   public-testnet onboarding needs.
5. Do not merge `feature/public-node-admission` wholesale; cherry-pick
   per commit with recorded reasons (commit 3 is a NEEDS_REVIEW park and
   must not silently enter the baseline).
