# Job lifecycle

The job state machine is fixed by the roadmap. The mapping from the chain's
task statuses is documented and conservative: an unknown chain state is
surfaced as `unknown`, never renamed into something friendlier.

```text
queued ──▶ assigned ──▶ accepted ──▶ executing ──▶ committed
   │            (scheduler)   (chain accept tx)      │
   │ (cancel, pre-accept only)                       ▼
   │                                        availability_ready
   ▼                                                 │  (2-of-3 DA quorum verified)
cancelled                                            ▼
                                               verifying ──▶ finalized ──▶ receipt/VWR
                                               (dispute)  │
                                                          ├──▶ fraud (challenger wins)
                                                          └──▶ refunded (availability failed)
```

| job state | chain state | who moves it |
|---|---|---|
| queued | `posted` | the developer (submit) |
| assigned | scheduler assignment | the Job API only (no chain effect) |
| accepted | `accepted` | the worker's accept transaction |
| executing | (worker-local) | the worker |
| committed | `result_submitted` | the worker's CommitV3 |
| availability_ready | `availability_pending` → `challenge_window_ready` | the DA providers' attestations |
| verifying | `challenged` | a watcher's challenge |
| finalized | `finalized` / `worker_wins` | the chain (settlement + VWR) |
| fraud | `challenger_wins` / `fraud` | the chain (deterministic dispute) |
| refunded | `refunded` / `availability_failed` | the chain |

Rules the code enforces:

- **cancel** is only possible pre-accept (`409 job_not_cancellable` later);
- **reassignment** is possible while `pending/assigned/accepted/running` on
  worker failure; from `committed` on the chain's dispute path owns everything
  (a scheduler must never reassign — it would double-pay);
- **verification status** in every presentation is a passthrough of what the
  chain confirmed; the API never says "verified" ahead of the chain.

# Verified Work Receipt

A VWR is the chain's settlement object: it exists only after the commit was
availability-checked (2-of-3 verified DA attestations), the challenge window
passed (or a dispute resolved), and the chain executed the settlement. It
binds the task, the worker, the GraphIDV2, the final output root and the
settlement transaction. A client should treat `verification.confirmed == true`
in the job view as the moment a VWR exists — and nothing before that.
