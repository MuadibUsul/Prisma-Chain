# B0-05 — Productization Toolchain Baseline

```text
Task ID     B0-05
Status      DONE (2026-10-05)
Branch      product/testnet-alpha
Goal        pin the toolchain and target matrix Phase B components are
            built and tested against; every entry cites where it is
            enforced or tested; unsupported combinations are listed
            explicitly, never implied
```

Sources are repository files and live checks on 2026-10-05 — nothing here
comes from memory.  Commands whose results are asserted inline were run
during this task.

## 1. Build & test toolchain

| component | version | where enforced / tested |
|---|---|---|
| Go — release & local build | `go1.24.1` (`E:/tools/go124/bin/go.exe`, `GOTOOLCHAIN=local`) | local full-suite runs (B0-02 regression); release images build with `golang:1.24` (`chain/Dockerfile`, `deploy/network.Dockerfile`) |
| Go — chain module requirement | `go 1.23.2` directive (`chain/go.mod`) | CI `chain` job: `actions/setup-go` with `go-version-file: chain/go.mod`; verified locally that Go 1.19 **fails** to parse this go.mod (checked, 2026-10-05) |
| Go — root module (compute/vm/tools) | `go 1.19` directive (`go.mod`) | CI `vm` job: setup-go from `go.mod`; verified locally that system Go 1.19 **builds** the root module (`go build ./...` exit 0, checked 2026-10-05) |
| Python — CI test runtime | `3.10` | CI `economics` job (unittest discover in `tools/`) and `network` job (`pip install -e './network[test]'` + pytest) — `.github/workflows/ci.yml` |
| Python — package requirement | `>=3.10` | `network/pyproject.toml` `requires-python` |
| Python — local dev | `3.10.6` | B0-02 regression runs (network suite 31/31) |
| Python — network container base | `python:3.11-slim` | `deploy/network.Dockerfile`; **not covered by CI test jobs** (gap, §4) |
| Docker | `28.0.1`, compose `v2.33.1-desktop.1` | devnet stack `deploy/multivalidator/compose.yaml` (B0-04 lifecycle) |
| Go build flags for release | `CGO_ENABLED=0 -trimpath` | both Dockerfiles (reproducibility input for B1-01) |
| Node / other runtimes | none | — |

## 2. GPU expectations (worker line)

| item | value | evidence |
|---|---|---|
| Proven compute capabilities | **SM86** (A40), **SM89** (RTX 2000 Ada Generation) | `docs/phase-f5b2-gpu-results.nvidia_a40.json`, `...rtx_2000_ada_generation.json` (`environment.compute_capability`) |
| Kernel gencode | `arch=compute_86,code=sm_86` + `arch=compute_89,code=sm_89` only | `gpu/f5b2/f5b2_ext.py` (build flags) |
| CUDA runtime in evidence | **12.8** (via `torch 2.8.0+cu128`) | both F.5B.2 result JSONs (`environment.cuda_runtime`) |
| Driver floor | **NOT ESTABLISHED** — the artifacts do not record a driver version; no claim is made here | F.5B.2 JSONs record device name / CC / CUDA runtime only; B2-02's GPU probe must record driver + runtime for every worker |
| Exactness claim | 310/310 node outputs + validation roots bit-identical to the frozen CPU executor; CPU fallback = 0; canonical float ops = 0 | `docs/phase-f5b2-report.md`, `docs/phase-f5b2-cross-gpu.json` |
| Local dev machine GPU | RTX 2060 SUPER (**SM75**, driver 616.92) — **below the proven floor**; the frozen kernels do not target it | `nvidia-smi` on the dev machine (2026-10-05) |
| Consequence | worker GPU gates (B2 product gate) require SM86/SM89 hardware (e.g. rented GPU); they cannot run on this dev machine, and CPU is not an execution path (no silent fallback, frozen rule) | phase F freeze rules + above |

## 3. OS / architecture matrix

| target | status | basis |
|---|---|---|
| Linux **amd64** | **release target** — the only platform with container images, CI runs and devnet evidence | all Dockerfiles are Linux bases; CI `ubuntu-latest`; devnet runs Linux containers |
| Linux arm64 | **not supported** — no build or test evidence anywhere | explicitly listed, not implied |
| Windows amd64 | **development only** — Go tests, Python tests and the toolchain run here (this machine), but there is no released Windows artifact and no CI job | CI is ubuntu-only; B0-02/B0-04 ran on Windows for dev purposes only |
| macOS | **not supported** — no evidence, no CI | explicitly listed |
| GPU hosts | x86_64 + NVIDIA SM86/SM89 (both proven targets are amd64 GPUs) | §2 |

## 4. Explicitly untested / unsupported combinations

```text
python:3.11-slim network container   not exercised by CI tests (CI tests 3.10 only)
Go 1.19 with the chain module        fails (go.mod directive 1.23.2) — do not use
SM75 / Turing and below (e.g. RTX 2060 SUPER)   not built for; do not claim worker support
arm64 (any OS)                       not supported until a real build+test exists
macOS                                not supported
Windows release artifacts            none exist; dev-only
NVIDIA driver floor                  not established (see section 2) — record per-worker in B2-02
```

## 5. Definition of Done check

```text
toolchain matrix documented                        yes (this file)
every entry cites where it is enforced or tested    yes (paths + live checks inline)
unsupported combinations listed, not implied        yes (section 4)
```

Not in scope here: pinning *new* versions or adding platform support.  If
a later task needs a different toolchain (e.g. arm64), it must produce a
real build+test and update this file — do not silently widen the matrix.
