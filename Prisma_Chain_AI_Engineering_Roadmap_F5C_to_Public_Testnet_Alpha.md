# Prisma Chain — AI Engineering Master Roadmap
**范围：从当前 `Phase F.5C = PARTIAL` 到 `Public Testnet Alpha`**
**用途：给 AI / Codex 作为后续工程主控文档。**
**原则：后续主线以已知工程实现、集成、测试、部署为主；除非触发本文件定义的 Research Reopen Gate，否则禁止重新开启量化 / GPU 算术 / 模型结构研究。**

## 0. AI 执行规则
AI 在执行本文档任务时，必须遵守以下规则：
- 一次只关闭一个明确 blocker，或一组强依赖任务；不要把协议、产品、部署、网页混在一个巨型改动里。
- 任何历史协议版本、hash domain、GraphID、Receipt、wire format 未被明确授权时不得 reinterpret。
- 禁止修改已经冻结的 A13W10 arithmetic、PolicyID、accuracy gate、GPU performance gate。
- 禁止为了让 E2E 通过而加入隐藏 CPU fallback、隐藏 V1 fallback、隐藏 shared filesystem 依赖。
- 任何概率检测（Freivalds）只能用于 detection，不能直接触发 slash；经济惩罚必须由确定性 fraud proof 结算。
- Validator / chain binary 不得依赖 CUDA、NVIDIA 驱动、GPU 或浮点 canonical arithmetic。
- 任何关键项 `NOT TESTED` 时，最终状态只能是 `PARTIAL`，禁止写成“基本完成”。
- 所有 verdict、Gas、余额、AppHash、cross-language / cross-node 结果尽量由脚本机械生成，不手工抄数。
- 每个任务完成后必须更新本地进度文档或新增阶段结果 JSON；不要只留在终端输出。
- 若发现计划与实际代码结构冲突，优先保持协议语义与既有证据，记录 deviation，不得静默修改目标。

## 1. 当前权威状态（Ground Truth）
以下状态来自当前工程阶段与 `docs/phase-f5c-progress.md`，作为后续执行的基线。
- 当前分支：`protocol/transformer-phase-f5c-wide-integer`
- 当前 head：`fdd92d4`
- 当前 Phase F 状态：`PARTIAL`
- 已完成 `CANONICAL_TENSOR_V2`：A13 / W10 / INT64_ACCUM，BE16 / BE64，typed proof，logical range rejection。
- 已完成 `GEMM_A13W10_I64_V1`：exact int64；右操作数支持 W10 或 A13；`MaxSafeK64(A13,W10)=4398046511103`。
- 已完成 `REQUANTIZE_WIDE_V1`、`CANONICAL_GRAPH_V2`、`NodeOutputManifestV2`、`GraphResultCommitV3`、`VerifiedGraphWorkReceiptV3`。
- 正式 Qwen3-0.6B layer0 V2 图：310 nodes；83 GEMM；126 wide requant；33 ADD；26 RMSNORM；24 ROPE；16 SOFTMAX；1 SILU；1 MUL。
- GraphIDV2：`8fb86087697e277602bb4b2088ba48eb4172c21d9c4f83914582f2cbf00d4def`。
- PolicyID：`eb9a9fef403c95cd0ab876893acf14e02928245306f1d1cc6e12f4d99c5eb7a0`。
- 正式 V2 graph logical GEMM MAC：`252,706,816`。
- Python V2 executor 310/310 node values 与 F.5B.2 frozen manifest 一致；Go replay 310/310 TensorRootV2 与 Python 一致。
- 已完成 `FREIVALDS_A13W10_I64_V1`：production 40 rounds；detection only。
- 已完成 `WIDE_GEMM_DISPUTE_V1` 核心：8×8 tile、BE64 partial state、K-step bisection helper、512-MAC arbiter core。
- 已完成 chain V2 honest path：post → accept → CommitV3 → finalize → VWR V3；old tx behavior unchanged。
- 未完成：V2 Graph dispute chain、Graph→Wide bridge、Watcher V2、Bundle V2、DA、query/CLI、Gas、4-validator devnet E2E、final docs、freeze tag。

## 2. 冻结事实：禁止重新研究
- A13W10 + signed int64 accumulator 是正式 arithmetic target。
- F.5A accuracy 已通过原始 AND gate；不再搜索 A/W bits。
- F.5B.1 TRUE_FUSED_MMA_A13W10 在 SM86 / SM89 上 83/83 real GEMM exact，weighted ratio 约 1.17×，已通过 6×/10× 冻结性能门。
- F.5B.2 完整 310-node real Qwen block 在 CPU / SM86 / SM89 逐节点 bit-exact，CPU fallback=0，canonical float ops=0。
- CANONICAL_MATH_V1 继续冻结。
- GEMM v0.1.1、Graph V1、Commit V1/V2、旧 TensorRoot / GraphID / Receipt hash domain 全部冻结。
- GPU backend 是 Worker implementation detail，不进入 consensus identity。

## 3. 总体里程碑
| Milestone | 目标 | 退出条件 |
|---|---|---|
| M1 — F.5C Dispute PASS | V2 Graph dispute + Graph→Wide bridge + 512-MAC | 真实 wide GEMM fraud 能链上确定性裁决 |
| M2 — Watcher/DA PASS | Bundle V2 + Watcher V2 + 2-of-3 DA | Worker 消失仍可验证 / challenge |
| M3 — PHASE_F PASS | 4-validator real-Qwen E2E 全套 | honest/fraud/false/DA/censor/restart 全过 + freeze |
| M4 — Productization Ready | 5 binaries + Job API + Scheduler + docs | 干净机器可安装并跑通 |
| M5 — Private Testnet Alpha | 跨公网、多主机、封闭邀请 | 外部模拟节点闭环 |
| M6 — Public Testnet Alpha | 陌生人可加入并提交/执行/验证 | 至少一个外部 Worker + 外部 Watcher 成功 |

# Phase A — 完成 F.5C 协议闭环
## A0. Baseline / Safety
### A0-01 — 冻结当前基线

- **阶段**：F.5C
- **前置依赖**：无
- **允许修改范围**：`docs/`, test scripts；不改协议逻辑
- **实现任务**：
  - 记录当前 branch、head、Go/Python toolchain 版本。
  - 运行 `go test ./...` 与 Python canonical / F.5C tests。
  - 保存 baseline test summary。
  - 确认 working tree 干净或明确记录非本任务改动。
- **测试要求**：
  - 旧 GEMM / Graph V1 / F.1 / F.5C 已完成测试全部绿。
  - 确认不存在未解释 baseline failure。
- **交付物**：
  - `docs/phase-f5c-baseline.json`
  - `docs/phase-f5c-baseline.md`
- **Definition of Done**：
  - baseline 可复现。
  - 后续所有失败能与 baseline 对比。
- **Stop Rule**：
  - 若 baseline 已红，先修或记录 root cause；不得继续堆新功能。

### A0-02 — 增加旧协议不可变回归锚点

- **阶段**：F.5C
- **前置依赖**：A0-01
- **允许修改范围**：`compute/`, `chain/` tests；不得改旧逻辑
- **实现任务**：
  - 把历史 V1 TensorRoot / GraphID / Commit / Receipt / GEMM vectors 固定为 golden fixtures。
  - 为 V1→V2、V2→V1 domain confusion 增加明确 rejection tests。
  - 为旧 transaction behavior 增加 regression tests。
- **测试要求**：
  - 同一 old fixture 在当前 branch 上 hash/ID 与历史值完全一致。
  - V1 proof 不能被 V2 verifier 接受；V2 proof 不能被 V1 verifier 接受。
- **交付物**：
  - `testdata/f5c_legacy_golden.json`
- **Definition of Done**：
  - 旧协议 byte-for-byte / behavior-for-behavior 不变。

## A1. V2 Graph Dispute Chain
### A1-01 — 枚举所有 V1-only graph dispute call sites

- **阶段**：F.5C
- **前置依赖**：A0-01
- **允许修改范围**：`chain/x/compute/graph_keeper.go`, graph dispute helpers
- **实现任务**：
  - 静态搜索所有 `graphDescriptor` / V1 GraphStateRoot / V1 trail root / V1 TensorRoot proof 调用点。
  - 为每个 call site 标记：V1-only / needs version dispatch / shared safe helper。
  - 生成迁移表。
- **测试要求**：
  - 迁移表覆盖 `GraphTrailClaim`、`GraphMidPoint`、node arbitration、snapshot restore、final settlement。
- **交付物**：
  - `docs/phase-f5c-v2-dispute-callsite-map.md`
- **Definition of Done**：
  - 不存在未分类的 V1-only call site。

### A1-02 — 实现 GraphStateRootV2

- **阶段**：F.5C
- **前置依赖**：A1-01
- **允许修改范围**：`compute/canonical/`
- **实现任务**：
  - 定义独立 V2 state leaf domain。
  - state leaf 绑定 `kind/index/TensorRootV2`。
  - 实现 deterministic ordering。
  - 实现 Go + Python mirror。
- **测试要求**：
  - Go/Python 对同一 live tensor set root 一致。
  - V1 StateRoot proof 不能验证为 V2。
- **交付物**：
  - `testdata/canonical_graph_v2_state_vectors.json`
- **Definition of Done**：
  - GraphStateRootV2 跨语言 bit-exact；domain-separated。

### A1-03 — 实现 TrailRootV2 / TrailProofV2

- **阶段**：F.5C
- **前置依赖**：A1-02
- **允许修改范围**：`compute/canonical/`
- **实现任务**：
  - 定义 V2 trail leaf domain。
  - leaf 绑定 GraphIDV2、step index、GraphStateRootV2。
  - 实现 Merkle build / inclusion proof / verify。
  - 禁止复用 V1 leaf domain。
- **测试要求**：
  - 首/尾 proof 验证。
  - 错误 index / wrong GraphID / V1 proof / wrong count 全拒绝。
- **交付物**：
  - `testdata/canonical_graph_v2_trail_vectors.json`
- **Definition of Done**：
  - TrailRootV2 可跨语言重放。

### A1-04 — 为 GraphTrailClaim 增加 V2 dispatch

- **阶段**：F.5C
- **前置依赖**：A1-02,A1-03
- **允许修改范围**：`chain/x/compute/graph_keeper.go`
- **实现任务**：
  - 按 task descriptor `protocol_version` 明确 dispatch V1/V2。
  - V2 initial root 必须等于 descriptor-derived GraphStateRootV2。
  - V2 endpoint proof 使用 TrailProofV2。
  - V1 路径完全保持原行为。
- **测试要求**：
  - V1 claim regression。
  - V2 honest claim PASS。
  - wrong-version / wrong-initial / wrong-proof rejection。
- **交付物**：
  - `chain/x/compute/*GraphV2* tests`
- **Definition of Done**：
  - 同一 message 不能跨版本解释；V2 claim 可进入 dispute。

### A1-05 — 为 GraphMidPoint 增加 V2 dispatch

- **阶段**：F.5C
- **前置依赖**：A1-04
- **允许修改范围**：`chain/x/compute/graph_keeper.go`, canonical dispute helpers
- **实现任务**：
  - V2 midpoint proof 验证 TrailRootV2。
  - 继续使用 block-height derived round clock。
  - 锁定 midpoint index 计算，禁止 party 自报任意 index。
  - V1 路径不变。
- **测试要求**：
  - honest midpoint / wrong midpoint / late midpoint / wrong proof。
  - V2 bisection interval 每轮单调收缩。
- **交付物**：
  - `chain/x/compute/graph_v2_midpoint_test.go`
- **Definition of Done**：
  - 能稳定二分到单节点。

### A1-06 — 实现 V2 first-divergent-node finalize

- **阶段**：F.5C
- **前置依赖**：A1-05
- **允许修改范围**：`compute/canonical/`, `chain/x/compute/`
- **实现任务**：
  - 当 high-low=1 时解析 first divergent node。
  - 校验 node id 与 descriptor V2 中 operator/version 一致。
  - 输出 versioned node arbitration context。
- **测试要求**：
  - 在人工构造 ADD/ROPE/GEMM fraud graph 上定位准确 node。
  - off-by-one / malformed snapshot rejection。
- **交付物**：
  - `docs/phase-f5c-v2-bisection-results.json`
- **Definition of Done**：
  - 定位 node 与 injected node 100% 一致。

### A1-07 — V2 cheap-op typed evidence adapter

- **阶段**：F.5C
- **前置依赖**：A1-06
- **允许修改范围**：`compute/canonical/`, `chain/x/compute/`
- **实现任务**：
  - ADD/MUL/REQUANTIZE/RMSNORM/ROPE/SILU/SOFTMAX evidence 全部切换到 TensorRootV2 typed chunk proofs。
  - 每个 operator 重新从 graph geometry derive evidence index。
  - 不得信任 caller supplied dtype/index。
- **测试要求**：
  - 每种 operator honest evidence PASS。
  - wrong dtype / wrong chunk / wrong state-root / wrong proof rejection。
- **交付物**：
  - `testdata/f5c_v2_cheap_arbiter_vectors.json`
- **Definition of Done**：
  - 所有 cheap op V2 arbitration bounded 且 typed。

### A1-08 — V2 dispute snapshot / restart

- **阶段**：F.5C
- **前置依赖**：A1-05,A1-06
- **允许修改范围**：`compute/canonical/`, chain persistence
- **实现任务**：
  - 定义 V2 snapshot domain/version。
  - snapshot 持久化 interval、claims、round clock、GraphIDV2。
  - restore 时验证 GraphIDV2 / protocol version。
  - 旧 V1 snapshot 不变。
- **测试要求**：
  - 在 bisection 中途 serialize → new process restore → 继续。
  - restart 前后 first divergent node / final verdict 完全一致。
- **交付物**：
  - `docs/phase-f5c-v2-dispute-restart.json`
- **Definition of Done**：
  - restart-equivalence PASS。

## A2. Graph → Wide GEMM Bridge / 512-MAC
### A2-01 — 定义 Wide GEMM dispute chain envelope

- **阶段**：F.5C
- **前置依赖**：A1-06
- **允许修改范围**：`chain/x/compute/types`, chain messages/envelopes
- **实现任务**：
  - 确定是否新增 proto messages 或使用明确 versioned envelope；选择最小 additive 方案。
  - 字段必须绑定 task id、GraphIDV2、node id、tile、round/epoch、party identity。
  - 定义 replay protection。
- **测试要求**：
  - wrong task/node/version/epoch 不能重放。
  - old GEMM v0.1.1 message 不能进入 wide dispute。
- **交付物**：
  - `docs/phase-f5c-wide-dispute-wire.md`
- **Definition of Done**：
  - wire version 明确且 additive。

### A2-02 — 实现 OpenWideGEMMDispute

- **阶段**：F.5C
- **前置依赖**：A2-01
- **允许修改范围**：`chain/x/compute/`
- **实现任务**：
  - 只允许 first-divergent node 为 `GEMM_A13W10_I64_V1` 时进入。
  - 校验 task active dispute / challenger identity / bond。
  - 绑定 M/N/K/transpose_b/tile geometry。
  - 初始化 wide dispute record。
- **测试要求**：
  - honest open PASS。
  - wrong operator / wrong node / duplicate open / wrong tile rejection。
- **交付物**：
  - `chain/x/compute/wide_gemm_dispute_test.go`
- **Definition of Done**：
  - Graph node 与 wide dispute context 一一绑定。

### A2-03 — 实现 bad row / bad column / 8×8 tile localization helper

- **阶段**：F.5C
- **前置依赖**：A2-02
- **允许修改范围**：off-chain watcher/helper code
- **实现任务**：
  - Freivalds mismatch 后从 residual 定位 bad row。
  - exact row recompute 找 bad column。
  - 映射到 canonical 8×8 output tile。
  - 处理 edge tile zero padding。
- **测试要求**：
  - 随机单点 corrupt / tile corrupt / transpose_b node。
  - 定位 tile 覆盖真实错误位置。
- **交付物**：
  - `testdata/f5c_wide_tile_localization.json`
- **Definition of Done**：
  - 所有 injected wide GEMM fraud 可确定性定位到 tile。

### A2-04 — 实现 WideTraceClaim

- **阶段**：F.5C
- **前置依赖**：A2-02
- **允许修改范围**：`chain/x/compute/`, `compute/canonical/`
- **实现任务**：
  - partial state 固定 64×signed int64，BE64。
  - trace length = ceil(K/8)+1。
  - 生成 / 验证 trace root + endpoint proofs。
  - trace 只在 challenge 后生成。
- **测试要求**：
  - honest worker/challenger claims。
  - wrong length / malformed BE64 / wrong tile / wrong proof rejection。
- **交付物**：
  - `testdata/f5c_wide_trace_vectors.json`
- **Definition of Done**：
  - chain 可锁定双方 trace claims。

### A2-05 — 实现 WideMidPoint

- **阶段**：F.5C
- **前置依赖**：A2-04
- **允许修改范围**：`chain/x/compute/`
- **实现任务**：
  - 按 K-step trace midpoint 二分。
  - round clock 绑定 block height。
  - proof index 由 interval derive。
  - 禁止跳步。
- **测试要求**：
  - first divergent K-step 定位。
  - late / duplicate / wrong-index midpoint rejection。
- **交付物**：
  - `docs/phase-f5c-wide-bisection.json`
- **Definition of Done**：
  - 区间收敛到一个 8-wide K step。

### A2-06 — 实现 ArbitrateWide512

- **阶段**：F.5C
- **前置依赖**：A2-05
- **允许修改范围**：`compute/canonical/`, `chain/x/compute/`
- **实现任务**：
  - 输入：8×8 A tile、8×8 B tile、prev state、party next states、typed proofs。
  - 按 logical arithmetic exact int64 重算 8×8×8 = 512 MAC。
  - 处理 transpose_b。
  - GPU decomposition / Karatsuba 不得进入 chain implementation。
  - 输出 WorkerWins / ChallengerWins / BothInvalid。
- **测试要求**：
  - A13×W10 与 A13×A13 两类真实 node。
  - 正负极值、edge padding、wrong proof、wrong state。
  - assert exactly 512 logical MAC for full tile。
- **交付物**：
  - `docs/phase-f5c-wide512-vectors.json`
- **Definition of Done**：
  - 512-MAC arbiter 与 Go/Python wide reference bit-exact。

### A2-07 — Graph→Wide settlement bridge

- **阶段**：F.5C
- **前置依赖**：A1-06,A2-06
- **允许修改范围**：`chain/x/compute/`
- **实现任务**：
  - Graph first-divergent GEMM node 转入 wide dispute。
  - wide verdict 回写 Graph dispute transcript。
  - 调用现有 economics settlement：refund/slash/reward。
  - fraud worker 永不生成 VWR。
- **测试要求**：
  - ChallengerWins / WorkerWins / BothInvalid 分支。
  - 余额逐最小单位校验。
- **交付物**：
  - `docs/phase-f5c-wide-settlement-unit.json`
- **Definition of Done**：
  - Graph dispute 与 wide verdict 完整闭环。

### A2-08 — Wide dispute restart persistence

- **阶段**：F.5C
- **前置依赖**：A2-05,A2-06
- **允许修改范围**：chain persistence
- **实现任务**：
  - snapshot wide interval / claims / tile / trace roots / deadline。
  - 恢复时校验 task/node/GraphIDV2。
  - 继续到相同 final step。
- **测试要求**：
  - mid-bisection restart。
  - pre-arbitration restart。
  - 恢复后的 verdict 与无重启一致。
- **交付物**：
  - `docs/phase-f5c-wide-restart.json`
- **Definition of Done**：
  - restart-equivalence PASS。

## A3. Watcher V2 / Verification Bundle
### A3-01 — 定义 GraphVerificationBundleV2 schema

- **阶段**：F.5C
- **前置依赖**：A0-02
- **允许修改范围**：`compute/canonical/`, watcher/tooling
- **实现任务**：
  - 包含 GraphDescriptorV2、inputs、weights/constants、all node outputs、TensorRootV2 proofs、ManifestV2 proofs、final outputs。
  - 明确 artifact version 与 hash domain。
  - 禁止包含正常路径 execution trace / dispute trail。
- **测试要求**：
  - encode/decode round-trip。
  - wrong version / truncated / duplicate node / wrong descriptor rejection。
- **交付物**：
  - `docs/graph-verification-bundle-v2.md`
  - `testdata/f5c_bundle_v2_small.bin`
- **Definition of Done**：
  - bundle schema 可独立解析并验证结构。

### A3-02 — 实现 BundleV2 builder

- **阶段**：F.5C
- **前置依赖**：A3-01
- **允许修改范围**：tooling / worker-side artifact builder
- **实现任务**：
  - 从 formal V2 execution 构造 bundle。
  - 节点顺序严格匹配 ManifestV2。
  - 生成 artifact root / size breakdown。
  - 支持 real Qwen 310-node bundle。
- **测试要求**：
  - small graph / real graph build。
  - bundle root 与 commit manifest/final root 一致。
- **交付物**：
  - `tools/f5c_build_bundle_v2`
  - `docs/phase-f5c-bundle-size.json`
- **Definition of Done**：
  - 真实 bundle 可机械生成。

### A3-03 — 实现 Watcher root phase

- **阶段**：F.5C
- **前置依赖**：A3-01,A3-02
- **允许修改范围**：watcher V2
- **实现任务**：
  - 验证 GraphIDV2。
  - 验证 artifact root。
  - 验证 NodeOutputManifestRootV2。
  - 验证每个 TensorRootV2 / proof。
  - 验证 final output root。
  - 任一 root mismatch 立即停止数学验证。
- **测试要求**：
  - honest bundle PASS。
  - tamper graph/input/node/final/manifest 分别拒绝。
- **交付物**：
  - `docs/phase-f5c-watcher-root-tests.json`
- **Definition of Done**：
  - root phase 对结构性 fraud 100% 拒绝。

### A3-04 — 实现 production post-commit randomness

- **阶段**：F.5C
- **前置依赖**：A3-03
- **允许修改范围**：watcher
- **实现任务**：
  - Worker CommitV3 锁定后才生成 Freivalds randomness。
  - 使用 CSPRNG + commit context + watcher nonce + round index。
  - 测试模式允许固定 seed，但显式 TEST ONLY。
  - 禁止 `r=H(manifest_root)` 作为唯一随机源。
- **测试要求**：
  - 同一 test seed 可重放。
  - production path randomness 在 pre-commit 时不可知。
- **交付物**：
  - `docs/freivalds-a13w10-v1.md`
- **Definition of Done**：
  - 随机性时序满足 post-commit requirement。

### A3-05 — Watcher wide GEMM Freivalds verifier

- **阶段**：F.5C
- **前置依赖**：A3-04
- **允许修改范围**：watcher / canonical verifier
- **实现任务**：
  - 83 real GEMM 使用 `FREIVALDS_A13W10_I64_V1`。
  - production 40 rounds。
  - 支持 A13×W10 与 A13×A13、transpose_b。
  - 记录 per-node rounds/time/bounds。
- **测试要求**：
  - honest 83/83 PASS。
  - 多个单点 / tile corrupt 必须检测。
  - false-accept bound reporting。
- **交付物**：
  - `docs/phase-f5c-watcher-freivalds.json`
- **Definition of Done**：
  - `wide_freivalds_gemms=83`; 无 full GEMM。

### A3-06 — Watcher cheap-op exact verifier

- **阶段**：F.5C
- **前置依赖**：A3-03
- **允许修改范围**：watcher
- **实现任务**：
  - RMSNORM / REQUANTIZE / ROPE / ADD / SOFTMAX / SILU / MUL exact canonical recompute。
  - 节点数由 graph 派生。
  - 逐 node compare committed output。
- **测试要求**：
  - honest 227 cheap nodes PASS。
  - 每类至少一个 targeted corruption。
- **交付物**：
  - `docs/phase-f5c-watcher-cheapops.json`
- **Definition of Done**：
  - 所有 non-GEMM nodes exact verified。

### A3-07 — Watcher fraud localization pipeline

- **阶段**：F.5C
- **前置依赖**：A3-05,A3-06,A2-03
- **允许修改范围**：watcher
- **实现任务**：
  - root phase PASS 后进入 math phase。
  - GEMM mismatch → row/column/tile localization。
  - cheap op mismatch → first suspected node/evidence builder。
  - 输出 challenge plan。
- **测试要求**：
  - 真实 wide GEMM fraud 定位。
  - 真实 ROPE fraud 定位。
  - 多处 corrupt 时至少稳定定位首个可证明节点。
- **交付物**：
  - `docs/phase-f5c-watcher-fraud-localization.json`
- **Definition of Done**：
  - 可自动生成链上 challenge 所需上下文。

### A3-08 — Watcher no-worker-filesystem audit

- **阶段**：F.5C
- **前置依赖**：A3-07
- **允许修改范围**：watcher runner
- **实现任务**：
  - 将 Worker local artifacts 从 test environment 移除。
  - Watcher 只给 chain metadata + BundleV2。
  - 加入 forbidden-path audit。
- **测试要求**：
  - Worker process stopped / filesystem unavailable 后 Watcher 仍 PASS/挑战。
- **交付物**：
  - `docs/phase-f5c-watcher-independence.json`
- **Definition of Done**：
  - Watcher 对 Worker 本地状态零依赖。

### A3-09 — Watcher restart-equivalence

- **阶段**：F.5C
- **前置依赖**：A3-08
- **允许修改范围**：watcher persistence
- **实现任务**：
  - 在 root phase / Freivalds / challenge build 之间 kill process。
  - 新进程从 chain + DA 重新获取 state。
  - 不从旧内存恢复随机未提交状态。
- **测试要求**：
  - honest verdict 重放一致。
  - fraud node/tile/challenge plan 一致。
- **交付物**：
  - `docs/phase-f5c-watcher-restart.json`
- **Definition of Done**：
  - fresh watcher instance produces same verifiable outcome。

### A3-10 — Watcher cost instrumentation

- **阶段**：F.5C
- **前置依赖**：A3-05,A3-06
- **允许修改范围**：watcher
- **实现任务**：
  - 记录 root validation、Freivalds、cheap-op、bundle decode、total seconds。
  - 记录 `full_gemm_calls`。
  - 记录 peak memory / downloaded bytes。
- **测试要求**：
  - 真实 Qwen bundle 上生成真实测量。
- **交付物**：
  - `docs/phase-f5c-watcher-cost.json`
- **Definition of Done**：
  - `full_gemm_calls=0`；所有 timing 非负且来自真实运行。

## A4. DA Integration
### A4-01 — 新增 GRAPH_VERIFICATION_BUNDLE_V2 artifact kind

- **阶段**：F.5C
- **前置依赖**：A3-01
- **允许修改范围**：Phase E DA registry / artifact enum
- **实现任务**：
  - 新增 versioned artifact kind。
  - 复用现有 DA_REPLICA_V1 provider identity / bond / quorum。
  - 不创建新 provider registry。
- **测试要求**：
  - 旧 GEMM DA artifact regression。
  - 新 kind encode/decode。
- **交付物**：
  - `docs/phase-f5c-da-artifact.md`
- **Definition of Done**：
  - 同一 DA network 支持 V1 GEMM 与 Graph Bundle V2。

### A4-02 — Provider pre-attestation verification

- **阶段**：F.5C
- **前置依赖**：A4-01,A3-02
- **允许修改范围**：DA provider
- **实现任务**：
  - decode bundle。
  - 验证 GraphIDV2 / artifact hash / all TensorRootV2 / manifest / final root。
  - 仅证明 stored bytes 与 Worker commitment 一致。
  - 不得运行 Transformer 数学。
- **测试要求**：
  - honest bundle attests。
  - wrong blob / wrong manifest / wrong final root refusal。
- **交付物**：
  - `docs/phase-f5c-da-attestation-tests.json`
- **Definition of Done**：
  - Provider 不会为不匹配 commitment 的 bytes attest。

### A4-03 — Graph V2 DA quorum gate

- **阶段**：F.5C
- **前置依赖**：A4-02
- **允许修改范围**：chain graph V2 finalize state machine
- **实现任务**：
  - ResultCommitV3 后等待 2-of-3 valid attestations。
  - 只有 quorum 后进入 challenge_window_ready。
  - finalize 再次确认 availability state。
- **测试要求**：
  - 0/3、1/3 不可 finalize。
  - 2/3、3/3 可以进入窗口。
- **交付物**：
  - `chain/x/compute/graph_v2_da_gate_test.go`
- **Definition of Done**：
  - Graph finalize 被 DA quorum 正确约束。

### A4-04 — Typed on-chain DA chunk challenge

- **阶段**：F.5C
- **前置依赖**：A4-01,A4-02
- **允许修改范围**：chain + DA provider
- **实现任务**：
  - challenge 绑定 task/artifact/node/tensor/chunk/provider。
  - provider 响应 typed chunk bytes + TensorChunkProofV2 + artifact binding。
  - chain 验证 proof 与 deadline。
- **测试要求**：
  - honest response PASS。
  - wrong chunk / wrong proof / wrong provider / replay rejection。
- **交付物**：
  - `docs/phase-f5c-da-chunk-challenge.json`
- **Definition of Done**：
  - DA availability 可由客观 on-chain proof 结算。

### A4-05 — DA after-attest loss timeout

- **阶段**：F.5C
- **前置依赖**：A4-04
- **允许修改范围**：chain + DA provider harness
- **实现任务**：
  - provider attest 后删除 artifact。
  - 发起 on-chain challenge。
  - 等待 block-height deadline。
  - 按已有 Phase E semantics 处罚 provider。
- **测试要求**：
  - provider 不响应时 objective timeout 成立。
- **交付物**：
  - `docs/phase-f5c-da-after-attest-loss.json`
- **Definition of Done**：
  - HTTP failure 不直接 slash；只有 missed on-chain deadline 才处罚。

### A4-06 — DA quorum loss availability_failed

- **阶段**：F.5C
- **前置依赖**：A4-03,A4-05
- **允许修改范围**：chain settlement
- **实现任务**：
  - 使有效 replicas 降到 <2。
  - 推进到 availability expiry。
  - task 标记 availability_failed。
  - refund requester，0 VWR。
- **测试要求**：
  - 余额逐最小单位校验。
- **交付物**：
  - `docs/phase-f5c-da-quorum-loss.json`
- **Definition of Done**：
  - quorum loss 永不生成 VWR。

### A4-07 — One-provider-offline recovery

- **阶段**：F.5C
- **前置依赖**：A4-03,A3-08
- **允许修改范围**：DA + watcher integration
- **实现任务**：
  - 让 1/3 provider offline。
  - Watcher 从剩余 2 provider 拉 bundle。
  - 完成完整 verification / challenge。
- **测试要求**：
  - honest 和 fraud 各跑一次。
- **交付物**：
  - `docs/phase-f5c-da-one-offline.json`
- **Definition of Done**：
  - 单 provider offline 不阻断 verification。

## A5. Query / CLI / Gas
### A5-01 — GraphTaskV2 query

- **阶段**：F.5C
- **前置依赖**：A1-04,A4-03
- **允许修改范围**：chain query surface
- **实现任务**：
  - 返回 task version/status/GraphIDV2/worker/challenge window/DA status/final root。
  - 不要暴露内部 raw store layout。
- **测试要求**：
  - posted/assigned/result/challenged/finalized/availability_failed 状态。
- **交付物**：
  - `chain/x/compute/query_v2.go`
- **Definition of Done**：
  - 外部工具不需要读 ABCI raw store。

### A5-02 — GraphDisputeV2 query

- **阶段**：F.5C
- **前置依赖**：A1-06,A2-07
- **允许修改范围**：chain query surface
- **实现任务**：
  - 返回 dispute phase、interval、deadline、node id、wide dispute state、parties。
  - 敏感/冗余 witness 不全量返回。
- **测试要求**：
  - graph bisection / wide bisection / resolved 状态。
- **交付物**：
  - `chain/x/compute/query_v2.go`
- **Definition of Done**：
  - CLI/Watcher 可从正式 query 恢复状态。

### A5-03 — ReceiptV3 query

- **阶段**：F.5C
- **前置依赖**：A2-07
- **允许修改范围**：chain query surface
- **实现任务**：
  - 按 task / receipt id 查询 VWR V3。
  - 返回 GraphIDV2、PolicyID、work vector、verification mode、settlement ref、DA ref。
- **测试要求**：
  - honest task 1 receipt；fraud/availability fail 0 receipt。
- **交付物**：
  - `chain/x/compute/query_v2.go`
- **Definition of Done**：
  - receipt query 可外部使用。

### A5-04 — DA status query

- **阶段**：F.5C
- **前置依赖**：A4-03
- **允许修改范围**：chain query surface
- **实现任务**：
  - 返回 provider attestations、valid replica count、required quorum、challenge state、expiry。
- **测试要求**：
  - 0/1/2/3 replicas 与 timeout 状态。
- **交付物**：
  - `chain/x/compute/query_v2.go`
- **Definition of Done**：
  - Watcher/CLI 可判断 availability。

### A5-05 — CLI V2 verbs

- **阶段**：F.5C
- **前置依赖**：A5-01,A5-02,A5-03,A5-04
- **允许修改范围**：CLI
- **实现任务**：
  - post graph-v2 task。
  - accept。
  - submit commit-v3。
  - query task/dispute/receipt/DA。
  - challenge / trail / midpoint / wide dispute / arbitrate actions 覆盖 E2E harness。
- **测试要求**：
  - CLI against local 1-validator smoke chain。
- **交付物**：
  - `cmd/prisma-cli or existing CLI package`
- **Definition of Done**：
  - devnet E2E 不再依赖 ad-hoc raw-store helpers。

### A5-06 — Gas instrumentation points

- **阶段**：F.5C
- **前置依赖**：A2-06,A4-04
- **允许修改范围**：chain gas accounting
- **实现任务**：
  - 为 PostGraphV2 / Accept / CommitV3 / DA attestation / challenge / trail / midpoint / wide open / wide trace / wide midpoint / wide512 / RoPE arbiter / finalize 记录 consumed gas。
  - Gas 按 bounded units，不按 wall time。
- **测试要求**：
  - 单元测试确保同输入 gas deterministic。
- **交付物**：
  - `chain/x/compute/graph_v2_gas.go`
- **Definition of Done**：
  - 所有关键 tx 有可测 gas path。

### A5-07 — 实测 512-MAC witness bytes / gas / proof depth

- **阶段**：F.5C
- **前置依赖**：A2-06,A5-06
- **允许修改范围**：devnet / measurement tests
- **实现任务**：
  - 在真实 Qwen GEMM node 上构造 final wide arbiter tx。
  - 记录 witness bytes、Merkle siblings/depth、gas。
  - 确认 transaction 未突破当前 block gas envelope。
- **测试要求**：
  - 至少 A13×W10、A13×A13 各一个真实 node。
- **交付物**：
  - `docs/phase-f5c-gas-results.json`
- **Definition of Done**：
  - 报告包含 authoritative 512-MAC measurements。
- **Stop Rule**：
  - 若超过现有 envelope，先记录 blocker；禁止直接无限提高 block gas。

## A6. 4-Validator Real-Qwen E2E
### A6-01 — 升级 multivalidator harness 到 Graph V2

- **阶段**：F.5C
- **前置依赖**：A5-05
- **允许修改范围**：`deploy/multivalidator`
- **实现任务**：
  - 4 validators 25% each。
  - 支持 Graph V2 tx / queries / DA provider endpoints / watcher runner。
  - 记录每个 validator height/app hash。
- **测试要求**：
  - 基础出块 / tx inclusion / restart smoke。
- **交付物**：
  - `deploy/multivalidator/f5c_v2_e2e.*`
- **Definition of Done**：
  - harness 能机械执行场景并收集证据。

### A6-02 — Honest real-Qwen E2E

- **阶段**：F.5C
- **前置依赖**：A3-10,A4-07,A6-01
- **允许修改范围**：devnet E2E
- **实现任务**：
  - Post 310-node Graph V2。
  - Worker accept。
  - 执行 frozen protocol executor。
  - CommitV3。
  - DA 2-of-3。
  - Watcher PASS。
  - challenge window end。
  - Finalize。
- **测试要求**：
  - exactly one VWR V3。
  - requester/worker/provider balances 正确。
  - 4 validators converge。
- **交付物**：
  - `docs/phase-f5c-e2e-honest.json`
- **Definition of Done**：
  - honest real block end-to-end PASS。

### A6-03 — Real wide GEMM fraud artifact generator

- **阶段**：F.5C
- **前置依赖**：A3-02
- **允许修改范围**：test tooling
- **实现任务**：
  - 选择真实 `GEMM_A13W10_I64_V1` node。
  - 修改一个 output element/tile。
  - 从错误 tensor 继续重算所有 downstream nodes。
  - 重新生成 self-consistent ManifestV2 / final output / valid worker signature。
- **测试要求**：
  - 所有 root/commit structural checks PASS；只有数学是错的。
- **交付物**：
  - `tools/f5c_make_real_gemm_fraud.py`
  - `testdata/f5c_real_gemm_fraud_manifest.json`
- **Definition of Done**：
  - 不是简单 manifest mismatch，而是真正 self-consistent wrong computation。

### A6-04 — Real GEMM fraud E2E

- **阶段**：F.5C
- **前置依赖**：A6-03,A2-07,A3-07,A4-07,A6-01
- **允许修改范围**：devnet E2E
- **实现任务**：
  - Worker 提交 bad-but-self-consistent bundle。
  - DA 正常 attest。
  - Watcher root phase PASS。
  - Freivalds detects。
  - 定位 bad tile。
  - Open Graph challenge。
  - Graph V2 bisection。
  - Graph→Wide bridge。
  - K-trace bisection。
  - ArbitrateWide512。
  - settlement。
- **测试要求**：
  - first divergent node = injected GEMM。
  - ChallengerWins。
  - requester refund / worker slash / challenger reward 正确。
  - worker VWR = 0。
  - 4 validators converge。
- **交付物**：
  - `docs/phase-f5c-e2e-gemm-fraud.json`
- **Definition of Done**：
  - 完整 512-MAC fraud proof 在真实图上闭环。

### A6-05 — Real ROPE fraud artifact generator

- **阶段**：F.5C
- **前置依赖**：A3-02
- **允许修改范围**：test tooling
- **实现任务**：
  - 选择真实 ROPE_FIXED_V1 node。
  - 修改 output。
  - 从错误 tensor 继续重算 downstream。
  - 构造 self-consistent bad bundle/commit。
- **测试要求**：
  - structural roots/manifest PASS。
- **交付物**：
  - `tools/f5c_make_real_rope_fraud.py`
- **Definition of Done**：
  - 只留下数学 fraud。

### A6-06 — Real ROPE fraud E2E

- **阶段**：F.5C
- **前置依赖**：A6-05,A1-07,A6-01
- **允许修改范围**：devnet E2E
- **实现任务**：
  - Watcher exact cheap-op detects。
  - Graph V2 bisection 定位 ROPE node。
  - bounded RoPE arbiter。
  - settlement。
- **测试要求**：
  - ChallengerWins。
  - 0 VWR。
  - balances 正确。
  - 4 validators converge。
- **交付物**：
  - `docs/phase-f5c-e2e-rope-fraud.json`
- **Definition of Done**：
  - Phase F 原 ROPE devnet 缺口关闭。

### A6-07 — False challenge E2E

- **阶段**：F.5C
- **前置依赖**：A6-02,A1-07
- **允许修改范围**：devnet E2E
- **实现任务**：
  - honest worker result。
  - 恶意 challenger 提交不同 final/trail。
  - 走完整 challenge path。
  - WorkerWins。
  - worker 继续 finalization。
- **测试要求**：
  - challenger bond 按 economics 处理。
  - worker exactly one VWR。
- **交付物**：
  - `docs/phase-f5c-e2e-false-challenge.json`
- **Definition of Done**：
  - spam challenge 不破坏 honest settlement。

### A6-08 — Combined adversarial E2E

- **阶段**：F.5C
- **前置依赖**：A6-04,A4-07
- **允许修改范围**：devnet E2E
- **实现任务**：
  - real GEMM fraud。
  - DA provider A offline。
  - validator-a proposer 对 challenge omission 1 block。
  - next honest proposer include。
  - 继续到 512-MAC / ChallengerWins。
- **测试要求**：
  - censored tx 只延迟，不被判 invalid。
  - 2 DA providers 足够验证。
  - 4 validators 最终 converge。
- **交付物**：
  - `docs/phase-f5c-e2e-combined.json`
- **Definition of Done**：
  - permissionless path 在单 proposer censorship + DA offline 下闭环。

### A6-09 — Wide dispute restart E2E

- **阶段**：F.5C
- **前置依赖**：A2-08,A6-04
- **允许修改范围**：devnet E2E
- **实现任务**：
  - 在 wide midpoint 后停止相关进程/节点。
  - 从 persisted state restart。
  - 继续 arbitration。
- **测试要求**：
  - same first divergent step。
  - same final verdict。
  - same balances。
- **交付物**：
  - `docs/phase-f5c-e2e-wide-restart.json`
- **Definition of Done**：
  - dispute state crash-safe。

### A6-10 — Validator offline/recovery E2E

- **阶段**：F.5C
- **前置依赖**：A6-01
- **允许修改范围**：devnet E2E
- **实现任务**：
  - 停止 1 validator。
  - 其余 3 继续出块并处理 V2 tx。
  - 恢复 validator 并 catch up。
- **测试要求**：
  - 恢复后 height catch-up。
  - 4 validator same app hash。
- **交付物**：
  - `docs/phase-f5c-e2e-validator-restart.json`
- **Definition of Done**：
  - Phase E liveness regression 保持。

### A6-11 — Availability failure E2E

- **阶段**：F.5C
- **前置依赖**：A4-06,A6-01
- **允许修改范围**：devnet E2E
- **实现任务**：
  - 让 DA quorum <2。
  - 推进 expiry。
  - task availability_failed。
  - refund。
- **测试要求**：
  - 0 VWR。
  - balances 正确。
  - 4 validators converge。
- **交付物**：
  - `docs/phase-f5c-e2e-availability-failure.json`
- **Definition of Done**：
  - availability failure 不被误判 computation fraud。

## A7. Final Docs / Freeze
### A7-01 — 完成 Canonical Tensor V2 spec

- **阶段**：F.5C
- **前置依赖**：A2-06,A4-04
- **允许修改范围**：`docs/`
- **实现任务**：
  - 写 dtype ranges / encoding / chunking / root domains / proof rules / rejection rules。
  - 包含 A13/W10/INT64_ACCUM edge cases。
- **测试要求**：
  - spec examples 与 golden vectors 一致。
- **交付物**：
  - `docs/canonical-tensor-v2.md`
- **Definition of Done**：
  - 第三方不看代码也能实现 typed tensor commitment。

### A7-02 — 完成 Canonical Graph V2 spec

- **阶段**：F.5C
- **前置依赖**：A1-08,A2-07
- **允许修改范围**：`docs/`
- **实现任务**：
  - Graph descriptor / GraphIDV2 / state/trail roots / ManifestV2 / CommitV3 / ReceiptV3 / dispute path。
  - 明确 V1 compatibility。
- **测试要求**：
  - 文档里的 hash/vector 与 tests 一致。
- **交付物**：
  - `docs/canonical-graph-v2.md`
- **Definition of Done**：
  - 第三方可独立实现 Graph V2。

### A7-03 — 完成 Wide GEMM spec

- **阶段**：F.5C
- **前置依赖**：A2-06
- **允许修改范围**：`docs/`
- **实现任务**：
  - A13/W10/A13 right operand semantics。
  - transpose_b。
  - MaxSafeK64。
  - work accounting。
  - tile/trace/512-MAC dispute。
  - 明确 GPU decomposition 非 protocol。
- **测试要求**：
  - spec 与 Go/Python vectors 一致。
- **交付物**：
  - `docs/wide-gemm-a13w10-v1.md`
- **Definition of Done**：
  - 数学定义完全 backend-independent。

### A7-04 — 完成 Wide Requant spec

- **阶段**：F.5C
- **前置依赖**：A0-02
- **允许修改范围**：`docs/`
- **实现任务**：
  - int64 input、mult、shift、ties-to-even、clamp、overflow admission。
  - 列出真实 profile usage。
- **测试要求**：
  - tie/saturation vectors。
- **交付物**：
  - `docs/wide-requant-v1.md`
- **Definition of Done**：
  - Go/Python 实现可由文档独立复现。

### A7-05 — 完成 Freivalds spec

- **阶段**：F.5C
- **前置依赖**：A3-05
- **允许修改范围**：`docs/`
- **实现任务**：
  - 40 rounds。
  - post-commit randomness。
  - A13×W10 / A13×A13 / transpose_b。
  - detection-only security boundary。
  - union-bound reporting。
- **测试要求**：
  - spec examples 与 watcher tests 一致。
- **交付物**：
  - `docs/freivalds-a13w10-v1.md`
- **Definition of Done**：
  - 不会被误读成概率 slashing。

### A7-06 — 生成 phase-f5c-report

- **阶段**：F.5C
- **前置依赖**：A6-11,A5-07
- **允许修改范围**：`docs/`
- **实现任务**：
  - 汇总协议、Watcher、DA、Gas、E2E、兼容性。
  - 所有关键数字从 JSON 读取。
  - 明确 CPU protocol E2E 与 F.5B.2 GPU evidence 是两条证据链。
- **测试要求**：
  - 报告引用的所有 artifact 存在。
- **交付物**：
  - `docs/phase-f5c-report.md`
- **Definition of Done**：
  - Q1..Qn 只给可证实结论。

### A7-07 — 生成 phase-f-final-report

- **阶段**：Phase F
- **前置依赖**：A7-01..A7-06
- **允许修改范围**：`docs/`
- **实现任务**：
  - 完整 lineage：F / F.1 / F.2A / F.3A / F.4A / F.5A / F.5B / F.5B.1 / F.5B.2 / F.5C。
  - 保留 W8A8/groupwise/A-only/int32-frontier 的失败研究。
  - 列出最终 GraphIDV2 / PolicyID / protocol versions / E2E evidence。
- **测试要求**：
  - 任何关键项 NOT TESTED 时自动判 PARTIAL。
- **交付物**：
  - `docs/phase-f-final-report.md`
- **Definition of Done**：
  - 只有所有预声明 gates PASS 才写 `PHASE_F = PASS`。

### A7-08 — Phase F freeze tag

- **阶段**：Phase F
- **前置依赖**：A7-07
- **允许修改范围**：git metadata / docs
- **实现任务**：
  - 确认全测试绿。
  - 确认 reports / JSON committed。
  - 创建 freeze tag（遵循 repo naming convention）。
  - 记录 tag→commit→GraphIDV2→PolicyID mapping。
- **测试要求**：
  - fresh checkout from tag 执行完整 regression。
- **交付物**：
  - `docs/phase-f-freeze.json`
- **Definition of Done**：
  - 可从 tag 完整重建协议与测试证据。
- **Stop Rule**：
  - 若 PHASE_F != PASS，禁止创建“pass”含义的 tag。

# Phase B — Testnet Productization
Phase B 只有在 `PHASE_F = PASS` 后启动。以下任务是工程产品化，不重新设计协议。
### B1-01 — prismad release build

- **阶段**：Productization
- **前置依赖**：PHASE_F=PASS
- **允许修改范围**：按组件对应目录；不得修改冻结 protocol semantics
- **实现任务**：
  - 构建 reproducible Linux binaries；版本输出含 protocol versions / git SHA；Docker image；healthcheck。
- **测试要求**：
  - 至少 unit test + clean-machine smoke test；涉及网络状态的任务加 integration test。
- **交付物**：
  - `docs/productization/b1-01.md`
- **Definition of Done**：
  - 组件可独立构建/运行；失败有明确 error，不 silent fallback。

### B1-02 — Public RPC node mode

- **阶段**：Productization
- **前置依赖**：PHASE_F=PASS
- **允许修改范围**：按组件对应目录；不得修改冻结 protocol semantics
- **实现任务**：
  - 增加 non-validator RPC profile；限流、request size bounds、sentry topology。
- **测试要求**：
  - 至少 unit test + clean-machine smoke test；涉及网络状态的任务加 integration test。
- **交付物**：
  - `docs/productization/b1-02.md`
- **Definition of Done**：
  - 组件可独立构建/运行；失败有明确 error，不 silent fallback。

### B1-03 — Genesis/config generator

- **阶段**：Productization
- **前置依赖**：PHASE_F=PASS
- **允许修改范围**：按组件对应目录；不得修改冻结 protocol semantics
- **实现任务**：
  - 生成 chain-id、genesis、seed peers、DA defaults、faucet params；checksum。
- **测试要求**：
  - 至少 unit test + clean-machine smoke test；涉及网络状态的任务加 integration test。
- **交付物**：
  - `docs/productization/b1-03.md`
- **Definition of Done**：
  - 组件可独立构建/运行；失败有明确 error，不 silent fallback。

### B2-01 — prisma-worker identity

- **阶段**：Productization
- **前置依赖**：PHASE_F=PASS
- **允许修改范围**：按组件对应目录；不得修改冻结 protocol semantics
- **实现任务**：
  - 本地 keystore；chain account + protocol key；init/import/rotate；禁止镜像内置私钥。
- **测试要求**：
  - 至少 unit test + clean-machine smoke test；涉及网络状态的任务加 integration test。
- **交付物**：
  - `docs/productization/b2-01.md`
- **Definition of Done**：
  - 组件可独立构建/运行；失败有明确 error，不 silent fallback。

### B2-02 — GPU capability probe

- **阶段**：Productization
- **前置依赖**：PHASE_F=PASS
- **允许修改范围**：按组件对应目录；不得修改冻结 protocol semantics
- **实现任务**：
  - GPU model/cc/VRAM/driver/CUDA；生成 capability advertisement。
- **测试要求**：
  - 至少 unit test + clean-machine smoke test；涉及网络状态的任务加 integration test。
- **交付物**：
  - `docs/productization/b2-02.md`
- **Definition of Done**：
  - 组件可独立构建/运行；失败有明确 error，不 silent fallback。

### B2-03 — Worker join/bond/register

- **阶段**：Productization
- **前置依赖**：PHASE_F=PASS
- **允许修改范围**：按组件对应目录；不得修改冻结 protocol semantics
- **实现任务**：
  - 连接 RPC；检查余额/领测试币；bond；注册 capability。
- **测试要求**：
  - 至少 unit test + clean-machine smoke test；涉及网络状态的任务加 integration test。
- **交付物**：
  - `docs/productization/b2-03.md`
- **Definition of Done**：
  - 组件可独立构建/运行；失败有明确 error，不 silent fallback。

### B2-04 — Worker job fetch/accept

- **阶段**：Productization
- **前置依赖**：PHASE_F=PASS
- **允许修改范围**：按组件对应目录；不得修改冻结 protocol semantics
- **实现任务**：
  - 拉取/订阅 supported jobs；静态 profile compatibility；accept；本地 journal。
- **测试要求**：
  - 至少 unit test + clean-machine smoke test；涉及网络状态的任务加 integration test。
- **交付物**：
  - `docs/productization/b2-04.md`
- **Definition of Done**：
  - 组件可独立构建/运行；失败有明确 error，不 silent fallback。

### B2-05 — Worker model/profile manager

- **阶段**：Productization
- **前置依赖**：PHASE_F=PASS
- **允许修改范围**：按组件对应目录；不得修改冻结 protocol semantics
- **实现任务**：
  - 只支持冻结 QWEN3_BLOCK_PROFILE_V2；下载/校验 model/profile artifact；cache。
- **测试要求**：
  - 至少 unit test + clean-machine smoke test；涉及网络状态的任务加 integration test。
- **交付物**：
  - `docs/productization/b2-05.md`
- **Definition of Done**：
  - 组件可独立构建/运行；失败有明确 error，不 silent fallback。

### B2-06 — Worker execution backend

- **阶段**：Productization
- **前置依赖**：PHASE_F=PASS
- **允许修改范围**：按组件对应目录；不得修改冻结 protocol semantics
- **实现任务**：
  - TRUE_FUSED_MMA_A13W10 production candidate；no silent CPU fallback；输出 execution metrics。
- **测试要求**：
  - 至少 unit test + clean-machine smoke test；涉及网络状态的任务加 integration test。
- **交付物**：
  - `docs/productization/b2-06.md`
- **Definition of Done**：
  - 组件可独立构建/运行；失败有明确 error，不 silent fallback。

### B2-07 — Worker CommitV3 builder

- **阶段**：Productization
- **前置依赖**：PHASE_F=PASS
- **允许修改范围**：按组件对应目录；不得修改冻结 protocol semantics
- **实现任务**：
  - build TensorRootV2/ManifestV2/CommitV3；签名；防 duplicate submit。
- **测试要求**：
  - 至少 unit test + clean-machine smoke test；涉及网络状态的任务加 integration test。
- **交付物**：
  - `docs/productization/b2-07.md`
- **Definition of Done**：
  - 组件可独立构建/运行；失败有明确 error，不 silent fallback。

### B2-08 — Worker DA upload

- **阶段**：Productization
- **前置依赖**：PHASE_F=PASS
- **允许修改范围**：按组件对应目录；不得修改冻结 protocol semantics
- **实现任务**：
  - 上传 BundleV2 到 3 providers；等待 quorum；提交 attest refs。
- **测试要求**：
  - 至少 unit test + clean-machine smoke test；涉及网络状态的任务加 integration test。
- **交付物**：
  - `docs/productization/b2-08.md`
- **Definition of Done**：
  - 组件可独立构建/运行；失败有明确 error，不 silent fallback。

### B2-09 — Worker crash recovery

- **阶段**：Productization
- **前置依赖**：PHASE_F=PASS
- **允许修改范围**：按组件对应目录；不得修改冻结 protocol semantics
- **实现任务**：
  - job journal；重启恢复 queued/accepted/executed/committed 状态；避免双重结果。
- **测试要求**：
  - 至少 unit test + clean-machine smoke test；涉及网络状态的任务加 integration test。
- **交付物**：
  - `docs/productization/b2-09.md`
- **Definition of Done**：
  - 组件可独立构建/运行；失败有明确 error，不 silent fallback。

### B3-01 — prisma-watcher daemon packaging

- **阶段**：Productization
- **前置依赖**：PHASE_F=PASS
- **允许修改范围**：按组件对应目录；不得修改冻结 protocol semantics
- **实现任务**：
  - 长期运行事件循环；chain subscriptions；Bundle download；persistent state。
- **测试要求**：
  - 至少 unit test + clean-machine smoke test；涉及网络状态的任务加 integration test。
- **交付物**：
  - `docs/productization/b3-01.md`
- **Definition of Done**：
  - 组件可独立构建/运行；失败有明确 error，不 silent fallback。

### B3-02 — Watcher auto-challenge

- **阶段**：Productization
- **前置依赖**：PHASE_F=PASS
- **允许修改范围**：按组件对应目录；不得修改冻结 protocol semantics
- **实现任务**：
  - fraud localization 后自动发送 Graph/Wide dispute messages；deadline scheduler。
- **测试要求**：
  - 至少 unit test + clean-machine smoke test；涉及网络状态的任务加 integration test。
- **交付物**：
  - `docs/productization/b3-02.md`
- **Definition of Done**：
  - 组件可独立构建/运行；失败有明确 error，不 silent fallback。

### B3-03 — Watcher restart recovery

- **阶段**：Productization
- **前置依赖**：PHASE_F=PASS
- **允许修改范围**：按组件对应目录；不得修改冻结 protocol semantics
- **实现任务**：
  - 只从 chain+DA 恢复；no worker local state。
- **测试要求**：
  - 至少 unit test + clean-machine smoke test；涉及网络状态的任务加 integration test。
- **交付物**：
  - `docs/productization/b3-03.md`
- **Definition of Done**：
  - 组件可独立构建/运行；失败有明确 error，不 silent fallback。

### B4-01 — prisma-da daemon packaging

- **阶段**：Productization
- **前置依赖**：PHASE_F=PASS
- **允许修改范围**：按组件对应目录；不得修改冻结 protocol semantics
- **实现任务**：
  - provider register/bond；artifact API；typed chunk proof endpoint。
- **测试要求**：
  - 至少 unit test + clean-machine smoke test；涉及网络状态的任务加 integration test。
- **交付物**：
  - `docs/productization/b4-01.md`
- **Definition of Done**：
  - 组件可独立构建/运行；失败有明确 error，不 silent fallback。

### B4-02 — DA storage engine

- **阶段**：Productization
- **前置依赖**：PHASE_F=PASS
- **允许修改范围**：按组件对应目录；不得修改冻结 protocol semantics
- **实现任务**：
  - NVMe/object store backend；artifact hash；quota；TTL 与 challenge window 对齐。
- **测试要求**：
  - 至少 unit test + clean-machine smoke test；涉及网络状态的任务加 integration test。
- **交付物**：
  - `docs/productization/b4-02.md`
- **Definition of Done**：
  - 组件可独立构建/运行；失败有明确 error，不 silent fallback。

### B4-03 — DA challenge responder

- **阶段**：Productization
- **前置依赖**：PHASE_F=PASS
- **允许修改范围**：按组件对应目录；不得修改冻结 protocol semantics
- **实现任务**：
  - 监听 on-chain DA challenge；deadline 前响应；metrics。
- **测试要求**：
  - 至少 unit test + clean-machine smoke test；涉及网络状态的任务加 integration test。
- **交付物**：
  - `docs/productization/b4-03.md`
- **Definition of Done**：
  - 组件可独立构建/运行；失败有明确 error，不 silent fallback。

### B5-01 — prisma-cli packaging

- **阶段**：Productization
- **前置依赖**：PHASE_F=PASS
- **允许修改范围**：按组件对应目录；不得修改冻结 protocol semantics
- **实现任务**：
  - wallet/task/worker/watcher/DA/dispute/receipt commands；shell completion。
- **测试要求**：
  - 至少 unit test + clean-machine smoke test；涉及网络状态的任务加 integration test。
- **交付物**：
  - `docs/productization/b5-01.md`
- **Definition of Done**：
  - 组件可独立构建/运行；失败有明确 error，不 silent fallback。

### B6-01 — Job API schema

- **阶段**：Productization
- **前置依赖**：PHASE_F=PASS
- **允许修改范围**：按组件对应目录；不得修改冻结 protocol semantics
- **实现任务**：
  - POST /v1/jobs；GET /v1/jobs/{id}；状态机 queued→assigned→executing→committed→verifying→finalized/challenged/failed。
- **测试要求**：
  - 至少 unit test + clean-machine smoke test；涉及网络状态的任务加 integration test。
- **交付物**：
  - `docs/productization/b6-01.md`
- **Definition of Done**：
  - 组件可独立构建/运行；失败有明确 error，不 silent fallback。

### B6-02 — API authentication

- **阶段**：Productization
- **前置依赖**：PHASE_F=PASS
- **允许修改范围**：按组件对应目录；不得修改冻结 protocol semantics
- **实现任务**：
  - testnet API key；rate limit；request size bounds；不做真实计费。
- **测试要求**：
  - 至少 unit test + clean-machine smoke test；涉及网络状态的任务加 integration test。
- **交付物**：
  - `docs/productization/b6-02.md`
- **Definition of Done**：
  - 组件可独立构建/运行；失败有明确 error，不 silent fallback。

### B6-03 — Scheduler worker registry

- **阶段**：Productization
- **前置依赖**：PHASE_F=PASS
- **允许修改范围**：按组件对应目录；不得修改冻结 protocol semantics
- **实现任务**：
  - 维护在线 workers/capabilities/health；过期剔除。
- **测试要求**：
  - 至少 unit test + clean-machine smoke test；涉及网络状态的任务加 integration test。
- **交付物**：
  - `docs/productization/b6-03.md`
- **Definition of Done**：
  - 组件可独立构建/运行；失败有明确 error，不 silent fallback。

### B6-04 — Scheduler matching

- **阶段**：Productization
- **前置依赖**：PHASE_F=PASS
- **允许修改范围**：按组件对应目录；不得修改冻结 protocol semantics
- **实现任务**：
  - 按 profile/GPU capability/availability 简单匹配；第一版 FIFO/priority；禁止过早市场竞价。
- **测试要求**：
  - 至少 unit test + clean-machine smoke test；涉及网络状态的任务加 integration test。
- **交付物**：
  - `docs/productization/b6-04.md`
- **Definition of Done**：
  - 组件可独立构建/运行；失败有明确 error，不 silent fallback。

### B6-05 — Scheduler failure/reassign

- **阶段**：Productization
- **前置依赖**：PHASE_F=PASS
- **允许修改范围**：按组件对应目录；不得修改冻结 protocol semantics
- **实现任务**：
  - accept timeout / worker disconnect / pre-commit failure 时重派；commit 后不可静默换 worker。
- **测试要求**：
  - 至少 unit test + clean-machine smoke test；涉及网络状态的任务加 integration test。
- **交付物**：
  - `docs/productization/b6-05.md`
- **Definition of Done**：
  - 组件可独立构建/运行；失败有明确 error，不 silent fallback。

### B6-06 — Job/VWR presentation

- **阶段**：Productization
- **前置依赖**：PHASE_F=PASS
- **允许修改范围**：按组件对应目录；不得修改冻结 protocol semantics
- **实现任务**：
  - API 返回 task id、worker、GraphIDV2、final root、settlement tx、VWR id。
- **测试要求**：
  - 至少 unit test + clean-machine smoke test；涉及网络状态的任务加 integration test。
- **交付物**：
  - `docs/productization/b6-06.md`
- **Definition of Done**：
  - 组件可独立构建/运行；失败有明确 error，不 silent fallback。

### B7-01 — Faucet service

- **阶段**：Productization
- **前置依赖**：PHASE_F=PASS
- **允许修改范围**：按组件对应目录；不得修改冻结 protocol semantics
- **实现任务**：
  - test token 领取；per-IP/account rate limit；仅测试网。
- **测试要求**：
  - 至少 unit test + clean-machine smoke test；涉及网络状态的任务加 integration test。
- **交付物**：
  - `docs/productization/b7-01.md`
- **Definition of Done**：
  - 组件可独立构建/运行；失败有明确 error，不 silent fallback。

### B7-02 — Docs site

- **阶段**：Productization
- **前置依赖**：PHASE_F=PASS
- **允许修改范围**：按组件对应目录；不得修改冻结 protocol semantics
- **实现任务**：
  - Quickstart、Run Worker、Run Watcher、Run DA、Submit Job、Protocol specs。
- **测试要求**：
  - 至少 unit test + clean-machine smoke test；涉及网络状态的任务加 integration test。
- **交付物**：
  - `docs/productization/b7-02.md`
- **Definition of Done**：
  - 组件可独立构建/运行；失败有明确 error，不 silent fallback。

### B7-03 — Status page

- **阶段**：Productization
- **前置依赖**：PHASE_F=PASS
- **允许修改范围**：按组件对应目录；不得修改冻结 protocol semantics
- **实现任务**：
  - height、validator health、RPC、DA quorum、worker count、job stats；只读。
- **测试要求**：
  - 至少 unit test + clean-machine smoke test；涉及网络状态的任务加 integration test。
- **交付物**：
  - `docs/productization/b7-03.md`
- **Definition of Done**：
  - 组件可独立构建/运行；失败有明确 error，不 silent fallback。

# Phase C — Private Testnet Alpha
### C1-01 — 多主机 validator 拓扑

- **阶段**：Private Testnet Alpha
- **前置依赖**：Productization Ready
- **允许修改范围**：deployment/ops/config；冻结 protocol semantics
- **实现任务**：
  - 4 validators 放到独立主机；最好不同 provider/region；public RPC 与 validator 分离。
- **测试要求**：
  - 每项有 runbook + result JSON/incident log。
- **交付物**：
  - `docs/private-testnet/c1-01.md`
- **Definition of Done**：
  - 测试结果可复现；失败有 root cause / issue。

### C1-02 — 部署 2 个 public RPC/sentry

- **阶段**：Private Testnet Alpha
- **前置依赖**：Productization Ready
- **允许修改范围**：deployment/ops/config；冻结 protocol semantics
- **实现任务**：
  - TLS/限流/metrics；validator 不直接暴露公共 RPC。
- **测试要求**：
  - 每项有 runbook + result JSON/incident log。
- **交付物**：
  - `docs/private-testnet/c1-02.md`
- **Definition of Done**：
  - 测试结果可复现；失败有 root cause / issue。

### C1-03 — 部署 3 个 DA providers

- **阶段**：Private Testnet Alpha
- **前置依赖**：Productization Ready
- **允许修改范围**：deployment/ops/config；冻结 protocol semantics
- **实现任务**：
  - 独立进程/主机；至少一个不同 provider；磁盘监控。
- **测试要求**：
  - 每项有 runbook + result JSON/incident log。
- **交付物**：
  - `docs/private-testnet/c1-03.md`
- **Definition of Done**：
  - 测试结果可复现；失败有 root cause / issue。

### C1-04 — 部署 2 个 Watchers

- **阶段**：Private Testnet Alpha
- **前置依赖**：Productization Ready
- **允许修改范围**：deployment/ops/config；冻结 protocol semantics
- **实现任务**：
  - 至少一个与 Worker/DA 不共享主机/磁盘。
- **测试要求**：
  - 每项有 runbook + result JSON/incident log。
- **交付物**：
  - `docs/private-testnet/c1-04.md`
- **Definition of Done**：
  - 测试结果可复现；失败有 root cause / issue。

### C1-05 — 部署 2 个官方 bootstrap GPU Workers

- **阶段**：Private Testnet Alpha
- **前置依赖**：Productization Ready
- **允许修改范围**：deployment/ops/config；冻结 protocol semantics
- **实现任务**：
  - 只用于保证基础算力，不作为去中心化证据。
- **测试要求**：
  - 每项有 runbook + result JSON/incident log。
- **交付物**：
  - `docs/private-testnet/c1-05.md`
- **Definition of Done**：
  - 测试结果可复现；失败有 root cause / issue。

### C1-06 — 部署 Job API/Scheduler/Faucet

- **阶段**：Private Testnet Alpha
- **前置依赖**：Productization Ready
- **允许修改范围**：deployment/ops/config；冻结 protocol semantics
- **实现任务**：
  - 控制面与 validator 分离；Postgres/queue 备份。
- **测试要求**：
  - 每项有 runbook + result JSON/incident log。
- **交付物**：
  - `docs/private-testnet/c1-06.md`
- **Definition of Done**：
  - 测试结果可复现；失败有 root cause / issue。

### C2-01 — 跨公网 Worker join test

- **阶段**：Private Testnet Alpha
- **前置依赖**：Productization Ready
- **允许修改范围**：deployment/ops/config；冻结 protocol semantics
- **实现任务**：
  - 全新机器按文档安装、bond、接任务、完成 VWR。
- **测试要求**：
  - 每项有 runbook + result JSON/incident log。
- **交付物**：
  - `docs/private-testnet/c2-01.md`
- **Definition of Done**：
  - 测试结果可复现；失败有 root cause / issue。

### C2-02 — 跨公网 Watcher test

- **阶段**：Private Testnet Alpha
- **前置依赖**：Productization Ready
- **允许修改范围**：deployment/ops/config；冻结 protocol semantics
- **实现任务**：
  - 不同网络环境验证 bundle 并正确不挑战 honest task。
- **测试要求**：
  - 每项有 runbook + result JSON/incident log。
- **交付物**：
  - `docs/private-testnet/c2-02.md`
- **Definition of Done**：
  - 测试结果可复现；失败有 root cause / issue。

### C2-03 — Worker disappearance drill

- **阶段**：Private Testnet Alpha
- **前置依赖**：Productization Ready
- **允许修改范围**：deployment/ops/config；冻结 protocol semantics
- **实现任务**：
  - Commit+DA quorum 后关机；Watcher/chain 仍完成流程。
- **测试要求**：
  - 每项有 runbook + result JSON/incident log。
- **交付物**：
  - `docs/private-testnet/c2-03.md`
- **Definition of Done**：
  - 测试结果可复现；失败有 root cause / issue。

### C2-04 — Validator offline drill

- **阶段**：Private Testnet Alpha
- **前置依赖**：Productization Ready
- **允许修改范围**：deployment/ops/config；冻结 protocol semantics
- **实现任务**：
  - 关 1 validator；3 个继续；恢复 catch-up。
- **测试要求**：
  - 每项有 runbook + result JSON/incident log。
- **交付物**：
  - `docs/private-testnet/c2-04.md`
- **Definition of Done**：
  - 测试结果可复现；失败有 root cause / issue。

### C2-05 — DA offline drill

- **阶段**：Private Testnet Alpha
- **前置依赖**：Productization Ready
- **允许修改范围**：deployment/ops/config；冻结 protocol semantics
- **实现任务**：
  - 关 1 DA 仍工作；关 2 DA 阻止 finalize。
- **测试要求**：
  - 每项有 runbook + result JSON/incident log。
- **交付物**：
  - `docs/private-testnet/c2-05.md`
- **Definition of Done**：
  - 测试结果可复现；失败有 root cause / issue。

### C2-06 — Watcher crash drill

- **阶段**：Private Testnet Alpha
- **前置依赖**：Productization Ready
- **允许修改范围**：deployment/ops/config；冻结 protocol semantics
- **实现任务**：
  - challenge 处理中重启 Watcher；从 chain/DA 恢复。
- **测试要求**：
  - 每项有 runbook + result JSON/incident log。
- **交付物**：
  - `docs/private-testnet/c2-06.md`
- **Definition of Done**：
  - 测试结果可复现；失败有 root cause / issue。

### C2-07 — Fraud injection drill

- **阶段**：Private Testnet Alpha
- **前置依赖**：Productization Ready
- **允许修改范围**：deployment/ops/config；冻结 protocol semantics
- **实现任务**：
  - 真实 wide GEMM fraud 在公网环境完成 512-MAC settlement。
- **测试要求**：
  - 每项有 runbook + result JSON/incident log。
- **交付物**：
  - `docs/private-testnet/c2-07.md`
- **Definition of Done**：
  - 测试结果可复现；失败有 root cause / issue。

### C2-08 — 连续运行稳定性

- **阶段**：Private Testnet Alpha
- **前置依赖**：Productization Ready
- **允许修改范围**：deployment/ops/config；冻结 protocol semantics
- **实现任务**：
  - 至少持续运行足够时间捕捉内存泄漏/状态漂移/重复 receipt；具体时长由当期运维计划定义并预先记录。
- **测试要求**：
  - 每项有 runbook + result JSON/incident log。
- **交付物**：
  - `docs/private-testnet/c2-08.md`
- **Definition of Done**：
  - 测试结果可复现；失败有 root cause / issue。

# Phase D — Public Testnet Alpha
### D1-01 — 发布版本化二进制

- **阶段**：Public Testnet Alpha
- **前置依赖**：Private Testnet Alpha PASS
- **允许修改范围**：release/deploy/docs；协议只做 bugfix-compatible changes
- **实现任务**：
  - GitHub Release：prismad / prisma-worker / prisma-watcher / prisma-da / prisma-cli；SHA256。
- **测试要求**：
  - 在干净外部环境完成 smoke / onboarding test。
- **交付物**：
  - `docs/public-testnet/d1-01.md`
- **Definition of Done**：
  - 外部用户不需要改源码即可完成目标。

### D1-02 — 发布容器镜像

- **阶段**：Public Testnet Alpha
- **前置依赖**：Private Testnet Alpha PASS
- **允许修改范围**：release/deploy/docs；协议只做 bugfix-compatible changes
- **实现任务**：
  - 固定 image digest；node/worker/watcher/da；不包含 secret。
- **测试要求**：
  - 在干净外部环境完成 smoke / onboarding test。
- **交付物**：
  - `docs/public-testnet/d1-02.md`
- **Definition of Done**：
  - 外部用户不需要改源码即可完成目标。

### D1-03 — 发布 genesis / chain-id / seeds

- **阶段**：Public Testnet Alpha
- **前置依赖**：Private Testnet Alpha PASS
- **允许修改范围**：release/deploy/docs；协议只做 bugfix-compatible changes
- **实现任务**：
  - 公开 checksum；文档化 reset policy。
- **测试要求**：
  - 在干净外部环境完成 smoke / onboarding test。
- **交付物**：
  - `docs/public-testnet/d1-03.md`
- **Definition of Done**：
  - 外部用户不需要改源码即可完成目标。

### D1-04 — 公开 RPC / Faucet / Job API

- **阶段**：Public Testnet Alpha
- **前置依赖**：Private Testnet Alpha PASS
- **允许修改范围**：release/deploy/docs；协议只做 bugfix-compatible changes
- **实现任务**：
  - 域名/TLS/status；限流。
- **测试要求**：
  - 在干净外部环境完成 smoke / onboarding test。
- **交付物**：
  - `docs/public-testnet/d1-04.md`
- **Definition of Done**：
  - 外部用户不需要改源码即可完成目标。

### D1-05 — 公开 Worker onboarding

- **阶段**：Public Testnet Alpha
- **前置依赖**：Private Testnet Alpha PASS
- **允许修改范围**：release/deploy/docs；协议只做 bugfix-compatible changes
- **实现任务**：
  - 一条文档路径完成安装→faucet→bond→join→job。
- **测试要求**：
  - 在干净外部环境完成 smoke / onboarding test。
- **交付物**：
  - `docs/public-testnet/d1-05.md`
- **Definition of Done**：
  - 外部用户不需要改源码即可完成目标。

### D1-06 — 公开 Watcher onboarding

- **阶段**：Public Testnet Alpha
- **前置依赖**：Private Testnet Alpha PASS
- **允许修改范围**：release/deploy/docs；协议只做 bugfix-compatible changes
- **实现任务**：
  - 安装→sync→verify；默认安全配置。
- **测试要求**：
  - 在干净外部环境完成 smoke / onboarding test。
- **交付物**：
  - `docs/public-testnet/d1-06.md`
- **Definition of Done**：
  - 外部用户不需要改源码即可完成目标。

### D1-07 — 公开 DA onboarding

- **阶段**：Public Testnet Alpha
- **前置依赖**：Private Testnet Alpha PASS
- **允许修改范围**：release/deploy/docs；协议只做 bugfix-compatible changes
- **实现任务**：
  - register/bond/store/respond challenge。
- **测试要求**：
  - 在干净外部环境完成 smoke / onboarding test。
- **交付物**：
  - `docs/public-testnet/d1-07.md`
- **Definition of Done**：
  - 外部用户不需要改源码即可完成目标。

### D1-08 — 公开 Developer quickstart

- **阶段**：Public Testnet Alpha
- **前置依赖**：Private Testnet Alpha PASS
- **允许修改范围**：release/deploy/docs；协议只做 bugfix-compatible changes
- **实现任务**：
  - API key→submit job→observe verify/finalize→query VWR。
- **测试要求**：
  - 在干净外部环境完成 smoke / onboarding test。
- **交付物**：
  - `docs/public-testnet/d1-08.md`
- **Definition of Done**：
  - 外部用户不需要改源码即可完成目标。

### D2-01 — 首个外部 GPU Worker 成功

- **阶段**：Public Testnet Alpha
- **前置依赖**：Private Testnet Alpha PASS
- **允许修改范围**：release/deploy/docs；协议只做 bugfix-compatible changes
- **实现任务**：
  - 非官方控制机器完成真实 job，收到 test reward/VWR。
- **测试要求**：
  - 在干净外部环境完成 smoke / onboarding test。
- **交付物**：
  - `docs/public-testnet/d2-01.md`
- **Definition of Done**：
  - 外部用户不需要改源码即可完成目标。

### D2-02 — 首个外部 Watcher 成功

- **阶段**：Public Testnet Alpha
- **前置依赖**：Private Testnet Alpha PASS
- **允许修改范围**：release/deploy/docs；协议只做 bugfix-compatible changes
- **实现任务**：
  - 非官方 watcher 独立验证真实 job。
- **测试要求**：
  - 在干净外部环境完成 smoke / onboarding test。
- **交付物**：
  - `docs/public-testnet/d2-02.md`
- **Definition of Done**：
  - 外部用户不需要改源码即可完成目标。

### D2-03 — 首个外部 DA provider 成功

- **阶段**：Public Testnet Alpha
- **前置依赖**：Private Testnet Alpha PASS
- **允许修改范围**：release/deploy/docs；协议只做 bugfix-compatible changes
- **实现任务**：
  - 加入 registry 并参与 2-of-3 availability。
- **测试要求**：
  - 在干净外部环境完成 smoke / onboarding test。
- **交付物**：
  - `docs/public-testnet/d2-03.md`
- **Definition of Done**：
  - 外部用户不需要改源码即可完成目标。

### D2-04 — 公开 fraud drill

- **阶段**：Public Testnet Alpha
- **前置依赖**：Private Testnet Alpha PASS
- **允许修改范围**：release/deploy/docs；协议只做 bugfix-compatible changes
- **实现任务**：
  - 计划性 testnet fraud injection；社区可观察 challenge→512-MAC→settlement。
- **测试要求**：
  - 在干净外部环境完成 smoke / onboarding test。
- **交付物**：
  - `docs/public-testnet/d2-04.md`
- **Definition of Done**：
  - 外部用户不需要改源码即可完成目标。

### D2-05 — Known limitations 文档

- **阶段**：Public Testnet Alpha
- **前置依赖**：Private Testnet Alpha PASS
- **允许修改范围**：release/deploy/docs；协议只做 bugfix-compatible changes
- **实现任务**：
  - 明确只支持冻结 Profile、test token 无真实价值、单 proposer censorship ≠ cartel resistance。
- **测试要求**：
  - 在干净外部环境完成 smoke / onboarding test。
- **交付物**：
  - `docs/public-testnet/d2-05.md`
- **Definition of Done**：
  - 外部用户不需要改源码即可完成目标。

# Phase E — Public Testnet Beta / Hardening
### E1-01 — 引入第三方 validators

- **阶段**：Public Testnet Beta
- **前置依赖**：Public Testnet Alpha stable
- **允许修改范围**：ops/product/params；重大 protocol 变化需新 version
- **实现任务**：
  - 逐步降低官方 voting power 集中度；记录 validator operator runbook。
- **测试要求**：
  - 预声明场景 + 结果记录。
- **交付物**：
  - `docs/testnet-beta/e1-01.md`
- **Definition of Done**：
  - hardening 结果可量化并进入 Mainnet Candidate 决策。

### E1-02 — 扩大外部 DA / Watcher

- **阶段**：Public Testnet Beta
- **前置依赖**：Public Testnet Alpha stable
- **允许修改范围**：ops/product/params；重大 protocol 变化需新 version
- **实现任务**：
  - 多云多地域；观察 availability/challenge 稳定性。
- **测试要求**：
  - 预声明场景 + 结果记录。
- **交付物**：
  - `docs/testnet-beta/e1-02.md`
- **Definition of Done**：
  - hardening 结果可量化并进入 Mainnet Candidate 决策。

### E1-03 — Worker reliability scoring

- **阶段**：Public Testnet Beta
- **前置依赖**：Public Testnet Alpha stable
- **允许修改范围**：ops/product/params；重大 protocol 变化需新 version
- **实现任务**：
  - 基于 uptime/job success/latency 形成调度权重；不改变 consensus。
- **测试要求**：
  - 预声明场景 + 结果记录。
- **交付物**：
  - `docs/testnet-beta/e1-03.md`
- **Definition of Done**：
  - hardening 结果可量化并进入 Mainnet Candidate 决策。

### E1-04 — Scheduler fairness / backpressure

- **阶段**：Public Testnet Beta
- **前置依赖**：Public Testnet Alpha stable
- **允许修改范围**：ops/product/params；重大 protocol 变化需新 version
- **实现任务**：
  - 队列、公平性、限流、失败重派；防单用户独占。
- **测试要求**：
  - 预声明场景 + 结果记录。
- **交付物**：
  - `docs/testnet-beta/e1-04.md`
- **Definition of Done**：
  - hardening 结果可量化并进入 Mainnet Candidate 决策。

### E1-05 — Upgrade rehearsal

- **阶段**：Public Testnet Beta
- **前置依赖**：Public Testnet Alpha stable
- **允许修改范围**：ops/product/params；重大 protocol 变化需新 version
- **实现任务**：
  - 协议兼容升级 / binary upgrade / state migration / rollback runbook。
- **测试要求**：
  - 预声明场景 + 结果记录。
- **交付物**：
  - `docs/testnet-beta/e1-05.md`
- **Definition of Done**：
  - hardening 结果可量化并进入 Mainnet Candidate 决策。

### E1-06 — Chaos testing

- **阶段**：Public Testnet Beta
- **前置依赖**：Public Testnet Alpha stable
- **允许修改范围**：ops/product/params；重大 protocol 变化需新 version
- **实现任务**：
  - RPC overload、DA disk loss、worker churn、watcher crash、validator restart。
- **测试要求**：
  - 预声明场景 + 结果记录。
- **交付物**：
  - `docs/testnet-beta/e1-06.md`
- **Definition of Done**：
  - hardening 结果可量化并进入 Mainnet Candidate 决策。

### E1-07 — Security review

- **阶段**：Public Testnet Beta
- **前置依赖**：Public Testnet Alpha stable
- **允许修改范围**：ops/product/params；重大 protocol 变化需新 version
- **实现任务**：
  - domain separation、signature replay、proof confusion、resource DoS、key management。
- **测试要求**：
  - 预声明场景 + 结果记录。
- **交付物**：
  - `docs/testnet-beta/e1-07.md`
- **Definition of Done**：
  - hardening 结果可量化并进入 Mainnet Candidate 决策。

### E1-08 — 经济参数回放

- **阶段**：Public Testnet Beta
- **前置依赖**：Public Testnet Alpha stable
- **允许修改范围**：ops/product/params；重大 protocol 变化需新 version
- **实现任务**：
  - bond/challenge bond/reward/DA penalty/worker price；test token only。
- **测试要求**：
  - 预声明场景 + 结果记录。
- **交付物**：
  - `docs/testnet-beta/e1-08.md`
- **Definition of Done**：
  - hardening 结果可量化并进入 Mainnet Candidate 决策。

# 4. Research Reopen Gate
只有以下情况允许重新开启 `research/*` 分支。普通 bug、接口缺失、部署问题、代码量大、运行慢，不构成研究理由。
| 触发条件 | 是否允许研究 | 先做什么 |
|---|---|---|
| 512-MAC / bounded dispute 在真实 witness/gas 下无法放进合理 chain envelope，且不是实现 bug | 是 | 先确认 witness 压缩 / proof sharing / gas accounting 已充分工程优化 |
| Watcher 被迫 full GEMM recompute，40-round Freivalds + exact cheap-op 无法满足 | 是 | 先排除实现错误与 bundle layout 问题 |
| BundleV2 大到工程分块/压缩/lazy fetch 仍不可用 | 可能 | 先做 chunk streaming / compression / selective proof fetch |
| Public Testnet worker 性能/稳定性问题 | 通常否 | 先做 runtime/IO/batching/driver/ops 优化 |
| 要支持新模型 / 新量化 / MoE | 是，但独立新版本 | 不得阻塞当前冻结 Profile 与 Testnet |
| 要把 useful compute 进入 PoUW consensus | 是，独立大 Phase | 不得与当前 settlement network 主线混在一起 |

# 5. 最终交付物
## 5.1 Phase F 最终交付
- 被冻结、版本化的 Canonical Tensor/Graph V2 + A13W10 wide arithmetic。
- Permissionless Watcher V2：83 wide GEMM 用 Freivalds，cheap ops exact，`full_gemm_calls=0`。
- DA_REPLICA_V1 对 GraphVerificationBundleV2 的 2-of-3 availability。
- 真实 wide GEMM fraud 可压缩到 deterministic 512-MAC chain arbitration。
- 真实 RoPE fraud bounded arbitration。
- Honest real block exactly one VWR V3；fraud/availability failure zero VWR。
- 4-validator devnet 在 honest/fraud/false/DA/censor/restart 场景全部 converge。
- `PHASE_F = PASS` + freeze tag + full lineage report。
## 5.2 Public Testnet Alpha 最终交付
- `prismad`
- `prisma-worker`
- `prisma-watcher`
- `prisma-da`
- `prisma-cli`
- Public RPC / Faucet / Job API / Scheduler
- Developer docs / Operator docs / Protocol specs
- 至少一个外部 Worker、Watcher（最好 DA）完成真实接入
## 5.3 真正完成的端到端定义
陌生开发者：
```text
API Key → Submit AI Job → Assigned Worker → Execute → CommitV3 → DA → Watcher → Finalize → VWR
```
陌生 GPU 提供者：
```text
Install prisma-worker → Join → Bond → Accept Job → Execute → Commit → Earn Test Reward
```
恶意 Worker：
```text
Bad Computation → Self-consistent Commit → Watcher Detects → Graph Bisection → Wide Tile → K-step → 512 MAC → Slash/Refund/Reward → NO VWR
```

# 6. AI 每次执行任务时的输出格式
每次 AI/Codex 完成一个 Task ID，最终必须至少输出：
- `Task ID`
- `branch / head`
- 改动文件列表
- 实现摘要
- 测试命令
- 真实测试结果
- 新增 artifacts
- 兼容性结果
- 是否触发新的 blocker
- 最终 verdict：PASS / PARTIAL / FAIL / BLOCKED
- 下一任务 ID（只能是依赖已满足的任务）

# 7. 推荐当前立即开始的任务
当前下一任务不是 Watcher，也不是产品化。Critical path 是：
```text
A1-01
  ↓
A1-02 → A1-03 → A1-04 → A1-05 → A1-06 → A1-07 → A1-08
  ↓
A2-01 → A2-02 → A2-03 → A2-04 → A2-05 → A2-06 → A2-07 → A2-08
  ↓
A3 Watcher
  ↓
A4 DA
  ↓
A5 Query/Gas
  ↓
A6 4-validator E2E
  ↓
A7 PHASE_F PASS / Freeze
```

**当前建议立即交给 Codex：`A1-01` 到 `A2-08`，目标 milestone：`F.5C-Dispute = PASS`。**

---
End of AI Engineering Master Roadmap.
