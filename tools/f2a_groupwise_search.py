"""EXPERIMENTAL / NON-PROTOCOL — F.2A search, freeze and heldout gate.

Subcommands (each stage is separately logged and committed):

  --anchor     prove the all-OFF policy reproduces the F.1 converted graph
  --baseline   evaluate the F.1 policy on the tuning set
  --search     per-site sensitivity (K first) then greedy combination on
               the TUNING set only
  --freeze     write docs/phase-f2a-frozen-policy.json + PolicyID
  --heldout    load the FINAL HELDOUT split (requires --unlock-heldout)
               and run the gate ONCE against the frozen policy
  --ablate     remove each groupwise site from the frozen policy and
               re-measure on TUNING

Data discipline: calibration derives scales; tuning selects the policy;
heldout is read only with --unlock-heldout and only after freezing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "compute" / "canonical" / "python"))

import canonical_ref as R  # noqa: E402
import f1_canonical_numpy as N  # noqa: E402
from convert_qwen3_block import (  # noqa: E402
    Conformer, K_HEAD_FACTOR, instrumented_forward, mse_optimal_scale, to_fx,
)
from f2a_groupwise_executor import (  # noqa: E402
    GroupPolicy, expected_macs, run_block, slice_steps_mse,
)

SITES = ["k", "q", "h", "ctx", "h2", "hm"]
GROUPS = [128, 64, 32, 16, 8]
P_STEP = max(1, int(np.ceil(1.02 / 127 * (1 << 20))))
SEARCH_CASES = 8          # tuning cases used during the sensitivity search
FROZEN_PATH = REPO / "docs" / "phase-f2a-frozen-policy.json"
SEARCH_PATH = REPO / "docs" / "phase-f2a-search-results.json"
HELDOUT_PATH = REPO / "docs" / "phase-f2a-heldout-results.json"
COMPLEXITY_PATH = REPO / "docs" / "phase-f2a-complexity.json"
OUTLIER_PATH = REPO / "docs" / "phase-f2a-outlier-analysis.json"


def load_split(name: str):
    return json.loads((REPO / "testdata" / f"qwen3_f2a_{name}.json").read_text(encoding="utf-8"))["cases"]


def wscale(w: np.ndarray) -> int:
    return max(1, int(np.ceil(np.max(np.abs(w)) * (1 << 20) / 127)))


def collect_policy_data(conf: Conformer):
    """Calibration over the F.2A calibration split: base steps (F.1 policy
    shape), weight steps and per-channel/per-slice samples for groupwise
    step derivation. Calibration data ONLY."""
    cases = load_split("calibration")
    sites_flat = ("h", "h2", "hm")
    sites_head = ("q", "k", "ctx")
    globals_ = {k: [] for k in ("h", "q", "k", "v", "ctx", "h2", "hm")}
    chan_flat = {k: [] for k in sites_flat}      # per case [seq, C]
    chan_head = {k: [] for k in sites_head}      # per case [seq, heads, C]
    for case in cases:
        hidden = conf.embedding(case["token_ids"])
        _, _, raw = instrumented_forward(conf, hidden, collect=True)
        for k in globals_:
            flat = np.abs(raw[k]).reshape(-1)
            flat = flat[:: max(1, flat.size // 100_000)]
            globals_[k].append(flat)
        for k in sites_flat:
            chan_flat[k].append(np.abs(raw[k]))          # [seq, C]
        for k in sites_head:
            chan_head[k].append(np.abs(raw[k]))          # [seq, heads, C]

    base_steps = {k: mse_optimal_scale(np.concatenate(v))[0] for k, v in globals_.items()}
    base_steps["p"] = P_STEP
    # F.1 policy shape: per-head k factor (link-level tuned)
    k_head_steps = []
    for h in range(8):
        vals = np.stack([c[:, h, :].reshape(-1) for c in chan_head["k"]])
        step = mse_optimal_scale(vals.reshape(-1))[0]
        k_head_steps.append(max(1, int(round(step * K_HEAD_FACTOR))))
    # F.1 uses PER-SLICE weight scales for the per-head matrices.
    wsteps = {"wq": [wscale(conf.q_slice(h)) for h in range(16)],
              "wk": [wscale(conf.k_slice(h)) for h in range(8)],
              "wv": [wscale(conf.v_slice(h)) for h in range(8)],
              "wo": [wscale(conf.o_slice(h)) for h in range(16)],
              "wg": wscale(conf.gate_slice()), "wu": wscale(conf.up_slice()),
              "wd": wscale(conf.down_slice())}
    # NOTE: F.1 used per-head k steps; keep the per-head structure for q too
    # (F.1 used a global q step; per-head MSE is the same policy family and
    # is what the groupwise derivation generalizes).
    q_head_steps = []
    for h in range(16):
        vals = np.stack([c[:, h, :].reshape(-1) for c in chan_head["q"]])
        q_head_steps.append(mse_optimal_scale(vals.reshape(-1))[0])
    return {
        "base_steps": base_steps,
        "k_head_steps": k_head_steps,
        "q_head_steps": q_head_steps,
        "wsteps": wsteps,
        "chan_flat": chan_flat,
        "chan_head": chan_head,
        "calibration_size": len(cases),
    }


def site_steps_for(pol_data, site: str, group: int):
    """Per-slice steps for one groupwise site (calibration-only)."""
    if group <= 0:
        return None
    per = {"h": 1024, "h2": 1024, "hm": 3072, "q": 128, "k": 128, "ctx": 128}[site]
    if site in ("h", "h2", "hm"):
        per_case = pol_data["chan_flat"][site]           # list of [seq, C]
        stacked = np.concatenate([c for c in per_case], axis=0)   # [N, C]
        out = []
        for lo in range(0, per, group):
            hi = min(per, lo + group)
            out.append(slice_steps_mse([stacked[:, lo:hi]])[0])
        return out
    # head-structured sites: one step list per head
    per_case = pol_data["chan_head"][site]               # list of [seq, heads, C]
    heads = per_case[0].shape[1]
    all_steps = []
    for h in range(heads):
        stacked = np.concatenate([c[:, h, :] for c in per_case], axis=0)  # [N, C]
        steps = []
        for lo in range(0, per, group):
            hi = min(per, lo + group)
            steps.append(slice_steps_mse([stacked[:, lo:hi]])[0])
        all_steps.append(steps)
    return all_steps


def build_site_steps(pol_data, policy: GroupPolicy):
    out = {}
    for site in SITES:
        g = policy.group(site)
        if g > 0:
            out[site] = site_steps_for(pol_data, site, g)
    return out


def evaluate(conf: Conformer, pol_data, policy: GroupPolicy, cases, max_cases=None):
    """Integer simulation vs the float64 reference on the given cases."""
    site_steps = build_site_steps(pol_data, policy)
    steps = dict(pol_data["base_steps"])
    for h, step in enumerate(pol_data["k_head_steps"]):
        steps[f"kh{h}"] = step
    worst_cos, worst_abs, mean_abs = 1.0, 0.0, 0.0
    n = 0
    sat_total = 0
    for case in cases[: max_cases or len(cases)]:
        hidden = conf.embedding(case["token_ids"])
        ref, _, _ = instrumented_forward(conf, hidden, collect=True)
        base = dict(pol_data["base_steps"])
        # the executor takes the per-head k steps through base_steps["k"] of
        # head 0's value; k head differences are applied via k_head_steps
        # uniformly below (F.1 policy effect) by overriding per-head steps.
        y, stats = run_block_cached(conf, hidden, policy, site_steps, pol_data, base)
        yf = y.astype(np.float64) / (1 << 20)
        cos = float(np.dot(yf.reshape(-1), ref.reshape(-1)) /
                    (np.linalg.norm(yf) * np.linalg.norm(ref)))
        absx = float(np.max(np.abs(yf - ref)))
        worst_cos = min(worst_cos, cos)
        worst_abs = max(worst_abs, absx)
        mean_abs += absx
        sat_total += stats.saturations
        n += 1
    return {"worst_cosine": worst_cos, "worst_max_abs": worst_abs,
            "mean_max_abs": mean_abs / max(n, 1), "cases": n,
            "saturations": sat_total}


def run_block_cached(conf, hidden, policy, site_steps, pol_data, base):
    """run_block with the F.1 per-head k steps applied when k is not
    groupwise (the executor's base path uses base_steps['k'])."""
    if not policy.enabled("k"):
        # emulate per-head k steps: temporarily provide per-head lists
        site_steps = dict(site_steps)
        site_steps["k"] = [[s] for s in pol_data["k_head_steps"]]
    if not policy.enabled("q"):
        site_steps2 = dict(site_steps)
        site_steps2.setdefault("q", [[s] for s in pol_data["q_head_steps"]])
        site_steps = site_steps2
    return run_block(conf, hidden, policy, site_steps, pol_data["wsteps"], base)


def complexity(policy: GroupPolicy) -> dict:
    """Structural cost of a policy (see section 38-48 of the spec)."""
    baseline = {"nodes": 310, "gemm_nodes": 83, "requantize_nodes": 126,
                "manifest_leaves": 310}
    added_gemm = 0
    added_requant = 0
    added_add = 0
    per = {"k": 128, "q": 128, "ctx": 128, "h": 1024, "h2": 1024, "hm": 3072}
    consumers = {"k": 16, "q": 16, "ctx": 16, "h": 32, "h2": 2, "hm": 1}
    for site, g in policy.groups.items():
        if g <= 0:
            continue
        slices = per[site] // g
        n = consumers[site]
        added_gemm += n * (slices - 1)
        added_requant += n * (slices - 1)
        added_add += n * (slices - 1)
    nodes = baseline["nodes"] + added_gemm + added_requant + added_add + \
        sum(1 for g in policy.groups.values() if g > 0) * 24  # slice_view nodes est.
    out = {
        "nodes": nodes, "nodes_x": nodes / baseline["nodes"],
        "gemm_nodes": baseline["gemm_nodes"] + added_gemm,
        "gemm_x": (baseline["gemm_nodes"] + added_gemm) / baseline["gemm_nodes"],
        "requantize_nodes": baseline["requantize_nodes"] + added_requant,
        "add_nodes_added": added_add,
        "manifest_leaves": nodes,
        "freivalds_checks": baseline["gemm_nodes"] + added_gemm,
        "dispute_depth": int(np.ceil(np.log2(nodes))) - int(np.ceil(np.log2(baseline["nodes"]))),
        "mac_invariant": "sum M*N*K_s == M*N*K (asserted at run time)",
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    for flag in ("anchor", "baseline", "search", "freeze", "heldout", "ablate"):
        parser.add_argument(f"--{flag}", action="store_true")
    parser.add_argument("--unlock-heldout", action="store_true")
    args = parser.parse_args()

    conf = Conformer(0)
    t0 = time.time()
    pol_data = collect_policy_data(conf)
    print(f"calibration done ({pol_data['calibration_size']} cases, {time.time()-t0:.1f}s)")
    print("base steps:", {k: round(v / (1 << 20), 6) for k, v in pol_data["base_steps"].items()})

    tuning = load_split("tuning")

    if args.anchor:
        # all-OFF policy must equal the F.1 converted graph bit-for-bit
        case = tuning[0]
        hidden = conf.embedding(case["token_ids"])
        y, stats = run_block_cached(conf, hidden, GroupPolicy({}), {}, pol_data,
                                    dict(pol_data["base_steps"]))
        assert stats.mac_total == expected_macs(16), (stats.mac_total, expected_macs(16))
        import json as _json
        from convert_qwen3_block import build_converted_graph
        chosen = {k: {"step_fx": int(v), "percentile": 100.0, "mse": 0.0, "clipped_fraction": 0.0,
                      "calibration_max_abs": 0.0} for k, v in pol_data["base_steps"].items()}
        for i, step in enumerate(pol_data["k_head_steps"]):
            chosen[f"k_h{i}"] = {"step_fx": step, "percentile": 100.0, "mse": 0.0,
                                 "clipped_fraction": 0.0, "calibration_max_abs": 0.0}
        for i, step in enumerate(pol_data["q_head_steps"]):
            chosen[f"q_h{i}"] = {"step_fx": step, "percentile": 100.0, "mse": 0.0,
                                 "clipped_fraction": 0.0, "calibration_max_abs": 0.0}
        snake, go, plan, tdata, gid = build_converted_graph(conf, 16, {"chosen": chosen, "p": 1.0})
        inputs = dict(tdata)
        inputs[0] = to_fx(hidden).astype(np.int32)
        sc = _json.loads(_json.dumps(snake, default=lambda b: b.hex()))
        sc["inputs"][0]["root"] = N.tensor_root_np(snake["inputs"][0]["desc"], inputs[0])
        for gi in sc["inputs"][1:]:
            gi["root"] = bytes.fromhex(gi["root"])
        out = N.execute_graph_np(sc, inputs, rope_tables={2: tdata[2]})
        graph_y = out["tensors"][(1, snake["outputs"][0]["index"])].astype(np.int64)
        delta = int(np.max(np.abs(graph_y - y)))
        ndiff = int(np.count_nonzero(graph_y != y))
        print(f"anchor 2: all-OFF policy vs F.1 graph max |delta| = {delta} "
              f"({ndiff} of {y.size} elements differ)")
        assert delta == 0, "executor does not reproduce the F.1 graph"
        print("anchor 2 OK: bit-identical to the F.1 converted graph")

    if args.baseline or args.search:
        base_policy = GroupPolicy({})
        base = evaluate(conf, pol_data, base_policy, tuning, SEARCH_CASES)
        print("baseline (F.1 policy) on tuning subset:", base)
        results = {"baseline": base, "search_cases": SEARCH_CASES,
                   "marginal": [], "greedy": []}

        if args.search:
            # sensitivity: one site at a time (K first per the spec)
            order = ["k", "q", "h", "ctx", "h2", "hm"]
            for site in order:
                for g in GROUPS:
                    if g > {"k": 128, "q": 128, "ctx": 128, "h": 1024, "h2": 1024, "hm": 3072}[site]:
                        continue
                    pol = GroupPolicy({site: g})
                    res = evaluate(conf, pol_data, pol, tuning, SEARCH_CASES)
                    entry = {"site": site, "group": g, **res,
                             "complexity": complexity(pol),
                             "gain": res["worst_cosine"] - base["worst_cosine"]}
                    results["marginal"].append(entry)
                    print(f"marginal {site:4s} g={g:4d}: cos={res['worst_cosine']:.5f} "
                          f"abs={res['worst_max_abs']:.3f} gain={entry['gain']:+.4f} "
                          f"nodes x{entry['complexity']['nodes_x']:.2f}")
            SEARCH_PATH.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
            print("written:", SEARCH_PATH)

    if args.freeze:
        policy = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))["policy"] \
            if FROZEN_PATH.exists() else None
        if policy is None:
            raise SystemExit("freeze requires a chosen policy: edit docs/phase-f2a-frozen-policy.json")
        print("frozen policy present:", policy)

    if args.heldout:
        if not args.unlock_heldout:
            raise SystemExit("heldout is locked; pass --unlock-heldout explicitly")
        frozen = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
        pol = GroupPolicy(frozen["policy"]["groups"])
        heldout = load_split("heldout")
        res = evaluate(conf, pol_data, pol, heldout)
        verdict = "PASS" if res["worst_cosine"] >= 0.995 and res["worst_max_abs"] <= 0.05 else "FAIL"
        out = {"policy_id": frozen["policy_id"], "frozen": True, "heldout_size": len(heldout),
               **res, "verdict": verdict,
               "thresholds": {"worst_cosine": 0.995, "worst_max_abs": 0.05}}
        HELDOUT_PATH.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
        print("heldout:", out)
        print("written:", HELDOUT_PATH)

    if args.ablate:
        frozen = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
        groups = dict(frozen["policy"]["groups"])
        full = evaluate(conf, pol_data, GroupPolicy(groups), tuning)
        rows = [{"policy": "frozen", **full}]
        for site in [s for s, g in groups.items() if g > 0]:
            reduced = dict(groups)
            reduced[site] = 0
            res = evaluate(conf, pol_data, GroupPolicy(reduced), tuning)
            rows.append({"policy": f"without {site}", **res})
            print(f"ablation without {site}: cos={res['worst_cosine']:.5f} abs={res['worst_max_abs']:.3f}")
        (REPO / "docs" / "phase-f2a-ablation.json").write_text(
            json.dumps(rows, indent=1) + "\n", encoding="utf-8")
        print("ablation written")


if __name__ == "__main__":
    main()
