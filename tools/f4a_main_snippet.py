def main() -> None:
    parser = argparse.ArgumentParser()
    for flag in ("anchor", "safety", "calibrate", "selection", "freeze", "heldout"):
        parser.add_argument(f"--{flag}", action="store_true")
    parser.add_argument("--unlock-heldout", action="store_true")
    args = parser.parse_args()

    conf = Conformer(0)
    cases, samples, head_samples = collect_calibration(conf)
    pairs = pair_family()
    assert len(pairs) == 21, pairs
    print(f"calibration collected ({len(cases)} cases); family: {len(pairs)} pairs")

    if args.safety:
        from convert_qwen3_block import build_converted_graph
        j8 = build_plan(conf, samples, head_samples, 8, 8)
        chosen = {k: {"step_fx": v["step_fx"], "percentile": v["percentile"], "mse": v["mse"],
                      "clipped_fraction": 0.0, "calibration_max_abs": 0.0}
                  for k, v in j8["sites"].items() if k not in ("k_heads", "q_heads", "p")}
        chosen["p"] = {"step_fx": j8["sites"]["p"]["step_fx"], "percentile": 100.0, "mse": 0.0,
                       "clipped_fraction": 0.0, "calibration_max_abs": 1.0}
        for i, st in enumerate(j8["sites"]["k_heads"]):
            chosen[f"k_h{i}"] = {"step_fx": st, "percentile": 100.0, "mse": 0.0,
                                 "clipped_fraction": 0.0, "calibration_max_abs": 0.0}
        for i, st in enumerate(j8["sites"]["q_heads"]):
            chosen[f"q_h{i}"] = {"step_fx": st, "percentile": 100.0, "mse": 0.0,
                                 "clipped_fraction": 0.0, "calibration_max_abs": 0.0}
        snake, _go, _plan, tdata, _gid = build_converted_graph(conf, 16, {"chosen": chosen, "p": 1.0})
        ks = []
        for node in snake["nodes"]:
            if node["operator_id"] != "GEMM_INT8_V1":
                continue
            a_ref = node["inputs"][0]
            kdim = (tdata[a_ref["index"]].shape[1] if a_ref["kind"] == 0
                    else snake["nodes"][a_ref["index"]]["output"]["shape"][1])
            ks.append(int(kdim))
        k_max = max(ks)
        table = {}
        for a, w in pairs:
            bound = max_safe_k_joint(a, w)
            table[f"A{a}W{w}"] = {"a_bits": a, "w_bits": w,
                                  "a_max_magnitude": 1 << (a - 1),
                                  "w_max_magnitude": 1 << (w - 1),
                                  "worst_product": 1 << (a + w - 2),
                                  "max_safe_k": bound, "real_graph_kmax": k_max,
                                  "safe": bool(bound >= k_max)}
        grid = {}
        for a in range(8, 14):
            grid[f"A{a}"] = {f"W{w}": bool(max_safe_k_joint(a, w) >= k_max) for w in range(8, 14)}
        doc = {"formula": "max_safe_k(a,w) = floor((2^31-1) / 2^(a+w-2))",
               "real_graph": {"gemm_nodes": len(ks), "k_max": k_max,
                              "all_k_values": sorted(set(ks))},
               "candidates": table, "safety_grid": grid,
               "diagnostic_excluded": {"A13W10": {"max_safe_k": max_safe_k_joint(13, 10),
                                                   "status": "UNSAFE_DIAGNOSTIC_ONLY"}},
               "note": "admission uses the THEORETICAL bound; observed ranges are recorded separately"}
        (REPO / "docs" / "phase-f4a-accumulator-safety.json").write_text(
            json.dumps(doc, indent=1) + "\n", encoding="utf-8")
        print(f"safety frontier written; K_max={k_max}; "
              f"frontier (a+w=21) safe: {all(table['A' + str(a) + 'W' + str(21 - a)]['safe'] for a in range(8, 14))}")

    if args.anchor:
        import json as _json
        from convert_qwen3_block import build_converted_graph
        j8 = build_plan(conf, samples, head_samples, 8, 8)
        chosen = {k: {"step_fx": v["step_fx"], "percentile": v["percentile"], "mse": v["mse"],
                      "clipped_fraction": 0.0, "calibration_max_abs": 0.0}
                  for k, v in j8["sites"].items() if k not in ("k_heads", "q_heads", "p")}
        chosen["p"] = {"step_fx": j8["sites"]["p"]["step_fx"], "percentile": 100.0, "mse": 0.0,
                       "clipped_fraction": 0.0, "calibration_max_abs": 1.0}
        for i, st in enumerate(j8["sites"]["k_heads"]):
            chosen[f"k_h{i}"] = {"step_fx": st, "percentile": 100.0, "mse": 0.0,
                                 "clipped_fraction": 0.0, "calibration_max_abs": 0.0}
        for i, st in enumerate(j8["sites"]["q_heads"]):
            chosen[f"q_h{i}"] = {"step_fx": st, "percentile": 100.0, "mse": 0.0,
                                 "clipped_fraction": 0.0, "calibration_max_abs": 0.0}
        snake, _go, _plan, tdata, _gid = build_converted_graph(conf, 16, {"chosen": chosen, "p": 1.0})
        hidden = conf.embedding(cases[0]["token_ids"])
        inputs = dict(tdata)
        inputs[0] = to_fx(hidden).astype(np.int32)
        sc = _json.loads(_json.dumps(snake, default=lambda b: b.hex()))
        sc["inputs"][0]["root"] = N.tensor_root_np(snake["inputs"][0]["desc"], inputs[0])
        for gi in sc["inputs"][1:]:
            gi["root"] = bytes.fromhex(gi["root"])
        out = N.execute_graph_np(sc, inputs, rope_tables={2: tdata[2]})
        graph_y = out["tensors"][(1, snake["outputs"][0]["index"])].astype(np.int64)
        y = run_block_bits(conf, hidden, j8, {"acc_abs_max": 0, "fx_saturations": 0})
        delta = int(np.max(np.abs(graph_y - y)))
        print(f"anchor 1: A8W8 executor vs graph max |delta| = {delta}")
        assert delta == 0, "A8W8 does not reproduce the converted graph"

        f3a_heldout = json.loads((REPO / "testdata" / "qwen3_f3a_heldout.json")
                                 .read_text(encoding="utf-8"))["cases"][:8]
        ref_cache = {}
        plan_13_8 = build_plan(conf, samples, head_samples, 13, 8)
        res = evaluate_with_stats(conf, plan_13_8, f3a_heldout,
                                  {"acc_abs_max": 0, "fx_saturations": 0}, ref_cache)
        print(f"anchor 2: A13W8 on F.3A heldout subset cos={res['worst_cosine']:.5f} "
              f"abs={res['worst_max_abs']:.4f} (F.3A: ~0.9994/0.09)")
        assert abs(res["worst_cosine"] - 0.9994) < 0.003 and abs(res["worst_max_abs"] - 0.09) < 0.03, res
        plan_13_10 = build_plan(conf, samples, head_samples, 13, 10)
        res = evaluate_with_stats(conf, plan_13_10, f3a_heldout,
                                  {"acc_abs_max": 0, "fx_saturations": 0}, ref_cache)
        print(f"anchor 3 (UNSAFE_DIAGNOSTIC_ONLY): A13W10 cos={res['worst_cosine']:.5f} "
              f"abs={res['worst_max_abs']:.4f} (F.3A diagnostic: ~0.99995/0.034)")
        assert res["worst_max_abs"] < 0.05, res

    if args.calibrate:
        doc = {"model": "Qwen/Qwen3-0.6B-Base", "revision": "57ca99e94acb83175495aa2c6b6b0cc498170924",
               "layer": 0, "seq": 16,
               "ranking": ["minimize W_bits", "then minimize A_bits", "then logical bytes"],
               "candidates": {}}
        for a, w in pairs:
            plan = build_plan(conf, samples, head_samples, a, w)
            blob = json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
            plan["policy_id"] = hashlib.sha256(blob).hexdigest()
            doc["candidates"][f"A{a}W{w}"] = plan
        (REPO / "docs" / "phase-f4a-candidate-policies.json").write_text(
            json.dumps(doc, indent=1) + "\n", encoding="utf-8")
        print("21 candidate policies written")

    if args.selection:
        doc = json.loads((REPO / "docs" / "phase-f4a-candidate-policies.json")
                         .read_text(encoding="utf-8"))
        selection = load_split("selection")
        ref_cache = {}
        results = {}
        names = sorted(doc["candidates"], key=lambda n: (int(n.split("W")[1]), int(n[1:n.index("W")])))
        for name in names:
            plan = doc["candidates"][name]
            res = evaluate_with_stats(conf, plan, selection,
                                      {"acc_abs_max": 0, "fx_saturations": 0}, ref_cache)
            v = "PASS" if res["worst_cosine"] >= 0.995 and res["worst_max_abs"] <= 0.05 else "FAIL"
            results[name] = {**res, "verdict": v, "policy_id": plan["policy_id"],
                             "max_safe_k": plan["theoretical_max_safe_k"]}
            print(f"selection {name:7s}: cos={res['worst_cosine']:.5f} abs={res['worst_max_abs']:.4f} "
                  f">0.05={res['count_gt_005']:5d} -> {v}")
        (REPO / "docs" / "phase-f4a-selection-results.json").write_text(
            json.dumps(results, indent=1) + "\n", encoding="utf-8")

    if args.freeze:
        doc = json.loads((REPO / "docs" / "phase-f4a-candidate-policies.json")
                         .read_text(encoding="utf-8"))
        sel = json.loads((REPO / "docs" / "phase-f4a-selection-results.json")
                         .read_text(encoding="utf-8"))
        family = {"family": [f"A{a}W{w}" for a, w in pairs],
                  "ranking": doc["ranking"], "candidates": {}}
        for name, plan in doc["candidates"].items():
            family["candidates"][name] = {
                "a_bits": plan["bits"], "w_bits": plan["w_bits"],
                "policy_id": plan["policy_id"],
                "max_safe_k": plan["theoretical_max_safe_k"],
                "safe": bool(plan["theoretical_max_safe_k"] >= 3072),
                "selection": {k: sel[name][k] for k in
                              ("worst_cosine", "worst_max_abs", "verdict")}}
        blob = json.dumps(family["candidates"], sort_keys=True, separators=(",", ":")).encode()
        family["family_sha256"] = hashlib.sha256(blob).hexdigest()
        (REPO / "docs" / "phase-f4a-frozen-family.json").write_text(
            json.dumps(family, indent=1) + "\n", encoding="utf-8")
        print("frozen family sha256:", family["family_sha256"][:16])

    if args.heldout:
        fam = json.loads((REPO / "docs" / "phase-f4a-frozen-family.json")
                         .read_text(encoding="utf-8"))
        pol = json.loads((REPO / "docs" / "phase-f4a-candidate-policies.json")
                         .read_text(encoding="utf-8"))
        heldout = load_split("heldout")
        ref_cache = {}
        results = {}
        passers = []
        for name in fam["family"]:
            entry = fam["candidates"][name]
            plan = pol["candidates"][name]
            payload = {"bits": plan["bits"], "w_bits": plan["w_bits"],
                       "sites": plan["sites"], "weights": plan["weights"]}
            stats = {"acc_abs_max": 0, "fx_saturations": 0}
            res = evaluate_with_stats(conf, payload, heldout, stats, ref_cache)
            v = "PASS" if (res["worst_cosine"] >= 0.995 and res["worst_max_abs"] <= 0.05
                           and res["count_gt_005"] == 0 and entry["safe"]) else "FAIL"
            results[name] = {**res, "verdict": v, "policy_id": entry["policy_id"],
                             "a_bits": entry["a_bits"], "w_bits": entry["w_bits"],
                             "max_safe_k": entry["max_safe_k"],
                             "acc_abs_max_observed": stats["acc_abs_max"],
                             "fx_saturations": stats["fx_saturations"]}
            if v == "PASS":
                passers.append((entry["w_bits"], entry["a_bits"], name))
            print(f"heldout {name:7s}: cos={res['worst_cosine']:.5f} abs={res['worst_max_abs']:.4f} "
                  f">0.05={res['count_gt_005']} acc_max={stats['acc_abs_max']} -> {v}")
        if passers:
            passers.sort()
            verdict = f"FEASIBLE_{passers[0][2]}"
            winner = passers[0][2]
        else:
            verdict = "NOT_FEASIBLE_WITHIN_INT32_AW_FRONTIER"
            winner = None
        doc = {"verdict": verdict, "winner": winner,
               "all_passers_by_ranking": [p[2] for p in sorted(passers)],
               "thresholds": {"worst_cosine": 0.995, "worst_max_abs": 0.05,
                              "count_gt_005_must_be": 0},
               "proceed_to_f4b": "YES" if winner else "NO",
               "results": results,
               "histograms": {k: v["histogram"] for k, v in results.items()}}
        (REPO / "docs" / "phase-f4a-heldout-results.json").write_text(
            json.dumps(doc, indent=1) + "\n", encoding="utf-8")
        (REPO / "docs" / "phase-f4a-search-results.json").write_text(
            json.dumps(doc["results"], indent=1) + "\n", encoding="utf-8")
        (REPO / "docs" / "phase-f4a-error-histograms.json").write_text(
            json.dumps(doc["histograms"], indent=1) + "\n", encoding="utf-8")
        print("VERDICT:", verdict)


if __name__ == "__main__":
    main()
