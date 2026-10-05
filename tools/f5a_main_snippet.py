def main() -> None:
    parser = argparse.ArgumentParser()
    for flag in ("anchor", "safety", "calibrate", "selection", "freeze", "heldout"):
        parser.add_argument(f"--{flag}", action="store_true")
    parser.add_argument("--unlock-heldout", action="store_true")
    args = parser.parse_args()

    conf = Conformer(0)
    cases, samples, head_samples = collect_calibration(conf)
    pairs = pair_family()
    assert len(pairs) == 15, pairs
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
            k64 = max_safe_k64(a, w)
            table[f"A{a}W{w}"] = {
                "a_bits": a, "w_bits": w,
                "a_max_magnitude": 1 << (a - 1), "w_max_magnitude": 1 << (w - 1),
                "worst_product": 1 << (a + w - 2),
                "max_safe_k32": max_safe_k_joint(a, w),
                "max_safe_k64": k64,
                "real_graph_kmax": k_max,
                "int32_safe": bool(max_safe_k_joint(a, w) >= k_max),
                "int64_safe": bool(k64 >= k_max),
                "why_new": ("excluded by int32 admission (MaxSafeK32 < 3072); "
                            "safe under the proposed int64 accumulator")}
        doc = {"formula": "max_safe_k64(a,w) = floor((2^63-1) / 2^(a+w-2))",
               "real_graph": {"gemm_nodes": len(ks), "k_max": k_max,
                              "all_k_values": sorted(set(ks))},
               "candidates": table,
               "note": "admission uses the THEORETICAL bound; observed ranges are recorded separately"}
        (REPO / "docs" / "phase-f5a-accumulator-safety.json").write_text(
            json.dumps(doc, indent=1) + "\n", encoding="utf-8")

        # Freivalds worst-case bounds (r in {0,1}^N; K=N=K_max upper bound)
        N = k_max
        fre = {"r_domain": "{0,1}^N", "K_upper": N, "N_upper": N, "candidates": {}}
        for a, w in pairs:
            x_bits = (N * (1 << (w - 1))).bit_length()
            y_bits = (N * (1 << (a - 1)) * N * (1 << (w - 1))).bit_length()
            z_bits = (N * N * (1 << (a + w - 2))).bit_length()
            fre["candidates"][f"A{a}W{w}"] = {
                "x_bits": x_bits, "y_bits": y_bits, "z_bits": z_bits,
                "required_signed_bits": max(x_bits, y_bits, z_bits) + 1,
                "int64_safe": max(x_bits, y_bits, z_bits) + 1 <= 63,
                "verdict": ("FREIVALDS_INT64_SAFE"
                            if max(x_bits, y_bits, z_bits) + 1 <= 63
                            else "FREIVALDS_REQUIRES_WIDER_INT")}
        (REPO / "docs" / "phase-f5a-freivalds-bounds.json").write_text(
            json.dumps(fre, indent=1) + "\n", encoding="utf-8")
        print(f"safety + freivalds bounds written; K_max={k_max}; "
              f"int64-safe all: {all(v['int64_safe'] for v in table.values())}; "
              f"freivalds int64-safe all: {all(v['int64_safe'] for v in fre['candidates'].values())}")

    if args.anchor:
        import f4a_joint_precision as F4A
        f4a_sel = json.loads((REPO / "testdata" / "qwen3_f4a_selection.json")
                             .read_text(encoding="utf-8"))["cases"][:4]
        for name, (a, w) in (("A12W9", (12, 9)), ("A13W8", (13, 8))):
            plan = build_plan(conf, samples, head_samples, a, w)
            f4a_plan = F4A.build_plan(conf, samples, head_samples, a, w)
            stats = {"acc_abs_max": 0, "fx_saturations": 0}
            for ci, case in enumerate(f4a_sel):
                hidden = conf.embedding(case["token_ids"])
                y_new = run_block_bits(conf, hidden, plan, stats)
                y_old = F4A.run_block_bits(conf, hidden, f4a_plan, {"acc_abs_max": 0, "fx_saturations": 0})
                delta = int(np.max(np.abs(y_new.astype(np.int64) - y_old.astype(np.int64))))
                assert delta == 0, f"anchor {name} case {ci}: int64 executor differs from F.4A (delta {delta})"
            print(f"anchor {name}: int64 executor BIT-IDENTICAL to the F.4A executor on 4 historical cases")
        f3a_heldout = json.loads((REPO / "testdata" / "qwen3_f3a_heldout.json")
                                 .read_text(encoding="utf-8"))["cases"][:8]
        plan_13_10 = build_plan(conf, samples, head_samples, 13, 10)
        res = evaluate_with_stats(conf, plan_13_10, f3a_heldout,
                                  {"acc_abs_max": 0, "fx_saturations": 0}, {})
        print(f"anchor diagnostic (UNSAFE_DIAGNOSTIC_ONLY in F.4A; new candidate here): A13W10 "
              f"cos={res['worst_cosine']:.5f} abs={res['worst_max_abs']:.4f} "
              f"(historical ~0.99993-0.99995 / 0.034-0.049)")
        assert 0.9998 <= res["worst_cosine"] <= 1.0 and res["worst_max_abs"] <= 0.06, res

    if args.calibrate:
        doc = {"model": "Qwen/Qwen3-0.6B-Base", "revision": "57ca99e94acb83175495aa2c6b6b0cc498170924",
               "layer": 0, "seq": 16, "accumulator_width": 64,
               "ranking": ["minimize a_bits + w_bits", "then minimize W_bits", "then minimize A_bits"],
               "candidates": {}}
        for a, w in pairs:
            plan = build_plan(conf, samples, head_samples, a, w)
            blob = json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
            plan["policy_id"] = hashlib.sha256(blob).hexdigest()
            doc["candidates"][f"A{a}W{w}"] = plan
        (REPO / "docs" / "phase-f5a-candidate-policies.json").write_text(
            json.dumps(doc, indent=1) + "\n", encoding="utf-8")
        print("15 candidate policies written")

    if args.selection:
        doc = json.loads((REPO / "docs" / "phase-f5a-candidate-policies.json")
                         .read_text(encoding="utf-8"))
        selection = load_split("selection")
        ref_cache = {}
        results = {}
        requant = {}
        names = sorted(doc["candidates"], key=lambda n: (int(n[1:n.index("W")]) + int(n.split("W")[1]),
                                                         int(n.split("W")[1]), int(n[1:n.index("W")])))
        for name in names:
            plan = doc["candidates"][name]
            stats = {"acc_abs_max": 0, "fx_saturations": 0, "requant_links": []}
            res = evaluate_with_stats(conf, plan, selection, stats, ref_cache)
            v = "PASS" if res["worst_cosine"] >= 0.995 and res["worst_max_abs"] <= 0.05 else "FAIL"
            worst_bits = max([l["product_bits"] for l in stats["requant_links"]] or [0])
            results[name] = {**res, "verdict": v, "policy_id": plan["policy_id"],
                             "max_safe_k64": plan["theoretical_max_safe_k"],
                             "acc_abs_max_observed": stats["acc_abs_max"],
                             "requant_product_bits_max": worst_bits}
            requant[name] = {
                "max_acc_bits": max([l["max_abs_acc"].bit_length() for l in stats["requant_links"]] or [0]),
                "largest_multiplier": max([l["mult"] for l in stats["requant_links"]] or [0]),
                "max_product_bits": worst_bits,
                "int64_safe": worst_bits <= 63,
                "classification": ("REQUANT_INT64_SAFE" if worst_bits <= 63
                                   else "REQUIRES_WIDE_MULTIPLY"),
                "links": len(stats["requant_links"])}
            print(f"selection {name:7s}: cos={res['worst_cosine']:.5f} abs={res['worst_max_abs']:.4f} "
                  f">0.05={res['count_gt_005']:5d} req_bits={worst_bits} -> {v}")
        (REPO / "docs" / "phase-f5a-selection-results.json").write_text(
            json.dumps(results, indent=1) + "\n", encoding="utf-8")
        (REPO / "docs" / "phase-f5a-requant-bounds.json").write_text(
            json.dumps({"note": "observed accumulator magnitudes and requant multiplier products "
                                "over the F5A selection split; classification per section 21",
                        "cumulative": requant}, indent=1) + "\n", encoding="utf-8")

    if args.freeze:
        doc = json.loads((REPO / "docs" / "phase-f5a-candidate-policies.json")
                         .read_text(encoding="utf-8"))
        sel = json.loads((REPO / "docs" / "phase-f5a-selection-results.json")
                         .read_text(encoding="utf-8"))
        requant = json.loads((REPO / "docs" / "phase-f5a-requant-bounds.json")
                             .read_text(encoding="utf-8"))["cumulative"]
        family = {"family": [f"A{a}W{w}" for a, w in pairs],
                  "ranking": doc["ranking"], "accumulator_width": 64, "candidates": {}}
        for name, plan in doc["candidates"].items():
            family["candidates"][name] = {
                "a_bits": plan["bits"], "w_bits": plan["w_bits"],
                "policy_id": plan["policy_id"],
                "max_safe_k64": plan["theoretical_max_safe_k"],
                "safe": bool(plan["theoretical_max_safe_k"] >= 3072),
                "requant": requant[name]["classification"],
                "selection": {k: sel[name][k] for k in
                              ("worst_cosine", "worst_max_abs", "verdict")}}
        blob = json.dumps(family["candidates"], sort_keys=True, separators=(",", ":")).encode()
        family["family_sha256"] = hashlib.sha256(blob).hexdigest()
        (REPO / "docs" / "phase-f5a-frozen-family.json").write_text(
            json.dumps(family, indent=1) + "\n", encoding="utf-8")
        print("frozen family sha256:", family["family_sha256"][:16])

    if args.heldout:
        fam = json.loads((REPO / "docs" / "phase-f5a-frozen-family.json")
                         .read_text(encoding="utf-8"))
        pol = json.loads((REPO / "docs" / "phase-f5a-candidate-policies.json")
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
                             "max_safe_k64": entry["max_safe_k64"],
                             "requant": entry["requant"],
                             "acc_abs_max_observed": stats["acc_abs_max"],
                             "fx_saturations": stats["fx_saturations"]}
            if v == "PASS":
                passers.append((entry["a_bits"] + entry["w_bits"], entry["w_bits"],
                                entry["a_bits"], name))
            print(f"heldout {name:7s}: cos={res['worst_cosine']:.5f} abs={res['worst_max_abs']:.4f} "
                  f">0.05={res['count_gt_005']} acc_max={stats['acc_abs_max']} -> {v}")
        if passers:
            passers.sort()
            winner = passers[0][3]
            verdict = f"FEASIBLE_INT64_{winner}"
        else:
            winner = None
            verdict = "NOT_FEASIBLE_WITHIN_A13_W13_INT64_FRONTIER"
        doc = {"verdict": verdict, "winner": winner,
               "all_passers_by_ranking": [p[3] for p in sorted(passers)],
               "thresholds": {"worst_cosine": 0.995, "worst_max_abs": 0.05,
                              "count_gt_005_must_be": 0},
               "proceed_to_f5b": "YES" if winner else "NO",
               "results": results,
               "histograms": {k: v["histogram"] for k, v in results.items()}}
        (REPO / "docs" / "phase-f5a-heldout-results.json").write_text(
            json.dumps(doc, indent=1) + "\n", encoding="utf-8")
        (REPO / "docs" / "phase-f5a-error-histograms.json").write_text(
            json.dumps(doc["histograms"], indent=1) + "\n", encoding="utf-8")
        print("VERDICT:", verdict)


if __name__ == "__main__":
    main()
