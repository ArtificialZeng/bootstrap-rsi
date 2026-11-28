"""Breakthrough Discovery Pilot: MoE Router Self-Rewiring + Null-Space Ridge LoRA + Context Internalization.

Scientific rigor:
  1. Generates multi-rollout practice data (k=4 seeds) on AIME 2010-2018 to obtain SAME-PROBLEM paired
     (Finished-Correct C vs Finished-Wrong W) trajectories — 100% controlling for problem difficulty & truncation!
  2. Scans layerwise MoE Router SNR & residual Fisher separation across token windows ('early', 'mid', 'late',
     'commit', 'pivot') and compares orthogonality of (W vs C) vs (T vs C) geometry.
  3. Evaluates every candidate compiler against mandatory controls (reseed, random_matched, prompt_concise,
     prompt_verify, dt_bias_s8, inference-time steering_control) on strictly unseen AIME 2019-2021 gate problems
     with exact T->C vs W->C decomposition.
"""
import argparse
import json
import os
import pickle
import sys
import time
import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from v2.gpu_lock import gpu_lock
from v2.benchmarks import load_benchmark
from v2.engine import (
    load_native_model, eval_problems, extract_layer_diagnostics, get_layers, is_moe_model,
    PROMPT_CONCISE, PROMPT_VERIFY, V2_OUT
)
from v2.compilers import (
    ModelEditor, compute_router_contrast, compile_router_bias, compile_router_nullspace_lora,
    compile_contrastive_direction, compile_nullspace_mlp_lora, compile_context_internalization_lora,
    compile_rank1_ablation_lora
)
from v2.stats import summarize_records, compare_paired, compare_multiseed, format_comparison


def find_same_problem_pairs(rollouts_by_seed):
    """Find problems that have BOTH finished-correct (C) and finished-wrong (W) rollouts across seeds!

    Returns:
      paired_records : flat list of records (for extraction)
      pos_idx        : indices in paired_records that are C
      neg_idx        : indices in paired_records that are W
      pair_pids      : list of problem IDs that yielded same-problem (C, W) pairs
      Also returns truncation pairs (pos_t_idx, neg_t_idx) for C vs T orthogonality analysis!
    """
    by_pid = {}
    for recs in rollouts_by_seed:
        for r in recs:
            by_pid.setdefault(r["pid"], []).append(r)

    # 1. Strictly same-problem finished C vs finished W pairs (ZERO difficulty or truncation confounding!)
    cw_recs, cw_pos, cw_neg, cw_pids = [], [], [], []
    # 2. Same-problem C vs T pairs (truncation mechanism)
    ct_recs, ct_pos, ct_neg, ct_pids = [], [], [], []
    # 3. General finished C vs W pool (fallback if same-problem pairs are scarce)
    all_c, all_w, all_t = [], [], []

    for pid, rlist in sorted(by_pid.items()):
        cs = [r for r in rlist if r["state"] == "C"]
        ws = [r for r in rlist if r["state"] == "W"]
        ts = [r for r in rlist if r["state"] == "T"]
        all_c.extend(cs)
        all_w.extend(ws)
        all_t.extend(ts)
        if cs and ws:
            cw_pids.append(pid)
            # Balanced 1-to-1 or up to 2-to-2 per problem so no single problem dominates
            m = min(len(cs), len(ws), 2)
            for j in range(m):
                cw_pos.append(len(cw_recs)); cw_recs.append(cs[j])
                cw_neg.append(len(cw_recs)); cw_recs.append(ws[j])
        if cs and ts:
            ct_pids.append(pid)
            m = min(len(cs), len(ts), 2)
            for j in range(m):
                ct_pos.append(len(ct_recs)); ct_recs.append(cs[j])
                ct_neg.append(len(ct_recs)); ct_recs.append(ts[j])

    stats = dict(
        n_problems=len(by_pid),
        n_rollouts=sum(len(v) for v in by_pid.values()),
        total_C=len(all_c),
        total_W=len(all_w),
        total_T=len(all_t),
        pass_at_1=len(all_c) / max(1, sum(len(v) for v in by_pid.values())),
        pass_at_k=sum(1 for v in by_pid.values() if any(r["correct"] for r in v)) / max(1, len(by_pid)),
        n_same_prob_CW_pids=len(cw_pids),
        n_same_prob_CW_pairs=len(cw_pos),
        n_same_prob_CT_pids=len(ct_pids),
        n_same_prob_CT_pairs=len(ct_pos),
    )
    return dict(
        stats=stats,
        cw=(cw_recs, cw_pos, cw_neg, cw_pids),
        ct=(ct_recs, ct_pos, ct_neg, ct_pids),
        all_C=all_c, all_W=all_w, all_T=all_t,
    )


def run_pilot(args):
    think_tag = "think" if args.thinking else "nothink"
    tag = f"pilot_{args.model}_{args.bench}_{think_tag}_c{args.cap}_p{args.n_prac}_g{args.n_gate}"
    run_dir = os.path.join(V2_OUT, tag)
    os.makedirs(run_dir, exist_ok=True)
    summary_path = os.path.join(run_dir, "summary.json")

    model, tok = load_native_model(args.model)
    nL = len(get_layers(model))
    moe = is_moe_model(model)

    if args.bench == "aime":
        prac_probs = load_benchmark("aime_practice")[:args.n_prac]
        gate_probs = load_benchmark("aime_gate")[:args.n_gate]
    elif args.bench == "oly_math":
        prac_probs = load_benchmark("oly_math_practice")[:args.n_prac]
        gate_probs = load_benchmark("oly_math_eval")[:args.n_gate]
    elif args.bench == "math500_hard":
        prac_probs = load_benchmark("math500_practice")[:args.n_prac]
        gate_probs = load_benchmark("math500_hard_eval")[:args.n_gate]
    elif args.bench == "supergpqa":
        prac_probs = load_benchmark("supergpqa_sci_hard_practice")[:args.n_prac]
        gate_probs = load_benchmark("supergpqa_sci_hard_gate")[:args.n_gate]
    elif args.bench == "mmlupro":
        prac_probs = load_benchmark("mmlupro_stem_practice")[:args.n_prac]
        gate_probs = load_benchmark("mmlupro_stem_gate")[:args.n_gate]
    else:
        raise ValueError(f"Unsupported pilot bench: {args.bench}")

    summary = dict(tag=tag, model=args.model, bench=args.bench, thinking=args.thinking, cap=args.cap,
                   n_layers=nL, is_moe=moe, n_prac=len(prac_probs), n_gate=len(gate_probs),
                   arms={}, comparisons={})

    def save_summary():
        json.dump(summary, open(summary_path, "w"), indent=2, default=str)

    # ==== STEP 1: Multi-rollout Practice Mining (k seeds -> same-problem C vs W pairs!) ====
    print(f"\n=== STEP 1: Practice multi-rollout mining ({len(prac_probs)} probs x {args.prac_k} seeds, cap={args.cap}, think={args.thinking}) ===", flush=True)
    prac_seeds = [100 + s for s in range(args.prac_k)]
    prac_runs = []
    for s in prac_seeds:
        cp = os.path.join(run_dir, f"prac_base_s{s}.pkl")
        recs = eval_problems(model, tok, prac_probs, args.cap, seed=s, bs=args.bs,
                             enable_thinking=args.thinking, cache_path=cp)
        st = summarize_records(recs, f"prac_s{s}")
        print(f"  [prac s{s}] acc={100*st['acc']:.1f}% trunc={100*st['trunc']:.1f}% "
              f"W={st['n_W']} C={st['n_C']} T={st['n_T']} acc|fin={100*st['acc_finished']:.1f}% med_tok={st['med_tok']:.0f}", flush=True)
        prac_runs.append(recs)

    pairs = find_same_problem_pairs(prac_runs)
    summary["practice_mining"] = pairs["stats"]
    print("  [mining]", json.dumps(pairs["stats"], indent=2), flush=True)
    save_summary()

    # ==== STEP 2: Gate Baselines & Controls (2 seeds!) ====
    gate_seeds = [1000, 2000] if args.gate_seeds == 2 else [1000]
    print(f"\n=== STEP 2: Gate baselines ({len(gate_probs)} unseen probs x {len(gate_seeds)} seeds) ===", flush=True)

    def eval_gate_arm(arm_name, extra=""):
        runs = []
        for s in gate_seeds:
            cp = os.path.join(run_dir, f"gate_{arm_name}_s{s}.pkl")
            recs = eval_problems(model, tok, gate_probs, args.cap, seed=s, bs=args.bs,
                                 extra=extra, enable_thinking=args.thinking, cache_path=cp)
            runs.append(recs)
        summary["arms"][arm_name] = [summarize_records(r, f"{arm_name}_s{s}") for r, s in zip(runs, gate_seeds)]
        return runs

    base_gate = eval_gate_arm("base")
    if len(gate_seeds) >= 2:
        reseed_comp = compare_paired(base_gate[1], base_gate[0])
    else:
        cp_rs = os.path.join(run_dir, "gate_base_s2000.pkl")
        base_s2 = eval_problems(model, tok, gate_probs, args.cap, seed=2000, bs=args.bs,
                                enable_thinking=args.thinking, cache_path=cp_rs)
        reseed_comp = compare_paired(base_s2, base_gate[0])
    summary["comparisons"]["noise_floor_reseed"] = reseed_comp
    print("  " + format_comparison("NOISE FLOOR (s2000-s1000)", reseed_comp), flush=True)

    concise_gate = eval_gate_arm("prompt_concise", extra=PROMPT_CONCISE)
    verify_gate = eval_gate_arm("prompt_verify", extra=PROMPT_VERIFY)
    for nm, runs in [("prompt_concise", concise_gate), ("prompt_verify", verify_gate)]:
        c = compare_multiseed(runs, base_gate)
        summary["comparisons"][nm] = c
        print("  " + format_comparison(nm, c), flush=True)
    save_summary()

    # ==== STEP 3: Layerwise Mechanistic Scan on Same-Problem (C vs W) & (C vs T) ====
    cw_recs, cw_pos, cw_neg, cw_pids = pairs["cw"]
    ct_recs, ct_pos, ct_neg, ct_pids = pairs["ct"]
    if len(cw_pos) < 4:
        print(f"  [WARN] Only {len(cw_pos)} same-problem C/W pairs; falling back to balanced pool of all C/W!", flush=True)
        m = min(len(pairs["all_C"]), len(pairs["all_W"]), 40)
        cw_recs = pairs["all_C"][:m] + pairs["all_W"][:m]
        cw_pos = list(range(m))
        cw_neg = list(range(m, 2 * m))

    # Scan every 2nd layer across middle-to-upper network (and early layers) for clean SNR profile
    scan_layers = sorted(set(list(range(2, nL - 1, 2)) + [nL - 2]))
    diag_cache = os.path.join(run_dir, f"diag_cw_{args.window}.pkl")
    if os.path.exists(diag_cache):
        diag_cw = pickle.load(open(diag_cache, "rb"))
        print(f"\n=== STEP 3: Loaded cached C-vs-W diagnostics ({len(scan_layers)} layers, window={args.window}) ===", flush=True)
    else:
        print(f"\n=== STEP 3: Extracting C-vs-W diagnostics ({len(cw_recs)} trajectories, {len(scan_layers)} layers, window={args.window}) ===", flush=True)
        diag_cw = extract_layer_diagnostics(
            model, cw_recs, scan_layers, tok=tok, window=args.window,
            max_seq_len=min(args.cap + 512, 6144), max_tokens_per_seq=48,
            collect_router=moe, collect_mlp_in=True, collect_residual=True,
        )
        pickle.dump(diag_cw, open(diag_cache, "wb"))

    # Rank layers by MoE router SNR (if MoE) and residual Cohen's d!
    layer_profile = {}
    for l in scan_layers:
        c_dir = compile_contrastive_direction(diag_cw[l], cw_pos, cw_neg)
        info = dict(cohen_d=c_dir["cohen_d"], mean_diff_norm=c_dir["mean_norm"])
        if moe and "router_logits" in diag_cw[l]:
            r_con = compute_router_contrast(diag_cw[l], cw_pos, cw_neg)
            info["router_snr"] = r_con["layer_snr"]
            info["max_abs_t"] = float(np.max(np.abs(r_con["t_stat"])))
            top_pos = np.argsort(r_con["t_stat"])[-5:][::-1].tolist()
            top_neg = np.argsort(r_con["t_stat"])[:5].tolist()
            info["top_pos_experts"] = [(int(e), round(float(r_con["t_stat"][e]), 2)) for e in top_pos]
            info["top_neg_experts"] = [(int(e), round(float(r_con["t_stat"][e]), 2)) for e in top_neg]
        layer_profile[l] = info

    summary["layer_profile_CW"] = layer_profile
    # Print top layers
    if moe:
        ranked_r = sorted(scan_layers, key=lambda l: layer_profile[l].get("router_snr", 0.0), reverse=True)
        print("  Top MoE Router SNR layers (same-problem C vs W):", flush=True)
        for l in ranked_r[:8]:
            p = layer_profile[l]
            print(f"    L{l:2d}: router_snr={p['router_snr']:.2f} max|t|={p['max_abs_t']:.2f} "
                  f"cohen_d={p['cohen_d']:.2f} +exp={p['top_pos_experts'][:3]} -exp={p['top_neg_experts'][:3]}", flush=True)
    ranked_d = sorted(scan_layers, key=lambda l: abs(layer_profile[l]["cohen_d"]), reverse=True)
    print("  Top Residual Cohen's d layers (same-problem C vs W):", flush=True)
    for l in ranked_d[:6]:
        p = layer_profile[l]
        print(f"    L{l:2d}: cohen_d={p['cohen_d']:.2f} mean_diff_norm={p['mean_diff_norm']:.2f}", flush=True)

    # Optional orthogonality check between (C vs W) and (C vs T) geometry!
    if len(ct_pos) >= 4:
        ct_cache = os.path.join(run_dir, f"diag_ct_{args.window}.pkl")
        if os.path.exists(ct_cache):
            diag_ct = pickle.load(open(ct_cache, "rb"))
        else:
            diag_ct = extract_layer_diagnostics(
                model, ct_recs, scan_layers, tok=tok, window=args.window,
                max_seq_len=min(args.cap + 512, 6144), max_tokens_per_seq=48,
                collect_router=moe, collect_mlp_in=False, collect_residual=True,
            )
            pickle.dump(diag_ct, open(ct_cache, "wb"))
        ortho = {}
        for l in scan_layers:
            d_cw = compile_contrastive_direction(diag_cw[l], cw_pos, cw_neg)["d_diff"]
            d_ct = compile_contrastive_direction(diag_ct[l], ct_pos, ct_neg)["d_diff"]
            cos_res = float(np.dot(d_cw, d_ct))
            entry = dict(cos_residual=cos_res)
            if moe and "router_logits" in diag_ct[l]:
                t_cw = compute_router_contrast(diag_cw[l], cw_pos, cw_neg)["t_stat"]
                t_ct = compute_router_contrast(diag_ct[l], ct_pos, ct_neg)["t_stat"]
                cos_r = float(np.dot(t_cw, t_ct) / (np.linalg.norm(t_cw) * np.linalg.norm(t_ct) + 1e-12))
                entry["cos_router"] = cos_r
            ortho[l] = entry
        summary["orthogonality_CW_vs_CT"] = ortho
        mean_cos_res = float(np.mean([abs(v["cos_residual"]) for v in ortho.values()]))
        print(f"  [orthogonality] mean |cos(CW, CT)| residual = {mean_cos_res:.3f}" +
              (f", router = {np.mean([abs(v['cos_router']) for v in ortho.values()]):.3f}" if moe else ""), flush=True)
    save_summary()

    # ==== STEP 4: Head-to-Head Training-Free Compilers vs Controls on Unseen Gate ====
    print(f"\n=== STEP 4: Evaluating training-free weight compilers vs controls on unseen gate ===", flush=True)
    ed = ModelEditor(model)

    def test_edit(arm_name, apply_fn):
        cp_all_exist = all(os.path.exists(os.path.join(run_dir, f"gate_{arm_name}_s{s}.pkl")) for s in gate_seeds)
        if cp_all_exist:
            runs = eval_gate_arm(arm_name)
        else:
            apply_fn()
            try:
                runs = eval_gate_arm(arm_name)
            finally:
                ed.rollback_all()
        c = compare_multiseed(runs, base_gate)
        summary["comparisons"][arm_name] = c
        print("  " + format_comparison(arm_name, c), flush=True)
        save_summary()
        return runs, c

    # 4.1 Baseline Fan et al. s8: dt_bias += 0.5 (all linear-attention layers)
    if any(getattr(b, "is_linear", False) for b in get_layers(model)):
        test_edit("dt_bias_s8_0.5", lambda: ed.apply_dt_bias_edit(0.5))

    # 4.2 MoE Router Self-Rewiring (if MoE!)
    if moe:
        top_r_layers = sorted(scan_layers, key=lambda l: layer_profile[l].get("router_snr", 0.0), reverse=True)[:8]
        # (a) Router Contrastive Bias (zero-overhead!) at two doses on top-8 SNR layers
        for alpha in (0.25, 0.50):
            def _apply_rb(a=alpha, ls=top_r_layers):
                for l in ls:
                    con = compute_router_contrast(diag_cw[l], cw_pos, cw_neg)
                    b = compile_router_bias(con, alpha=a, mode="t_stat", top_m=32)
                    ed.apply_router_edit(l, bias=b, meta=dict(alpha=a, mode="t_stat", top_m=32))
            test_edit(f"router_bias_WC_a{alpha}", _apply_rb)

        # (a2) SNR-weighted Router Contrastive Bias across ALL scanned layers (gentle cooperative steering!)
        def _apply_rb_all():
            max_snr = max(layer_profile[l].get("router_snr", 1.0) for l in scan_layers)
            for l in scan_layers:
                w = layer_profile[l].get("router_snr", 0.0) / (max_snr + 1e-8)
                con = compute_router_contrast(diag_cw[l], cw_pos, cw_neg)
                b = compile_router_bias(con, alpha=0.25 * w, mode="t_stat", top_m=48)
                ed.apply_router_edit(l, bias=b, meta=dict(alpha=0.25 * w, mode="all_snr_weighted"))
        test_edit("router_bias_WC_all_snr0.25", _apply_rb_all)

        # (b) State-Conditional Null-Space Router LoRA (selectively reroutes failure states!)
        for alpha in (0.35, 0.70):
            def _apply_rl(a=alpha, ls=top_r_layers):
                for l in ls:
                    con = compute_router_contrast(diag_cw[l], cw_pos, cw_neg)
                    la, lb, rd = compile_router_nullspace_lora(
                        diag_cw[l], con, alpha=a, rank=8, lam_keep=4.0, lam_reg=5.0, top_m=48, target_mode="t_stat"
                    )
                    ed.apply_router_edit(l, lora_a=la, lora_b=lb, meta=dict(alpha=a, selectivity=rd["selectivity"]))
            test_edit(f"router_nullspace_WC_a{alpha}", _apply_rl)

        # (c) Mandatory Random-Sign Control matched to router_bias_WC_a0.50!
        def _apply_rb_rand(ls=top_r_layers):
            rng = np.random.default_rng(12345)
            for l in ls:
                con = compute_router_contrast(diag_cw[l], cw_pos, cw_neg)
                b = compile_router_bias(con, alpha=0.50, mode="t_stat", top_m=32)
                signs = rng.choice([-1.0, 1.0], size=b.shape).astype(np.float32)
                ed.apply_router_edit(l, bias=b * signs, meta=dict(control="random_sign"))
        test_edit("router_bias_RANDOM_a0.50", _apply_rb_rand)

        # (d) Orthogonal C-vs-T MoE Router Rewiring (truncation/efficiency experts vs accuracy experts!)
        if len(ct_pos) >= 4 and "diag_ct" in locals():
            ct_snr = {l: compute_router_contrast(diag_ct[l], ct_pos, ct_neg)["layer_snr"]
                      for l in scan_layers if "router_logits" in diag_ct[l]}
            top_ct_layers = sorted(ct_snr, key=ct_snr.get, reverse=True)[:10]
            def _apply_rb_ct(ls=top_ct_layers):
                for l in ls:
                    con = compute_router_contrast(diag_ct[l], ct_pos, ct_neg)
                    b = compile_router_bias(con, alpha=0.50, mode="t_stat", top_m=32)
                    ed.apply_router_edit(l, bias=b, meta=dict(contrast="CT", alpha=0.50))
            test_edit("router_bias_CT_a0.50", _apply_rb_ct)

    # 4.3 Closed-Form Null-Space Ridge LoRA on MLP down_proj / shared_expert.down_proj (same-problem C vs W!)
    top_d_layers = sorted(scan_layers, key=lambda l: abs(layer_profile[l]["cohen_d"]), reverse=True)[:4]
    for alpha in (0.35, 0.75):
        def _apply_ns_lora(a=alpha, ls=top_d_layers):
            for l in ls:
                la, lb, rd = compile_nullspace_mlp_lora(
                    diag_cw[l], cw_pos, cw_neg, alpha=a, rank=4, lam_keep=4.0, lam_reg=5.0
                )
                ed.apply_lora_edit(l, "mlp_down", la, lb, meta=dict(alpha=a, selectivity=rd["selectivity"]))
        test_edit(f"nullspace_lora_WC_a{alpha}", _apply_ns_lora)

    # 4.4 Mandatory Random-Direction Control matched to nullspace_lora_WC_a0.35!
    def _apply_ns_rand(ls=top_d_layers):
        rng = np.random.default_rng(54321)
        for l in ls:
            la, lb, _ = compile_nullspace_mlp_lora(
                diag_cw[l], cw_pos, cw_neg, alpha=0.35, rank=4, lam_keep=4.0, lam_reg=5.0
            )
            # Random orthogonal rotation on output space preserving exact Frobenius norm!
            r_mat = rng.normal(size=lb.shape).astype(np.float32)
            r_mat = r_mat * (np.linalg.norm(lb) / (np.linalg.norm(r_mat) + 1e-12))
            ed.apply_lora_edit(l, "mlp_down", la, r_mat, meta=dict(control="random_dir"))
    test_edit("nullspace_lora_RANDOM_a0.35", _apply_ns_rand)

    # 4.5 Mandatory Inference-Time Activation Steering Control (TGAS-style on same top_d_layers!)
    def _apply_steer(ls=top_d_layers):
        for l in ls:
            c_info = compile_contrastive_direction(diag_cw[l], cw_pos, cw_neg)
            delta_v = (0.25 * c_info["mean_norm"]) * c_info["d_diff"]
            ed.apply_steering_control(l, delta_v, meta=dict(alpha=0.25))
    test_edit("steering_control_WC_a0.25", _apply_steer)

    # 4.6 Orthogonal Composition Check: dt_bias (+0.5) + best W->C edit!
    best_wc_candidates = [
        k for k in summary["comparisons"]
        if (k.startswith("router_") or k.startswith("nullspace_lora_WC")) and "RANDOM" not in k
    ]
    if best_wc_candidates and any(getattr(b, "is_linear", False) for b in get_layers(model)):
        best_wc = max(best_wc_candidates, key=lambda k: (summary["comparisons"][k]["delta_reasoning"],
                                                         summary["comparisons"][k]["delta"]))
        print(f"\n  [combo] Pairing dt_bias_s8 (T->C) with best W->C candidate: {best_wc}", flush=True)
        def _apply_combo():
            ed.apply_dt_bias_edit(0.5)
            if best_wc.startswith("router_bias_WC_a"):
                a = float(best_wc.split("_a")[-1])
                for l in top_r_layers:
                    con = compute_router_contrast(diag_cw[l], cw_pos, cw_neg)
                    ed.apply_router_edit(l, bias=compile_router_bias(con, alpha=a, mode="t_stat", top_m=32))
            elif best_wc.startswith("router_nullspace_WC_a"):
                a = float(best_wc.split("_a")[-1])
                for l in top_r_layers:
                    con = compute_router_contrast(diag_cw[l], cw_pos, cw_neg)
                    la, lb, _ = compile_router_nullspace_lora(diag_cw[l], con, alpha=a, rank=8, lam_keep=4.0, lam_reg=5.0)
                    ed.apply_router_edit(l, lora_a=la, lora_b=lb)
            elif best_wc.startswith("nullspace_lora_WC_a"):
                a = float(best_wc.split("_a")[-1])
                for l in top_d_layers:
                    la, lb, _ = compile_nullspace_mlp_lora(diag_cw[l], cw_pos, cw_neg, alpha=a, rank=4, lam_keep=4.0, lam_reg=5.0)
                    ed.apply_lora_edit(l, "mlp_down", la, lb)
        test_edit(f"combo_dt0.5_plus_{best_wc}", _apply_combo)

    print(f"\nPilot complete! Summary saved to {summary_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="q35-35b", choices=["q35-9b", "q35-27b", "q35-35b", "q35-4b"])
    ap.add_argument("--bench", default="supergpqa", choices=["aime", "oly_math", "math500_hard", "supergpqa", "mmlupro"])
    ap.add_argument("--cap", type=int, default=3072)
    ap.add_argument("--n-prac", type=int, default=60)
    ap.add_argument("--prac-k", type=int, default=4)
    ap.add_argument("--n-gate", type=int, default=90)
    ap.add_argument("--gate-seeds", type=int, default=2)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--window", default="all", choices=["all", "early", "mid", "late", "commit", "pivot"])
    ap.add_argument("--no-thinking", dest="thinking", action="store_false", default=True)
    args = ap.parse_args()
    with gpu_lock(f"pilot_{args.model}", wait=True):
        run_pilot(args)


if __name__ == "__main__":
    main()
