"""Experiment B-Final: Strictly Deconfounded W->C Compilation + Unpaired-vs-Deconfounded Mirage Ablation.

Uses Brief-CoT (cap=768) on SuperGPQA-Hard & GPQA-Diamond where acc|finished ~ 40-50%,
yielding abundant W (finished-wrong) trajectories and TRUE same-problem (C, W) pairs.
Directly pits:
  - Strictly Same-Problem Deconfounded MoE Router Bias & Null-Space Router LoRA
  - Strictly Same-Problem Deconfounded Null-Space MLP LoRA
  - Deliberate Unpaired (Topic-Confounded) MoE Router Bias Ablation (proving the Topic-Confounding Mirage!)
  - Closed-Form Context-to-Weight Internalization (Router + MLP LoRA)
  - Fan et al. dt_bias +0.5
  - Mandatory Controls: Reseed noise floor, Random-Sign matched, Inference-Time Steering (TGAS)
Evaluated on strictly unseen SuperGPQA-Hard test (n=128 x 2 seeds = 256 evals) + zero-shot GPQA-Diamond transfer (n=128)!
"""
import argparse
import json
import os
import pickle
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from v2.gpu_lock import gpu_lock
from v2.benchmarks import load_benchmark
from v2.engine import (
    load_native_model, eval_problems, extract_layer_diagnostics, get_layers, is_moe_model, V2_OUT
)
from v2.compilers import (
    ModelEditor, compute_router_contrast, compile_router_bias, compile_router_nullspace_lora,
    compile_contrastive_direction, compile_nullspace_mlp_lora, compile_context_internalization_lora
)
from v2.pilot_internalize import compile_router_context_internalization
from v2.pilot_breakthrough import find_same_problem_pairs
from v2.stats import summarize_records, compare_paired, compare_multiseed, format_comparison

BRIEF_EXTRA = "\nBe concise: give only a brief 3-5 sentence justification (under 200 words) and immediately put your final single capital letter in \\boxed{}."
VERIFY_BRIEF = "\nBriefly identify the core physical/mathematical principle, verify for common traps or sign/factor errors in 4-6 concise sentences, and immediately output the single capital letter in \\boxed{}."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="q35-35b", choices=["q35-35b", "q35-9b", "q35-27b"])
    ap.add_argument("--cap", type=int, default=768)
    ap.add_argument("--n-prac", type=int, default=128)
    ap.add_argument("--prac-k", type=int, default=4)
    ap.add_argument("--n-test", type=int, default=128)
    ap.add_argument("--n-gpqa", type=int, default=128)
    ap.add_argument("--test-seeds", type=int, default=2)
    ap.add_argument("--bs", type=int, default=64)
    args = ap.parse_args()

    tag = f"expB_final_{args.model}_brief_c{args.cap}_p{args.n_prac}_t{args.n_test}"
    run_dir = os.path.join(V2_OUT, tag)
    os.makedirs(run_dir, exist_ok=True)
    out_json = os.path.join(run_dir, "summary.json")

    prac_probs = load_benchmark("supergpqa_sci_hard_practice")[:args.n_prac]
    test_probs = load_benchmark("supergpqa_sci_hard_test")[:args.n_test]
    gpqa_probs = load_benchmark("gpqa_diamond")[:args.n_gpqa]

    with gpu_lock(tag, wait=True):
        model, tok = load_native_model(args.model)
        nL = len(get_layers(model))
        moe = is_moe_model(model)
        ed = ModelEditor(model)

        summary = dict(tag=tag, model=args.model, n_layers=nL, is_moe=moe, cap=args.cap,
                       n_prac=len(prac_probs), n_test=len(test_probs), n_gpqa=len(gpqa_probs),
                       arms={}, comparisons={}, gpqa_transfer={})

        def save():
            json.dump(summary, open(out_json, "w"), indent=2, default=str)

        # ==== STEP 1: Multi-Rollout Same-Problem (C, W) Mining ====
        print(f"\n=== STEP 1: Mining {len(prac_probs)} SuperGPQA-Hard probs x {args.prac_k} seeds (cap={args.cap}, temp=0.85) ===", flush=True)
        prac_runs = []
        for s in range(args.prac_k):
            cp = os.path.join(run_dir, f"prac_s{100+s}.pkl")
            recs = eval_problems(model, tok, prac_probs, args.cap, seed=100 + s, bs=args.bs,
                                 temp=0.85, extra=BRIEF_EXTRA, enable_thinking=False, cache_path=cp)
            st = summarize_records(recs, f"prac_s{100+s}")
            print(f"  [prac s{100+s}] acc={100*st['acc']:.1f}% trunc={100*st['trunc']:.1f}% "
                  f"C={st['n_C']} W={st['n_W']} T={st['n_T']} acc|fin={100*st['acc_finished']:.1f}%", flush=True)
            prac_runs.append(recs)

        pairs = find_same_problem_pairs(prac_runs)
        summary["practice_mining"] = pairs["stats"]
        print("  [mining]", json.dumps(pairs["stats"], indent=2), flush=True)
        save()

        cw_recs, cw_pos, cw_neg, cw_pids = pairs["cw"]
        deconfounded = len(cw_pos) >= 6
        if not deconfounded:
            m = min(len(pairs["all_C"]), len(pairs["all_W"]), 48)
            cw_recs = pairs["all_C"][:m] + pairs["all_W"][:m]
            cw_pos, cw_neg = list(range(m)), list(range(m, 2 * m))
        summary["strictly_same_problem_pairs"] = deconfounded
        summary["n_cw_pairs_used"] = len(cw_pos)

        # Build strictly DISJOINT-PROBLEM (topic-confounded!) contrast of EXACT SAME SIZE for Mirage proof!
        cw_pid_set = set(cw_pids)
        disj_c = [r for r in pairs["all_C"] if r["pid"] not in cw_pid_set]
        disj_w = [r for r in pairs["all_W"] if r["pid"] not in cw_pid_set]
        m_u = min(len(disj_c), len(disj_w), len(cw_pos))
        unp_recs = disj_c[:m_u] + disj_w[:m_u]
        unp_pos, unp_neg = list(range(m_u)), list(range(m_u, 2 * m_u))

        # ==== STEP 2: 40-Layer Mechanistic Mirage Audit (Same-Problem vs Unpaired) ====
        scan_layers = sorted(set(list(range(2, nL - 1, 2)) + [nL - 2]))
        diag_cw = extract_layer_diagnostics(
            model, cw_recs, scan_layers, tok=tok, window="all",
            max_seq_len=args.cap + 512, max_tokens_per_seq=48,
            collect_router=moe, collect_mlp_in=True, collect_residual=True,
        )
        diag_unp = extract_layer_diagnostics(
            model, unp_recs, scan_layers, tok=tok, window="all",
            max_seq_len=args.cap + 512, max_tokens_per_seq=48,
            collect_router=moe, collect_mlp_in=False, collect_residual=True,
        ) if m_u >= 4 else None

        prof_same, prof_unp = {}, {}
        for l in scan_layers:
            cd = compile_contrastive_direction(diag_cw[l], cw_pos, cw_neg)
            es = dict(cohen_d=cd["cohen_d"], mean_diff_norm=cd["mean_norm"])
            if moe and "router_logits" in diag_cw[l]:
                rc = compute_router_contrast(diag_cw[l], cw_pos, cw_neg)
                es["router_snr"] = rc["layer_snr"]
                es["max_abs_t"] = float(np.max(np.abs(rc["t_stat"])))
            prof_same[l] = es
            if diag_unp is not None:
                cdu = compile_contrastive_direction(diag_unp[l], unp_pos, unp_neg)
                eu = dict(cohen_d=cdu["cohen_d"], mean_diff_norm=cdu["mean_norm"])
                if moe and "router_logits" in diag_unp[l]:
                    rcu = compute_router_contrast(diag_unp[l], unp_pos, unp_neg)
                    eu["router_snr"] = rcu["layer_snr"]
                    eu["max_abs_t"] = float(np.max(np.abs(rcu["t_stat"])))
                prof_unp[l] = eu

        summary["profile_same_problem"] = prof_same
        summary["profile_unpaired_mirage"] = prof_unp
        if prof_unp:
            md_s = float(np.mean([abs(v["cohen_d"]) for v in prof_same.values()]))
            md_u = float(np.mean([abs(v["cohen_d"]) for v in prof_unp.values()]))
            msg = f"  [MIRAGE PROOF, N={m_u}] mean |Cohen d|: Unpaired={md_u:.3f} vs Same-Problem={md_s:.3f} (ratio={md_u/(md_s+1e-8):.2f}x)"
            if moe:
                mr_s = float(np.mean([v["router_snr"] for v in prof_same.values()]))
                mr_u = float(np.mean([v["router_snr"] for v in prof_unp.values()]))
                msg += f" | Router SNR: Unpaired={mr_u:.2f} vs Same-Problem={mr_s:.2f} (ratio={mr_u/(mr_s+1e-8):.2f}x)"
            print(msg, flush=True)
        save()

        top_d = sorted(scan_layers, key=lambda l: abs(prof_same[l]["cohen_d"]), reverse=True)[:4]
        top_r = sorted(scan_layers, key=lambda l: prof_same[l].get("router_snr", 0.0), reverse=True)[:8] if moe else []
        top_r_unp = sorted(scan_layers, key=lambda l: prof_unp[l].get("router_snr", 0.0), reverse=True)[:8] if (moe and prof_unp) else []

        # Also extract paired bare vs VERIFY_BRIEF for Context-to-Weight Internalization!
        prac_v = eval_problems(model, tok, prac_probs[:64], args.cap, seed=100, bs=args.bs,
                               temp=0.85, extra=VERIFY_BRIEF, enable_thinking=False,
                               cache_path=os.path.join(run_dir, "prac_verify_s100.pkl"))
        intern_layers = sorted(set([int(round(f * (nL - 1))) for f in (0.35, 0.50, 0.65, 0.80)]))
        d_b_int = extract_layer_diagnostics(model, prac_runs[0][:64], intern_layers, tok=tok, window="all",
                                            max_seq_len=args.cap + 512, max_tokens_per_seq=40,
                                            collect_router=moe, collect_mlp_in=True, collect_residual=True)
        d_v_int = extract_layer_diagnostics(model, prac_v, intern_layers, tok=tok, window="all",
                                            max_seq_len=args.cap + 512, max_tokens_per_seq=40,
                                            collect_router=moe, collect_mlp_in=False, collect_residual=True)

        # ==== STEP 3: Strictly Unseen Test Set (128 probs x 2 seeds = 256 evals!) ====
        test_seeds = [1000, 2000][:args.test_seeds]

        def eval_set(probs, seeds, prefix, extra=BRIEF_EXTRA, apply_fn=None):
            runs = []
            all_hit = all(os.path.exists(os.path.join(run_dir, f"{prefix}_s{s}.pkl")) for s in seeds)
            if not all_hit and apply_fn:
                apply_fn()
            try:
                for s in seeds:
                    cp = os.path.join(run_dir, f"{prefix}_s{s}.pkl")
                    runs.append(eval_problems(model, tok, probs, args.cap, seed=s, bs=args.bs,
                                              extra=extra, enable_thinking=False, cache_path=cp))
            finally:
                if not all_hit and apply_fn:
                    ed.rollback_all()
            return runs

        print(f"\n=== STEP 3: Held-Out Test ({len(test_probs)} probs x {len(test_seeds)} seeds = {len(test_probs)*len(test_seeds)} evals) ===", flush=True)
        base_test = eval_set(test_probs, test_seeds, "test_base")
        summary["arms"]["base"] = [summarize_records(r, f"base_s{s}") for r, s in zip(base_test, test_seeds)]
        if len(test_seeds) >= 2:
            nf = compare_paired(base_test[1], base_test[0])
            summary["comparisons"]["noise_floor_reseed"] = nf
            print("  " + format_comparison("NOISE FLOOR (s2000-s1000)", nf), flush=True)

        arm_apply_map = {}

        def score_test(name, extra=BRIEF_EXTRA, apply_fn=None):
            if apply_fn:
                arm_apply_map[name] = apply_fn
            runs = eval_set(test_probs, test_seeds, f"test_{name}", extra=extra, apply_fn=apply_fn)
            summary["arms"][name] = [summarize_records(r, f"{name}_s{s}") for r, s in zip(runs, test_seeds)]
            comp = compare_multiseed(runs, base_test)
            summary["comparisons"][name] = comp
            print("  " + format_comparison(name, comp), flush=True)
            save()

        score_test("prompt_verify_brief", extra=VERIFY_BRIEF)
        if any(getattr(b, "is_linear", False) for b in get_layers(model)):
            score_test("dt_bias_+0.5", apply_fn=lambda: ed.apply_dt_bias_edit(0.5))

        if moe:
            def _rb_same():
                for l in top_r:
                    con = compute_router_contrast(diag_cw[l], cw_pos, cw_neg)
                    ed.apply_router_edit(l, bias=compile_router_bias(con, alpha=0.40, mode="t_stat", top_m=32))
            score_test("sameprob_router_bias_a0.40", apply_fn=_rb_same)

            def _rl_same():
                for l in top_r:
                    con = compute_router_contrast(diag_cw[l], cw_pos, cw_neg)
                    la, lb, _ = compile_router_nullspace_lora(diag_cw[l], con, alpha=0.50, rank=8, lam_keep=4.0, lam_reg=5.0)
                    ed.apply_router_edit(l, lora_a=la, lora_b=lb)
            score_test("sameprob_router_nullspace_a0.50", apply_fn=_rl_same)

            if diag_unp is not None:
                def _rb_unp():
                    for l in top_r_unp:
                        con = compute_router_contrast(diag_unp[l], unp_pos, unp_neg)
                        ed.apply_router_edit(l, bias=compile_router_bias(con, alpha=0.40, mode="t_stat", top_m=32))
                score_test("UNPAIRED_mirage_router_bias_a0.40", apply_fn=_rb_unp)

            def _r_intern():
                for l in intern_layers:
                    la, lb, _ = compile_router_context_internalization(d_b_int[l], d_v_int[l], alpha=0.6, rank=8, lam_keep=2.0, lam_reg=5.0)
                    ed.apply_router_edit(l, lora_a=la, lora_b=lb)
            score_test("ctx_internalize_router_a0.6", apply_fn=_r_intern)

            def _rb_rand():
                rng = np.random.default_rng(31415)
                for l in top_r:
                    con = compute_router_contrast(diag_cw[l], cw_pos, cw_neg)
                    b = compile_router_bias(con, alpha=0.40, mode="t_stat", top_m=32)
                    ed.apply_router_edit(l, bias=b * rng.choice([-1.0, 1.0], size=b.shape).astype(np.float32))
            score_test("random_router_control_a0.40", apply_fn=_rb_rand)

        def _ns_mlp():
            for l in top_d:
                la, lb, _ = compile_nullspace_mlp_lora(diag_cw[l], cw_pos, cw_neg, alpha=0.50, rank=4, lam_keep=4.0, lam_reg=5.0)
                ed.apply_lora_edit(l, "mlp_down", la, lb)
        score_test("sameprob_nullspace_mlp_a0.50", apply_fn=_ns_mlp)

        def _ci_mlp():
            for l in intern_layers:
                la, lb, _ = compile_context_internalization_lora(d_b_int[l], d_v_int[l], alpha=0.6, rank=8, lam_keep=2.0, lam_reg=5.0)
                ed.apply_lora_edit(l, "mlp_down", la, lb)
        score_test("ctx_internalize_mlp_a0.6", apply_fn=_ci_mlp)

        def _steer():
            for l in top_d:
                ci = compile_contrastive_direction(diag_cw[l], cw_pos, cw_neg)
                ed.apply_steering_control(l, (0.30 * ci["mean_norm"]) * ci["d_diff"])
        score_test("steering_control_TGAS", apply_fn=_steer)

        # ==== STEP 4: Zero-Shot Cross-Benchmark Transfer to GPQA-Diamond (n=128)! ====
        print(f"\n=== STEP 4: Zero-Shot Transfer to GPQA-Diamond (n={len(gpqa_probs)}) ===", flush=True)
        gpqa_base = eval_set(gpqa_probs, [1000], "gpqa_base")
        summary["gpqa_transfer"]["base"] = summarize_records(gpqa_base[0], "gpqa_base")
        # Transfer dt_bias + top 2 weight compilers!
        cand_names = [k for k in arm_apply_map if "random" not in k and "UNPAIRED" not in k and "steering" not in k]
        top_transfer = sorted(cand_names, key=lambda k: summary["comparisons"][k]["delta"], reverse=True)[:3]
        for nm in top_transfer:
            runs = eval_set(gpqa_probs, [1000], f"gpqa_{nm}", apply_fn=arm_apply_map[nm])
            comp = compare_paired(runs[0], gpqa_base[0])
            summary["gpqa_transfer"][nm] = comp
            print("  GPQA TRANSFER " + format_comparison(nm, comp), flush=True)
            save()

    print(f"\nExperiment B-Final complete -> {out_json}", flush=True)


if __name__ == "__main__":
    main()
