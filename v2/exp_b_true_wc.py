"""Experiment B: Strictly Deconfounded Pure W->C Testbed (Zero-Truncation + Same-Problem (C, W) Pairs).

Directly eliminates BOTH confounders of prior training-free weight editing:
  1. Truncation confounder (T->C): uses MMLU-Pro STEM / GPQA-Diamond in direct CoT where trunc < 10%.
  2. Topic/Difficulty confounder (discovered in Pilot 1): mines k=6 rollouts at temp=0.85 to build
     STRICTLY SAME-PROBLEM (C, W) pairs, and compares unpaired vs deconfounded same-problem SNR!
Tests:
  - Same-problem Null-Space Ridge LoRA (mlp_down & attn_out)
  - Same-problem MoE Router Self-Rewiring (if MoE)
  - Closed-Form Context-to-Weight Internalization (compiling a careful verification prompt -> bare weights!)
  - Mandatory controls: reseed noise floor, random-direction norm-matched, inference-time steering, prompt!
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

PROMPT_CAREFUL = (
    "\nSolve this methodically: write down the governing equations, check units and boundary conditions, "
    "verify your final arithmetic independently, and then output the single best option letter in \\boxed{}."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="q35-9b", choices=["q35-9b", "q35-35b", "q35-27b"])
    ap.add_argument("--bench", default="supergpqa_concept", choices=["mmlupro", "gpqa", "supergpqa_concept"])
    ap.add_argument("--cap", type=int, default=2048)
    ap.add_argument("--n-prac", type=int, default=64)
    ap.add_argument("--prac-k", type=int, default=6)
    ap.add_argument("--prac-temp", type=float, default=0.85)
    ap.add_argument("--n-eval", type=int, default=128)
    ap.add_argument("--eval-seeds", type=int, default=2)
    ap.add_argument("--bs", type=int, default=32)
    args = ap.parse_args()

    tag = f"expB_{args.model}_{args.bench}_pureWC_c{args.cap}_p{args.n_prac}_e{args.n_eval}"
    run_dir = os.path.join(V2_OUT, tag)
    os.makedirs(run_dir, exist_ok=True)
    out_json = os.path.join(run_dir, "summary.json")

    if args.bench == "mmlupro":
        prac_probs = load_benchmark("mmlupro_stem_practice")[:args.n_prac]
        eval_probs = load_benchmark("mmlupro_stem_gate")[:args.n_eval]
    elif args.bench == "supergpqa_concept":
        prac_probs = load_benchmark("supergpqa_concept_practice")[:args.n_prac]
        eval_probs = load_benchmark("supergpqa_concept_gate")[:args.n_eval]
    else:
        all_gpqa = load_benchmark("gpqa_diamond")
        prac_probs = all_gpqa[:args.n_prac]
        eval_probs = all_gpqa[args.n_prac:args.n_prac + args.n_eval]

    with gpu_lock(tag, wait=True):
        model, tok = load_native_model(args.model)
        nL = len(get_layers(model))
        moe = is_moe_model(model)
        ed = ModelEditor(model)

        summary = dict(tag=tag, model=args.model, bench=args.bench, n_layers=nL, is_moe=moe,
                       n_prac=len(prac_probs), n_eval=len(eval_probs), arms={}, comparisons={})

        def save():
            json.dump(summary, open(out_json, "w"), indent=2, default=str)

        # Step 1: Mine k rollouts at temp=0.85 for TRUE same-problem (C, W) pairs!
        print(f"\n=== STEP 1: Same-Problem (C, W) Mining ({len(prac_probs)} probs x {args.prac_k} seeds, temp={args.prac_temp}) ===", flush=True)
        prac_runs = []
        for s in range(args.prac_k):
            cp = os.path.join(run_dir, f"prac_s{100+s}.pkl")
            recs = eval_problems(model, tok, prac_probs, args.cap, seed=100 + s, bs=args.bs,
                                 temp=args.prac_temp, enable_thinking=False, cache_path=cp)
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
            print(f"  [WARN] Only {len(cw_pos)} same-problem C/W pairs; supplementing with all C/W.", flush=True)
            m = min(len(pairs["all_C"]), len(pairs["all_W"]), 40)
            cw_recs = pairs["all_C"][:m] + pairs["all_W"][:m]
            cw_pos, cw_neg = list(range(m)), list(range(m, 2 * m))
        summary["strictly_same_problem_pairs"] = deconfounded

        # Step 2: Compare Deconfounded (same-problem) vs Unpaired SNR!
        scan_layers = sorted(set(list(range(2, nL - 1, 2)) + [nL - 2]))
        diag_cw = extract_layer_diagnostics(
            model, cw_recs, scan_layers, tok=tok, window="commit",
            max_seq_len=args.cap + 512, max_tokens_per_seq=48,
            collect_router=moe, collect_mlp_in=True, collect_residual=True,
        )
        profile = {}
        for l in scan_layers:
            cd = compile_contrastive_direction(diag_cw[l], cw_pos, cw_neg)
            ent = dict(cohen_d=cd["cohen_d"], mean_diff_norm=cd["mean_norm"])
            if moe and "router_logits" in diag_cw[l]:
                rc = compute_router_contrast(diag_cw[l], cw_pos, cw_neg)
                ent["router_snr"] = rc["layer_snr"]
                ent["max_abs_t"] = float(np.max(np.abs(rc["t_stat"])))
            profile[l] = ent
        summary["layer_profile_deconfounded"] = profile

        top_d_layers = sorted(scan_layers, key=lambda l: abs(profile[l]["cohen_d"]), reverse=True)[:6]
        print("  Top Deconfounded Residual Cohen's d layers:",
              [(l, round(profile[l]["cohen_d"], 3)) for l in top_d_layers], flush=True)
        if moe:
            top_r_layers = sorted(scan_layers, key=lambda l: profile[l].get("router_snr", 0.0), reverse=True)[:8]
            print("  Top Deconfounded MoE Router SNR layers:",
                  [(l, round(profile[l]["router_snr"], 3)) for l in top_r_layers], flush=True)

        # Also measure UNPAIRED C-vs-W SNR (cross-problem) on the same layers to quantify Topic-Confounding Mirage!
        if deconfounded and len(pairs["all_C"]) >= 6 and len(pairs["all_W"]) >= 6:
            # Build strictly disjoint-problem C vs W sample of equal size to cw_pos
            cw_pid_set = set(cw_pids)
            disj_c = [r for r in pairs["all_C"] if r["pid"] not in cw_pid_set]
            disj_w = [r for r in pairs["all_W"] if r["pid"] not in cw_pid_set]
            m_u = min(len(disj_c), len(disj_w), len(cw_pos))
            if m_u >= 4:
                unp_recs = disj_c[:m_u] + disj_w[:m_u]
                unp_pos, unp_neg = list(range(m_u)), list(range(m_u, 2 * m_u))
                diag_unp = extract_layer_diagnostics(
                    model, unp_recs, scan_layers, tok=tok, window="commit",
                    max_seq_len=args.cap + 512, max_tokens_per_seq=48,
                    collect_router=moe, collect_mlp_in=False, collect_residual=True,
                )
                unp_prof = {}
                for l in scan_layers:
                    cd_u = compile_contrastive_direction(diag_unp[l], unp_pos, unp_neg)
                    e_u = dict(cohen_d=cd_u["cohen_d"])
                    if moe and "router_logits" in diag_unp[l]:
                        e_u["router_snr"] = compute_router_contrast(diag_unp[l], unp_pos, unp_neg)["layer_snr"]
                    unp_prof[l] = e_u
                summary["layer_profile_unpaired_mirage"] = unp_prof
                mean_d_same = float(np.mean([abs(profile[l]["cohen_d"]) for l in scan_layers]))
                mean_d_unp = float(np.mean([abs(unp_prof[l]["cohen_d"]) for l in scan_layers]))
                msg = f"  [MIRAGE CHECK, matched N={m_u}] mean |Cohen d|: unpaired={mean_d_unp:.3f} vs same-problem={mean_d_same:.3f}"
                if moe:
                    mr_s = float(np.mean([profile[l].get("router_snr", 0) for l in scan_layers]))
                    mr_u = float(np.mean([unp_prof[l].get("router_snr", 0) for l in scan_layers]))
                    msg += f" | router_snr: unpaired={mr_u:.2f} vs same-problem={mr_s:.2f}"
                print(msg, flush=True)

        # Also collect paired bare vs PROMPT_CAREFUL practice trajectories for Context-to-Weight Internalization!
        prac_bare_i = prac_runs[0]
        prac_care_i = eval_problems(model, tok, prac_probs, args.cap, seed=100, bs=args.bs,
                                    extra=PROMPT_CAREFUL, temp=args.prac_temp, enable_thinking=False,
                                    cache_path=os.path.join(run_dir, "prac_careful_s100.pkl"))
        intern_layers = sorted(set([int(round(f * (nL - 1))) for f in (0.4, 0.55, 0.7, 0.85)]))
        d_bare_i = extract_layer_diagnostics(
            model, prac_bare_i, intern_layers, tok=tok, window="commit",
            max_seq_len=args.cap + 512, max_tokens_per_seq=40,
            collect_router=moe, collect_mlp_in=True, collect_residual=True,
        )
        d_care_i = extract_layer_diagnostics(
            model, prac_care_i, intern_layers, tok=tok, window="commit",
            max_seq_len=args.cap + 512, max_tokens_per_seq=40,
            collect_router=moe, collect_mlp_in=False, collect_residual=True,
        )

        # Step 3: Evaluate on strictly unseen eval_probs x eval_seeds!
        eval_seeds = [1000, 2000][:args.eval_seeds]
        def eval_arm(name, extra="", apply_fn=None):
            runs = []
            all_cached = all(os.path.exists(os.path.join(run_dir, f"eval_{name}_s{s}.pkl")) for s in eval_seeds)
            if not all_cached and apply_fn:
                apply_fn()
            try:
                for s in eval_seeds:
                    cp = os.path.join(run_dir, f"eval_{name}_s{s}.pkl")
                    runs.append(eval_problems(model, tok, eval_probs, args.cap, seed=s, bs=args.bs,
                                              extra=extra, enable_thinking=False, cache_path=cp))
            finally:
                if not all_cached and apply_fn:
                    ed.rollback_all()
            summary["arms"][name] = [summarize_records(r, f"{name}_s{s}") for r, s in zip(runs, eval_seeds)]
            return runs

        base_runs = eval_arm("base")
        if len(eval_seeds) >= 2:
            nf = compare_paired(base_runs[1], base_runs[0])
            summary["comparisons"]["noise_floor_reseed"] = nf
            print("  " + format_comparison("NOISE FLOOR (s2000-s1000)", nf), flush=True)

        def score_arm(name, extra="", apply_fn=None):
            runs = eval_arm(name, extra=extra, apply_fn=apply_fn)
            comp = compare_multiseed(runs, base_runs)
            summary["comparisons"][name] = comp
            print("  " + format_comparison(name, comp), flush=True)
            save()

        score_arm("prompt_careful", extra=PROMPT_CAREFUL)

        # Null-Space Ridge LoRA (strictly same-problem C vs W!)
        for a in (0.35, 0.70):
            def _ns(alpha=a):
                for l in top_d_layers[:4]:
                    la, lb, rd = compile_nullspace_mlp_lora(diag_cw[l], cw_pos, cw_neg, alpha=alpha, rank=4, lam_keep=4.0, lam_reg=5.0)
                    ed.apply_lora_edit(l, "mlp_down", la, lb, meta=dict(alpha=alpha, selectivity=rd["selectivity"]))
            score_arm(f"sameprob_nullspace_lora_a{a}", apply_fn=_ns)

        # Context-to-Weight Internalization of PROMPT_CAREFUL (dropped prompt at test time!)
        for a in (0.5, 1.0):
            def _ci(alpha=a):
                for l in intern_layers:
                    la, lb, rd = compile_context_internalization_lora(d_bare_i[l], d_care_i[l], alpha=alpha, rank=8, lam_keep=2.0, lam_reg=5.0)
                    ed.apply_lora_edit(l, "mlp_down", la, lb, meta=dict(alpha=alpha, fit_err=rd["fit_rel_err"]))
            score_arm(f"ctx_internalize_careful_a{a}", apply_fn=_ci)

        # MoE Router Self-Rewiring (strictly same-problem C vs W!)
        if moe:
            for a in (0.30, 0.55):
                def _rb(alpha=a):
                    for l in top_r_layers:
                        con = compute_router_contrast(diag_cw[l], cw_pos, cw_neg)
                        ed.apply_router_edit(l, bias=compile_router_bias(con, alpha=alpha, mode="t_stat", top_m=32))
                score_arm(f"sameprob_router_bias_a{a}", apply_fn=_rb)
            def _rl():
                for l in top_r_layers:
                    con = compute_router_contrast(diag_cw[l], cw_pos, cw_neg)
                    la, lb, _ = compile_router_nullspace_lora(diag_cw[l], con, alpha=0.50, rank=8, lam_keep=4.0, lam_reg=5.0)
                    ed.apply_router_edit(l, lora_a=la, lora_b=lb)
            score_arm("sameprob_router_nullspace_a0.50", apply_fn=_rl)

            def _rci():
                for l in intern_layers:
                    la, lb, _ = compile_router_context_internalization(d_bare_i[l], d_care_i[l], alpha=0.6, rank=8, lam_keep=2.0, lam_reg=5.0)
                    ed.apply_router_edit(l, lora_a=la, lora_b=lb)
            score_arm("ctx_internalize_router_careful_a0.6", apply_fn=_rci)

        # Mandatory Controls: Random-Direction Norm-Matched & Inference-Time Steering
        def _rand():
            rng = np.random.default_rng(77777)
            for l in top_d_layers[:4]:
                la, lb, _ = compile_nullspace_mlp_lora(diag_cw[l], cw_pos, cw_neg, alpha=0.35, rank=4, lam_keep=4.0, lam_reg=5.0)
                rm = rng.normal(size=lb.shape).astype(np.float32)
                rm = rm * (np.linalg.norm(lb) / (np.linalg.norm(rm) + 1e-12))
                ed.apply_lora_edit(l, "mlp_down", la, rm)
        score_arm("random_matched_control", apply_fn=_rand)

        def _steer():
            for l in top_d_layers[:4]:
                ci = compile_contrastive_direction(diag_cw[l], cw_pos, cw_neg)
                ed.apply_steering_control(l, (0.30 * ci["mean_norm"]) * ci["d_diff"])
        score_arm("steering_control_TGAS", apply_fn=_steer)

    print(f"\nExperiment B complete -> {out_json}", flush=True)


if __name__ == "__main__":
    main()
