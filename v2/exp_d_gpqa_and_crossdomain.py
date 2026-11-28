"""Experiment D: High-Latent-Gap GPQA-Diamond Same-Problem (C, W) Editing + AIME->Physics Cross-Domain Transfer.

Part 1 (GPQA-Diamond, cap=768 Brief-CoT, pass@1=65% -> pass@3=80%):
  - Mines 64 GPQA-Diamond practice problems x 5 seeds (~18-25 strictly same-problem (C, W) pairs!)
  - Quantifies Unpaired vs Same-Problem Mirage on GPQA-Diamond
  - Evaluates all training-free compilers + controls on the remaining 134 unseen GPQA-Diamond problems x 2 seeds (= 268 evals)!

Part 2 (Cross-Domain Competition Math -> Physics Transfer, nothink cap=3072):
  - Evaluates dt_bias (+0.5, +0.8), prompt_concise, and random-sign dt_bias on AIME 2022-2025 (n=64)
    and zero-shot out-of-domain OlympiadBench-Physics (n=64)!
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from v2.gpu_lock import gpu_lock
from v2.benchmarks import load_benchmark
from v2.engine import (
    load_native_model, eval_problems, extract_layer_diagnostics, get_layers, is_moe_model,
    PROMPT_CONCISE, V2_OUT
)
from v2.compilers import (
    ModelEditor, compute_router_contrast, compile_router_bias, compile_router_nullspace_lora,
    compile_contrastive_direction, compile_nullspace_mlp_lora
)
from v2.pilot_breakthrough import find_same_problem_pairs
from v2.stats import summarize_records, compare_paired, compare_multiseed, format_comparison

BRIEF_EXTRA = "\nBe concise: give only a brief 3-5 sentence justification (under 200 words) and immediately put your final single capital letter in \\boxed{}."


def run_exp_d():
    run_dir = os.path.join(V2_OUT, "expD_gpqa_and_crossdomain_q35_35b")
    os.makedirs(run_dir, exist_ok=True)
    out_json = os.path.join(run_dir, "summary.json")
    summary = dict(part1_gpqa={}, part2_crossdomain={})

    def save():
        json.dump(summary, open(out_json, "w"), indent=2, default=str)

    with gpu_lock("expD_q35_35b", wait=True):
        model, tok = load_native_model("q35-35b")
        nL = len(get_layers(model))
        ed = ModelEditor(model)
        lin_layers = [i for i, b in enumerate(get_layers(model)) if getattr(b, "is_linear", False)]
        n_v_heads = int(get_layers(model)[lin_layers[0]].linear_attn.dt_bias.shape[0])

        # ==================== PART 1: GPQA-DIAMOND SAME-PROBLEM (C, W) ====================
        print("\n================ EXP D PART 1: GPQA-DIAMOND SAME-PROBLEM (C, W) ================", flush=True)
        all_gpqa = load_benchmark("gpqa_diamond")
        prac_g = all_gpqa[:64]
        eval_g = all_gpqa[64:]  # 134 strictly unseen GPQA-Diamond problems!

        # Reuse cached s100..s102 from calib_brief_wc if present!
        calib_dir = os.path.join(V2_OUT, "calib_brief_wc")
        prac_runs = []
        for s in (100, 101, 102, 103, 104):
            cp = os.path.join(run_dir, f"gpqa_prac_s{s}.pkl")
            alt = os.path.join(calib_dir, f"gpqa_diamond_s{s}.pkl")
            if not os.path.exists(cp) and os.path.exists(alt):
                import shutil; shutil.copyfile(alt, cp)
            recs = eval_problems(model, tok, prac_g, 768, seed=s, bs=64, temp=0.85,
                                 extra=BRIEF_EXTRA, enable_thinking=False, cache_path=cp)
            prac_runs.append(recs)

        pairs = find_same_problem_pairs(prac_runs)
        summary["part1_gpqa"]["mining"] = pairs["stats"]
        print("  [GPQA mining]", json.dumps(pairs["stats"], indent=2), flush=True)
        cw_recs, cw_pos, cw_neg, cw_pids = pairs["cw"]
        summary["part1_gpqa"]["n_sameprob_pairs"] = len(cw_pos)

        scan_layers = sorted(set(list(range(2, nL - 1, 2)) + [nL - 2]))
        diag_cw = extract_layer_diagnostics(
            model, cw_recs, scan_layers, tok=tok, window="all",
            max_seq_len=1280, max_tokens_per_seq=48,
            collect_router=True, collect_mlp_in=True, collect_residual=True,
        )

        # Unpaired Mirage check (strictly disjoint problems, matched N!)
        cw_set = set(cw_pids)
        dc = [r for r in pairs["all_C"] if r["pid"] not in cw_set]
        dw = [r for r in pairs["all_W"] if r["pid"] not in cw_set]
        mu = min(len(dc), len(dw), len(cw_pos))
        diag_unp = extract_layer_diagnostics(
            model, dc[:mu] + dw[:mu], scan_layers, tok=tok, window="all",
            max_seq_len=1280, max_tokens_per_seq=48,
            collect_router=True, collect_mlp_in=False, collect_residual=True,
        ) if mu >= 4 else None

        prof_s, prof_u = {}, {}
        for l in scan_layers:
            cd = compile_contrastive_direction(diag_cw[l], cw_pos, cw_neg)
            rc = compute_router_contrast(diag_cw[l], cw_pos, cw_neg)
            prof_s[l] = dict(cohen_d=cd["cohen_d"], router_snr=rc["layer_snr"])
            if diag_unp is not None:
                cdu = compile_contrastive_direction(diag_unp[l], list(range(mu)), list(range(mu, 2 * mu)))
                rcu = compute_router_contrast(diag_unp[l], list(range(mu)), list(range(mu, 2 * mu)))
                prof_u[l] = dict(cohen_d=cdu["cohen_d"], router_snr=rcu["layer_snr"])
        summary["part1_gpqa"]["profile_same"] = prof_s
        summary["part1_gpqa"]["profile_unpaired"] = prof_u
        if prof_u:
            print(f"  [GPQA MIRAGE, N={mu}] |Cohen d|: Unpaired={np.mean([abs(v['cohen_d']) for v in prof_u.values()]):.3f} "
                  f"vs Same={np.mean([abs(v['cohen_d']) for v in prof_s.values()]):.3f} | "
                  f"Router SNR: Unpaired={np.mean([v['router_snr'] for v in prof_u.values()]):.2f} "
                  f"vs Same={np.mean([v['router_snr'] for v in prof_s.values()]):.2f}", flush=True)

        top_r = sorted(scan_layers, key=lambda l: prof_s[l]["router_snr"], reverse=True)[:8]
        top_d = sorted(scan_layers, key=lambda l: abs(prof_s[l]["cohen_d"]), reverse=True)[:4]

        seeds = [1000, 2000]
        def eval_g_arm(name, apply_fn=None):
            runs = []
            hit = all(os.path.exists(os.path.join(run_dir, f"gpqa_eval_{name}_s{s}.pkl")) for s in seeds)
            if not hit and apply_fn:
                apply_fn()
            try:
                for s in seeds:
                    runs.append(eval_problems(model, tok, eval_g, 768, seed=s, bs=64,
                                              extra=BRIEF_EXTRA, enable_thinking=False,
                                              cache_path=os.path.join(run_dir, f"gpqa_eval_{name}_s{s}.pkl")))
            finally:
                if not hit and apply_fn:
                    ed.rollback_all()
            return runs

        base_g = eval_g_arm("base")
        summary["part1_gpqa"]["noise_floor"] = compare_paired(base_g[1], base_g[0])
        print("  " + format_comparison("GPQA NOISE FLOOR", summary["part1_gpqa"]["noise_floor"]), flush=True)

        summary["part1_gpqa"]["comparisons"] = {}
        for nm, fn in [
            ("dt_bias_+0.5", lambda: ed.apply_dt_bias_edit(0.5)),
            ("sameprob_router_bias_a0.35", lambda: [ed.apply_router_edit(l, bias=compile_router_bias(compute_router_contrast(diag_cw[l], cw_pos, cw_neg), alpha=0.35, mode="t_stat", top_m=32)) for l in top_r]),
            ("sameprob_router_nullspace_a0.45", lambda: [ed.apply_router_edit(l, **dict(zip(("lora_a", "lora_b"), compile_router_nullspace_lora(diag_cw[l], compute_router_contrast(diag_cw[l], cw_pos, cw_neg), alpha=0.45, rank=8, lam_keep=4.0, lam_reg=5.0)[:2]))) for l in top_r]),
            ("sameprob_nullspace_mlp_a0.45", lambda: [ed.apply_lora_edit(l, "mlp_down", *compile_nullspace_mlp_lora(diag_cw[l], cw_pos, cw_neg, alpha=0.45, rank=4, lam_keep=4.0, lam_reg=5.0)[:2]) for l in top_d]),
            ("random_router_a0.35", lambda: [ed.apply_router_edit(l, bias=compile_router_bias(compute_router_contrast(diag_cw[l], cw_pos, cw_neg), alpha=0.35, mode="t_stat", top_m=32) * np.random.default_rng(404 + l).choice([-1.0, 1.0], size=256).astype(np.float32)) for l in top_r]),
            ("steering_TGAS", lambda: [ed.apply_steering_control(l, (0.25 * compile_contrastive_direction(diag_cw[l], cw_pos, cw_neg)["mean_norm"]) * compile_contrastive_direction(diag_cw[l], cw_pos, cw_neg)["d_diff"]) for l in top_d]),
        ]:
            runs = eval_g_arm(nm, apply_fn=fn)
            c = compare_multiseed(runs, base_g)
            summary["part1_gpqa"]["comparisons"][nm] = c
            print("  " + format_comparison(f"GPQA {nm}", c), flush=True)
            save()

        # ==================== PART 2: AIME -> OLYMPIAD PHYSICS CROSS-DOMAIN TRANSFER ====================
        print("\n================ EXP D PART 2: AIME 22-25 & OLYMPIAD-PHYSICS CROSS-DOMAIN (nothink cap=3072) ================", flush=True)
        for bname, bsplit in [("aime_test", load_benchmark("aime_test")[:64]), ("oly_phys", load_benchmark("oly_phys")[:64])]:
            summary["part2_crossdomain"][bname] = {}
            def eval_cd(nm, extra="", apply_fn=None, seed=1000):
                cp = os.path.join(run_dir, f"cd_{bname}_{nm}_s{seed}.pkl")
                if not os.path.exists(cp) and apply_fn:
                    apply_fn()
                try:
                    return eval_problems(model, tok, bsplit, 3072, seed=seed, bs=64,
                                         extra=extra, enable_thinking=False, cache_path=cp)
                finally:
                    if apply_fn:
                        ed.rollback_all()

            b1 = eval_cd("base", seed=1000)
            b2 = eval_cd("base", seed=2000)
            summary["part2_crossdomain"][bname]["noise_floor"] = compare_paired(b2, b1)
            print("  " + format_comparison(f"{bname} NOISE FLOOR", summary["part2_crossdomain"][bname]["noise_floor"]), flush=True)

            for nm, ex, fn in [
                ("prompt_concise", PROMPT_CONCISE, None),
                ("dt_bias_+0.5", "", lambda: ed.apply_dt_bias_edit(0.5)),
                ("dt_bias_+0.8", "", lambda: ed.apply_dt_bias_edit(0.8)),
                ("dt_bias_rand_0.5", "", lambda: ed.apply_dt_bias_edit({
                    l: np.random.default_rng(8000 + l).choice([-0.5, 0.5], size=(n_v_heads,)).astype(np.float32)
                    for l in lin_layers
                })),
            ]:
                r1 = eval_cd(nm, extra=ex, apply_fn=fn, seed=1000)
                r2 = eval_cd(nm, extra=ex, apply_fn=fn, seed=2000)
                c = compare_multiseed([r1, r2], [b1, b2])
                summary["part2_crossdomain"][bname][nm] = c
                print("  " + format_comparison(f"{bname} {nm}", c), flush=True)
                save()

    print(f"\nExperiment D complete -> {out_json}", flush=True)


if __name__ == "__main__":
    run_exp_d()
