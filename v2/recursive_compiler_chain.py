"""Multi-Generation Orthogonal Recursive Self-Evolution Chain (Strict-Gated Analytical Compiler).

Tests whether compounding training-free weight updates across generations (Gen 0 -> Gen 1 -> Gen 2 -> Gen 3)
using inherited weights M_g to diagnose residual failures and compile new orthogonal updates (dt_bias,
MoE Router Self-Rewiring, Null-Space Ridge LoRA) achieves multi-generation gains on unseen AIME 2022-2025,
OlympiadBench-Math, and out-of-domain OlympiadBench-Physics.
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
    compile_contrastive_direction, compile_nullspace_mlp_lora
)
from v2.pilot_breakthrough import find_same_problem_pairs
from v2.stats import summarize_records, compare_multiseed, strict_gate_decision, format_comparison


def run_recursive_chain(args):
    think_tag = "think" if args.thinking else "nothink"
    tag = f"rchain_{args.model}_{args.bench}_{think_tag}_c{args.cap}_g{args.gens}"
    run_dir = os.path.join(V2_OUT, tag)
    os.makedirs(run_dir, exist_ok=True)
    summary_path = os.path.join(run_dir, "summary.json")

    model, tok = load_native_model(args.model)
    nL = len(get_layers(model))
    moe = is_moe_model(model)
    has_linear = any(getattr(b, "is_linear", False) for b in get_layers(model))
    ed = ModelEditor(model)

    prac_probs = load_benchmark("aime_practice")[:args.n_prac]
    gate_probs = load_benchmark("aime_gate")[:args.n_gate]
    test_probs = load_benchmark("aime_test")
    phys_probs = load_benchmark("oly_phys")[:args.n_phys]

    gate_seeds = [1000, 2000]
    test_seeds = [1000, 2000, 3000][:args.test_seeds]

    def eval_multi(probs, seeds, prefix):
        return [
            eval_problems(model, tok, probs, args.cap, seed=s, bs=args.bs,
                          enable_thinking=args.thinking,
                          cache_path=os.path.join(run_dir, f"{prefix}_s{s}.pkl"))
            for s in seeds
        ]

    # Generation 0 baselines
    gen0_gate = eval_multi(gate_probs, gate_seeds, "gen0_gate")
    gen0_test = eval_multi(test_probs, test_seeds, "gen0_test")
    gen0_phys = eval_multi(phys_probs, [1000], "gen0_phys") if args.n_phys > 0 else None

    cur_gate = gen0_gate
    log = dict(
        tag=tag, model=args.model, cap=args.cap, n_layers=nL, is_moe=moe,
        gen0_gate=[summarize_records(r, f"g0_gate_s{s}") for r, s in zip(gen0_gate, gate_seeds)],
        gen0_test=[summarize_records(r, f"g0_test_s{s}") for r, s in zip(gen0_test, test_seeds)],
        generations=[],
    )
    if gen0_phys:
        log["gen0_phys"] = summarize_records(gen0_phys[0], "g0_phys")

    scan_layers = sorted(set(list(range(2, nL - 1, 2)) + [nL - 2]))

    for g in range(1, args.gens + 1):
        print(f"\n================ RECURSIVE GENERATION {g} ({tag}) ================", flush=True)
        # Re-generate practice rollouts under CURRENT INHERITED WEIGHTS M_{g-1}!
        prac_seeds = [100 * g + s for s in range(args.prac_k)]
        prac_runs = eval_multi(prac_probs, prac_seeds, f"gen{g-1}_prac")
        pairs = find_same_problem_pairs(prac_runs)
        print(f"  [gen{g} mining] {pairs['stats']}", flush=True)

        cw_recs, cw_pos, cw_neg, _ = pairs["cw"]
        if len(cw_pos) < 4:
            m = min(len(pairs["all_C"]), len(pairs["all_W"]), 40)
            cw_recs = pairs["all_C"][:m] + pairs["all_W"][:m]
            cw_pos = list(range(m))
            cw_neg = list(range(m, 2 * m))

        diag_cw = extract_layer_diagnostics(
            model, cw_recs, scan_layers, tok=tok, window=args.window,
            max_seq_len=min(args.cap + 512, 6144), max_tokens_per_seq=48,
            collect_router=moe, collect_mlp_in=True, collect_residual=True,
        )

        # Rank layers by MoE router SNR & residual Cohen's d
        r_snr = {}
        d_snr = {}
        for l in scan_layers:
            d_snr[l] = abs(compile_contrastive_direction(diag_cw[l], cw_pos, cw_neg)["cohen_d"])
            if moe and "router_logits" in diag_cw[l]:
                r_snr[l] = compute_router_contrast(diag_cw[l], cw_pos, cw_neg)["layer_snr"]
        top_r = sorted(r_snr, key=r_snr.get, reverse=True)[:8] if r_snr else []
        top_d = sorted(d_snr, key=d_snr.get, reverse=True)[:4]

        # Build candidate orthogonal edits appropriate for generation g's residual failure mode!
        # If truncation > 20% and dt_bias not yet used, include dt_bias; always include W->C router & nullspace LoRA!
        trunc_rate = pairs["stats"]["total_T"] / max(1, pairs["stats"]["n_rollouts"])
        used_kinds = {e["kind"] for _, es in ed._committed for _, e in es}
        candidates = []

        if has_linear and trunc_rate > 0.15 and "dt_bias" not in used_kinds:
            candidates.append(("dt_bias_0.35", lambda: ed.apply_dt_bias_edit(0.35, meta=dict(gen=g))))
            candidates.append(("dt_bias_0.50", lambda: ed.apply_dt_bias_edit(0.50, meta=dict(gen=g))))

        if moe:
            for a in (0.30, 0.50):
                def _rb(alpha=a, ls=top_r):
                    for l in ls:
                        con = compute_router_contrast(diag_cw[l], cw_pos, cw_neg)
                        ed.apply_router_edit(l, bias=compile_router_bias(con, alpha=alpha, mode="t_stat", top_m=32),
                                             meta=dict(gen=g, alpha=alpha))
                candidates.append((f"router_bias_a{a}", _rb))

            def _rl(ls=top_r):
                for l in ls:
                    con = compute_router_contrast(diag_cw[l], cw_pos, cw_neg)
                    la, lb, rd = compile_router_nullspace_lora(diag_cw[l], con, alpha=0.45, rank=8, lam_keep=4.0, lam_reg=5.0)
                    ed.apply_router_edit(l, lora_a=la, lora_b=lb, meta=dict(gen=g, selectivity=rd["selectivity"]))
            candidates.append(("router_nullspace_a0.45", _rl))

        for a in (0.35, 0.60):
            def _nl(alpha=a, ls=top_d):
                for l in ls:
                    la, lb, rd = compile_nullspace_mlp_lora(diag_cw[l], cw_pos, cw_neg, alpha=alpha, rank=4, lam_keep=4.0, lam_reg=5.0)
                    ed.apply_lora_edit(l, "mlp_down", la, lb, meta=dict(gen=g, alpha=alpha, selectivity=rd["selectivity"]))
            candidates.append((f"nullspace_lora_a{a}", _nl))

        # Evaluate candidates on STRICT multi-seed gate vs CURRENT generation M_{g-1}
        cand_results = []
        best_cand = None
        for cname, apply_fn in candidates:
            apply_fn()
            try:
                g_runs = eval_multi(gate_probs, gate_seeds, f"gen{g}_cand_{cname}_gate")
            finally:
                ed.rollback_staged()
            comp = compare_multiseed(g_runs, cur_gate)
            gate_dec = strict_gate_decision(comp, min_delta=args.min_delta, min_z=args.min_z, max_p_one=0.08)
            print("  " + format_comparison(f"g{g} {cname}", comp) + f" | GATE={gate_dec['passed']} (z={gate_dec['z']:.2f})", flush=True)
            cand_results.append(dict(name=cname, comp=comp, gate=gate_dec))
            if gate_dec["passed"]:
                if best_cand is None or comp["delta"] > best_cand[1]["delta"]:
                    best_cand = (cname, comp, gate_dec, apply_fn, g_runs)

        gen_entry = dict(gen=g, mining=pairs["stats"], candidates=cand_results, kept=None)
        if best_cand is None:
            print(f"  [gen{g}] No candidate cleared strict gate -> chain terminates cleanly (zero Goodhart drift!).", flush=True)
            log["generations"].append(gen_entry)
            json.dump(log, open(summary_path, "w"), indent=2, default=str)
            break

        cname, comp, gate_dec, apply_fn, g_runs = best_cand
        apply_fn()
        ed.commit_staged(tag=f"gen{g}_{cname}")
        cur_gate = g_runs
        gen_entry["kept"] = dict(name=cname, gate_delta=comp["delta"], gate_reasoning=comp["delta_reasoning"], gate_z=gate_dec["z"])
        print(f"  [gen{g}] COMMITTED {cname} into persistent model weights!", flush=True)

        # Track held-out test trajectory across committed generations
        g_test = eval_multi(test_probs, test_seeds, f"gen{g}_test")
        c_vs_0 = compare_multiseed(g_test, gen0_test)
        gen_entry["heldout_vs_gen0"] = c_vs_0
        print("  HELD-OUT vs GEN0: " + format_comparison(f"gen{g} ({cname})", c_vs_0), flush=True)
        log["generations"].append(gen_entry)
        json.dump(log, open(summary_path, "w"), indent=2, default=str)

    # Out-of-domain transfer on OlympiadBench-Physics for final committed model!
    if ed._committed and gen0_phys is not None:
        fin_phys = eval_multi(phys_probs, [1000], f"gen{len(ed._committed)}_phys")
        c_phys = compare_multiseed(fin_phys, gen0_phys)
        log["ood_physics_transfer"] = c_phys
        print("\nOOD PHYSICS TRANSFER: " + format_comparison("final_vs_gen0_phys", c_phys), flush=True)

    log["final_edits"] = ed.summary()
    json.dump(log, open(summary_path, "w"), indent=2, default=str)
    print(f"\nRecursive chain finished -> {summary_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="q35-35b", choices=["q35-9b", "q35-27b", "q35-35b"])
    ap.add_argument("--bench", default="aime")
    ap.add_argument("--cap", type=int, default=4096)
    ap.add_argument("--gens", type=int, default=3)
    ap.add_argument("--n-prac", type=int, default=45)
    ap.add_argument("--prac-k", type=int, default=4)
    ap.add_argument("--n-gate", type=int, default=90)
    ap.add_argument("--n-phys", type=int, default=80)
    ap.add_argument("--test-seeds", type=int, default=2)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--window", default="all")
    ap.add_argument("--no-thinking", dest="thinking", action="store_false", default=True)
    ap.add_argument("--min-delta", type=float, default=0.015)
    ap.add_argument("--min-z", type=float, default=1.4)
    args = ap.parse_args()
    with gpu_lock(f"rchain_{args.model}", wait=True):
        run_recursive_chain(args)


if __name__ == "__main__":
    main()
