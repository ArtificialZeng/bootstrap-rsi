"""Pilot 2: Training-Free Context-to-Weight Internalization (Zero-Backprop Self-Distillation).

Tests whether a reasoning/verification prompt scaffold p that improves accuracy in-context
can be compiled directly into permanent weights (LoRALinear + MoE Router mlp.gate) via
closed-form null-space ridge regression — and retain gains AFTER dropping the prompt!
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
    load_native_model, eval_problems, extract_layer_diagnostics, get_layers, is_moe_model,
    PROMPT_CONCISE, PROMPT_VERIFY, V2_OUT
)
from v2.compilers import (
    ModelEditor, solve_nullspace_ridge, compile_context_internalization_lora
)
from v2.stats import summarize_records, compare_multiseed, format_comparison


SCAFFOLDS = {
    "verify": PROMPT_VERIFY,
    "concise": PROMPT_CONCISE,
    "structured": (
        "\nFollow a strict 3-stage plan: (1) Identify key invariants and constraints concisely; "
        "(2) Solve cleanly without circular detours; (3) Independently verify your final numeric answer "
        "by plugging it back into the original conditions before writing \\boxed{}."
    ),
}


def compile_router_context_internalization(diag_bare, diag_ctx, alpha=0.8, rank=8, lam_keep=1.0, lam_reg=5.0,
                                           improved_rec_ids=None, preserve_rec_ids=None):
    """Compile router logit shift (mean_logits_ctx - logits_bare) into state-conditional mlp.gate LoRA!

    For problems where the prompt scaffold helped (improved_rec_ids), pushes bare hidden states H_bare
    toward the scaffolded router logit distribution; preserves already-correct bare states via null-space!
    """
    rec_idx_b = diag_bare["tok_rec_idx"]
    rec_idx_c = diag_ctx["tok_rec_idx"]
    H_b = diag_bare["pre_mlp_norm"]
    L_b = diag_bare["router_logits"]
    L_c = diag_ctx["router_logits"]

    n_recs = max(int(rec_idx_b.max()), int(rec_idx_c.max())) + 1
    target_rids = improved_rec_ids if improved_rec_ids else list(range(n_recs))

    X_list, dY_list = [], []
    for r in target_rids:
        mb = rec_idx_b == r
        mc = rec_idx_c == r
        if not np.any(mb) or not np.any(mc):
            continue
        target_l = L_c[mc].mean(axis=0, keepdims=True)
        delta_l = alpha * (target_l - L_b[mb])
        X_list.append(H_b[mb])
        dY_list.append(delta_l)

    X_fail = np.vstack(X_list)
    Y_target = np.vstack(dY_list)
    X_keep = None
    if preserve_rec_ids:
        mk = np.isin(rec_idx_b, list(preserve_rec_ids))
        if np.any(mk):
            X_keep = H_b[mk]

    return solve_nullspace_ridge(X_fail, Y_target, X_keep=X_keep, lam_keep=lam_keep, lam_reg=lam_reg, rank=rank)


def run_internalize(args):
    think_tag = "think" if args.thinking else "nothink"
    tag = f"intern_{args.model}_{args.bench}_{args.scaffold}_{think_tag}_c{args.cap}_p{args.n_prac}_g{args.n_gate}"
    run_dir = os.path.join(V2_OUT, tag)
    os.makedirs(run_dir, exist_ok=True)
    summary_path = os.path.join(run_dir, "summary.json")

    model, tok = load_native_model(args.model)
    nL = len(get_layers(model))
    moe = is_moe_model(model)
    extra_prompt = SCAFFOLDS[args.scaffold]

    prac_probs = load_benchmark("aime_practice" if args.bench == "aime" else "oly_math_practice")[:args.n_prac]
    gate_probs = load_benchmark("aime_gate" if args.bench == "aime" else "oly_math_eval")[:args.n_gate]

    # Step 1: Collect paired bare vs scaffolded practice rollouts
    prac_bare = eval_problems(model, tok, prac_probs, args.cap, seed=100, bs=args.bs,
                              enable_thinking=args.thinking,
                              cache_path=os.path.join(run_dir, "prac_bare.pkl"))
    prac_ctx = eval_problems(model, tok, prac_probs, args.cap, seed=100, bs=args.bs, extra=extra_prompt,
                             enable_thinking=args.thinking,
                             cache_path=os.path.join(run_dir, "prac_ctx.pkl"))

    # Identify problems where scaffold fixed a bare failure vs already-correct problems
    improved = [i for i, (b, c) in enumerate(zip(prac_bare, prac_ctx)) if c["correct"] and not b["correct"]]
    both_ok = [i for i, (b, c) in enumerate(zip(prac_bare, prac_ctx)) if b["correct"] and c["correct"]]
    print(f"  [prac] bare={100*np.mean([r['correct'] for r in prac_bare]):.1f}% "
          f"ctx={100*np.mean([r['correct'] for r in prac_ctx]):.1f}% "
          f"improved={len(improved)} both_ok={len(both_ok)}", flush=True)

    # Step 2: Gate baselines (2 seeds)
    gate_seeds = [1000, 2000]
    gate_bare = [eval_problems(model, tok, gate_probs, args.cap, seed=s, bs=args.bs,
                               enable_thinking=args.thinking,
                               cache_path=os.path.join(run_dir, f"gate_bare_s{s}.pkl")) for s in gate_seeds]
    gate_ctx = [eval_problems(model, tok, gate_probs, args.cap, seed=s, bs=args.bs, extra=extra_prompt,
                              enable_thinking=args.thinking,
                              cache_path=os.path.join(run_dir, f"gate_ctx_s{s}.pkl")) for s in gate_seeds]
    c_prompt = compare_multiseed(gate_ctx, gate_bare)
    print("  " + format_comparison(f"IN-CONTEXT ({args.scaffold})", c_prompt), flush=True)

    # Step 3: Extract paired bare & ctx diagnostics on middle-to-upper layers
    target_layers = [int(round(f * (nL - 1))) for f in (0.35, 0.45, 0.55, 0.65, 0.75, 0.85)]
    target_layers = sorted(set(target_layers))
    d_bare = extract_layer_diagnostics(
        model, prac_bare, target_layers, tok=tok, window="commit",
        max_seq_len=min(args.cap + 512, 6144), max_tokens_per_seq=40,
        collect_router=moe, collect_mlp_in=True, collect_residual=True,
    )
    d_ctx = extract_layer_diagnostics(
        model, prac_ctx, target_layers, tok=tok, window="commit",
        max_seq_len=min(args.cap + 512, 6144), max_tokens_per_seq=40,
        collect_router=moe, collect_mlp_in=False, collect_residual=True,
    )

    summary = dict(tag=tag, model=args.model, scaffold=args.scaffold, target_layers=target_layers,
                   n_improved_prac=len(improved), comparisons={"in_context_prompt": c_prompt})
    ed = ModelEditor(model)

    def eval_internalized(arm_name, apply_fn):
        apply_fn()
        try:
            runs = [eval_problems(model, tok, gate_probs, args.cap, seed=s, bs=args.bs, extra="",
                                  enable_thinking=args.thinking,
                                  cache_path=os.path.join(run_dir, f"gate_{arm_name}_s{s}.pkl"))
                    for s in gate_seeds]
        finally:
            ed.rollback_all()
        c = compare_multiseed(runs, gate_bare)
        summary["comparisons"][arm_name] = c
        print("  " + format_comparison(f"BARE + {arm_name}", c), flush=True)
        json.dump(summary, open(summary_path, "w"), indent=2, default=str)

    # (A) MLP Null-Space LoRA Internalization (zero prompt at test time!)
    for alpha in (0.4, 0.8):
        def _apply_mlp(a=alpha):
            for l in target_layers:
                la, lb, rd = compile_context_internalization_lora(
                    d_bare[l], d_ctx[l], alpha=a, rank=8, lam_keep=2.0, lam_reg=5.0
                )
                ed.apply_lora_edit(l, "mlp_down", la, lb, meta=dict(alpha=a, fit_err=rd["fit_rel_err"]))
        eval_internalized(f"intern_mlp_a{alpha}", _apply_mlp)

    # (B) MoE Router Context Internalization (zero prompt & ~0% latency overhead!)
    if moe:
        for alpha in (0.4, 0.8):
            def _apply_rt(a=alpha):
                for l in target_layers:
                    la, lb, rd = compile_router_context_internalization(
                        d_bare[l], d_ctx[l], alpha=a, rank=8, lam_keep=2.0, lam_reg=5.0,
                        improved_rec_ids=improved if len(improved) >= 4 else None,
                        preserve_rec_ids=both_ok if len(both_ok) >= 4 else None,
                    )
                    ed.apply_router_edit(l, lora_a=la, lora_b=lb, meta=dict(alpha=a, selectivity=rd["selectivity"]))
            eval_internalized(f"intern_router_a{alpha}", _apply_rt)

    json.dump(summary, open(summary_path, "w"), indent=2, default=str)
    print(f"Internalization pilot complete -> {summary_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="q35-35b", choices=["q35-9b", "q35-27b", "q35-35b"])
    ap.add_argument("--bench", default="aime", choices=["aime", "oly_math"])
    ap.add_argument("--scaffold", default="structured", choices=list(SCAFFOLDS))
    ap.add_argument("--cap", type=int, default=4096)
    ap.add_argument("--n-prac", type=int, default=45)
    ap.add_argument("--n-gate", type=int, default=60)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--no-thinking", dest="thinking", action="store_false", default=True)
    args = ap.parse_args()
    with gpu_lock(f"intern_{args.model}", wait=True):
        run_internalize(args)


if __name__ == "__main__":
    main()
