"""Experiment A (Fast Dose Curve): Rescuing Compulsive Overthinking on Qwen3.5-35B-A3B Thinking Mode.

Tests whether training-free Gated DeltaNet dt_bias rewriting rescues Qwen3.5-35B-A3B (enable_thinking=True)
from catastrophic over-verification on AIME 2024-2025 (cap=4096), compared against concise prompting and
same-magnitude random-sign dt_bias controls. Also logs exact T->C vs W->C decomposition and token savings!
"""
import argparse
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from v2.gpu_lock import gpu_lock
from v2.benchmarks import load_benchmark
from v2.engine import load_native_model, eval_problems, get_layers, PROMPT_CONCISE, V2_OUT
from v2.compilers import ModelEditor
from v2.stats import summarize_records, compare_paired, format_comparison


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="q35-35b", choices=["q35-9b", "q35-27b", "q35-35b"])
    ap.add_argument("--bench", default="aime_24_25")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--cap", type=int, default=4096)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1000)
    args = ap.parse_args()

    tag = f"expA_{args.model}_{args.bench}_think_c{args.cap}_n{args.n}"
    run_dir = os.path.join(V2_OUT, tag)
    os.makedirs(run_dir, exist_ok=True)
    out_json = os.path.join(run_dir, "summary.json")

    probs = load_benchmark(args.bench)[:args.n]
    summary = dict(tag=tag, model=args.model, bench=args.bench, n=len(probs), cap=args.cap, arms={}, comparisons={})

    with gpu_lock(tag, wait=True):
        model, tok = load_native_model(args.model)
        ed = ModelEditor(model)
        lin_layers = [i for i, b in enumerate(get_layers(model)) if getattr(b, "is_linear", False)]
        n_v_heads = int(get_layers(model)[lin_layers[0]].linear_attn.dt_bias.shape[0])

        def run_arm(name, extra="", apply_fn=None, seed=None):
            s = args.seed if seed is None else seed
            cp = os.path.join(run_dir, f"{name}_s{s}.pkl")
            if os.path.exists(cp):
                recs = eval_problems(model, tok, probs, args.cap, seed=s, bs=args.bs,
                                     extra=extra, enable_thinking=True, cache_path=cp)
            else:
                if apply_fn:
                    apply_fn()
                try:
                    recs = eval_problems(model, tok, probs, args.cap, seed=s, bs=args.bs,
                                         extra=extra, enable_thinking=True, cache_path=cp)
                finally:
                    ed.rollback_all()
            st = summarize_records(recs, name)
            summary["arms"][name] = st
            print(f"  [{name:18s}] acc={100*st['acc']:5.1f}% trunc={100*st['trunc']:5.1f}% "
                  f"C={st['n_C']:2d} W={st['n_W']:2d} T={st['n_T']:2d} acc|fin={100*st['acc_finished']:5.1f}% "
                  f"mean_tok={st['mean_tok']:.0f}", flush=True)
            json.dump(summary, open(out_json, "w"), indent=2, default=str)
            return recs

        base = run_arm("base")
        for name, extra, fn in [
            ("prompt_concise", PROMPT_CONCISE, None),
            ("dt_bias_+0.3", "", lambda: ed.apply_dt_bias_edit(0.3)),
            ("dt_bias_+0.5", "", lambda: ed.apply_dt_bias_edit(0.5)),
            ("dt_bias_+0.8", "", lambda: ed.apply_dt_bias_edit(0.8)),
            ("dt_bias_+1.2", "", lambda: ed.apply_dt_bias_edit(1.2)),
            ("dt_bias_+1.6", "", lambda: ed.apply_dt_bias_edit(1.6)),
            ("dt_bias_rand_0.8", "", lambda: ed.apply_dt_bias_edit({
                l: np.random.default_rng(9000 + l).choice([-0.8, 0.8], size=(n_v_heads,)).astype(np.float32)
                for l in lin_layers
            })),
        ]:
            recs = run_arm(name, extra=extra, apply_fn=fn)
            comp = compare_paired(recs, base)
            summary["comparisons"][name] = comp
            print("    -> " + format_comparison(name, comp), flush=True)
            json.dump(summary, open(out_json, "w"), indent=2, default=str)

    print(f"\nExperiment A complete -> {out_json}", flush=True)


if __name__ == "__main__":
    main()
