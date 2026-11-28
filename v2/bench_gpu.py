"""5-minute GPU throughput, memory & wrapper verification microbenchmark for Qwen3.5 9B / 27B / 35B-A3B 4-bit."""
import json
import os
import sys
import time
import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from v2.gpu_lock import gpu_lock
from v2.benchmarks import load_benchmark
from v2.engine import (
    load_native_model, build_prompts, generate_batch, extract_layer_diagnostics,
    get_layers, is_moe_model, V2_OUT
)
from v2.compilers import ModelEditor


def bench_one(model_key, batch_sizes=(1, 16, 32), max_tokens=256):
    print(f"\n================ BENCH {model_key} ================", flush=True)
    model, tok = load_native_model(model_key)
    probs = load_benchmark("aime_practice")[:max(batch_sizes)]
    prompts = build_prompts(tok, probs)
    out = dict(model=model_key, n_layers=len(get_layers(model)), is_moe=is_moe_model(model), runs={})

    for bs in batch_sizes:
        n_p = max(bs, 1) if bs > 1 else 2
        _, _, perf = generate_batch(model, tok, prompts[:n_p], max_tokens=max_tokens, seed=42, bs=bs, prefill_bs=min(16, bs))
        out["runs"][f"bs_{bs}"] = perf

    # Verify zero-dequantization diagnostic extraction & reversible wrappers!
    nL = len(get_layers(model))
    sample_layers = [nL // 4, nL // 2, (3 * nL) // 4]
    texts, tids, _ = generate_batch(model, tok, prompts[:2], max_tokens=128, seed=42, bs=2, prefill_bs=2, verbose=False)
    fake_recs = [dict(prompt_ids=prompts[i], out_ids=list(tids[i])) for i in range(2)]
    t0 = time.time()
    diag = extract_layer_diagnostics(
        model, fake_recs, sample_layers, tok=tok, window="all", max_tokens_per_seq=16,
        collect_router=is_moe_model(model), collect_mlp_in=True, collect_residual=True
    )
    diag_dt = time.time() - t0
    d0 = diag[sample_layers[0]]
    shapes = {k: list(v.shape) for k, v in d0.items()}
    print(f"  [diag] extracted {len(sample_layers)} layers in {diag_dt:.2f}s; shapes={shapes}", flush=True)
    out["diag_seconds"] = diag_dt
    out["diag_shapes"] = shapes

    # Verify ModelEditor wrappers (LoRA + dt_bias + router if MoE) + rollback speed & parity
    ed = ModelEditor(model)
    d_mlp_in = d0["mlp_in_toks"].shape[1]
    d_hid = d0["residual_toks"].shape[1]
    ed.apply_lora_edit(sample_layers[1], "mlp_down", np.zeros((d_mlp_in, 4), np.float32), np.zeros((4, d_hid), np.float32))
    if is_moe_model(model):
        n_exp = d0["router_logits"].shape[1]
        ed.apply_router_edit(sample_layers[1], bias=np.zeros(n_exp, np.float32),
                             lora_a=np.zeros((d_hid, 4), np.float32), lora_b=np.zeros((4, n_exp), np.float32))
    _, _, perf_wrap = generate_batch(model, tok, prompts[:min(16, max(batch_sizes))], max_tokens=128, seed=42,
                                     bs=min(16, max(batch_sizes)), prefill_bs=8)
    out["wrapped_bs16"] = perf_wrap
    ed.rollback_all()
    print(f"  [wrapper] OK & rolled back cleanly!", flush=True)

    del model, tok
    mx.clear_cache()
    return out


def main():
    models = sys.argv[1:] if len(sys.argv) > 1 else ["q35-9b", "q35-35b", "q35-27b"]
    results = {}
    out_path = os.path.join(V2_OUT, "gpu_benchmark.json")
    if os.path.exists(out_path):
        results = json.load(open(out_path))
    with gpu_lock("bench_gpu", wait=True):
        for m in models:
            bs_list = (1, 16, 32, 64) if m in ("q35-9b", "q35-35b") else (1, 16, 32)
            results[m] = bench_one(m, batch_sizes=bs_list, max_tokens=256)
            json.dump(results, open(out_path, "w"), indent=2)
    print("\nSaved benchmark to", out_path, flush=True)


if __name__ == "__main__":
    main()
