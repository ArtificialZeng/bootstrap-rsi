"""High-throughput MLX native-quantized inference, disk caching, and internal activation / MoE router readout."""
import glob
import hashlib
import json
import os
import pickle
import re
import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.generate import batch_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models.base import create_attention_mask, create_ssm_mask

from .benchmarks import grade_response

HF = os.path.expanduser("~/.cache/huggingface/hub")
V2_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(V2_OUT, exist_ok=True)

MODEL_REGISTRY = {
    "q35-4b": "models--mlx-community--Qwen3.5-4B-MLX-bf16",
    "q35-9b": "models--mlx-community--Qwen3.5-9B-4bit",
    "q35-27b": "models--mlx-community--Qwen3.5-27B-4bit",
    "q35-35b": "models--mlx-community--Qwen3.5-35B-A3B-4bit",
    "q3-8b": "models--mlx-community--Qwen3-8B-8bit",
    "q3-14b": "models--mlx-community--Qwen3-14B-8bit",
}

SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
PROMPT_CONCISE = "\nThink efficiently: keep your reasoning concise, avoid repeating steps, and finalize your answer clearly."
PROMPT_VERIFY = (
    "\nBefore writing your final \\boxed{} answer, explicitly double-check your algebra, edge cases, "
    "and arithmetic from scratch using an independent method; if you catch any mistake, fix it."
)


def resolve_model_path(model_key):
    repo_dir = MODEL_REGISTRY.get(model_key, model_key)
    snaps = sorted(glob.glob(f"{HF}/{repo_dir}/snapshots/*"))
    if not snaps:
        raise FileNotFoundError(f"Model snapshot not found for {model_key} ({repo_dir})")
    return snaps[-1]


def load_native_model(model_key, cache_limit_gb=12):
    """Load MLX model natively WITHOUT full-model BF16 dequantization."""
    path = resolve_model_path(model_key)
    t0 = time.time()
    model, tok = load(path)
    mx.set_cache_limit(cache_limit_gb << 30)
    layers = get_layers(model)
    n_lin = sum(1 for b in layers if getattr(b, "is_linear", False))
    n_moe = sum(1 for b in layers if hasattr(b.mlp, "gate") and hasattr(b.mlp, "switch_mlp"))
    print(f"[engine] loaded {model_key} in {time.time()-t0:.1f}s: "
          f"L={len(layers)} (linear_attn={n_lin}, moe={n_moe}) path={os.path.basename(path)}", flush=True)
    return model, tok


def get_text_model(model):
    """Return the underlying text backbone holding embed_tokens and layers (works for Qwen3 & Qwen3.5)."""
    if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
        return model.language_model.model
    return model.model


def get_layers(model):
    return get_text_model(model).layers


def is_moe_model(model):
    layers = get_layers(model)
    return any(hasattr(b.mlp, "gate") and hasattr(b.mlp, "switch_mlp") for b in layers)


def build_prompts(tok, problems, extra="", system_prompt=None, enable_thinking=True):
    out = []
    for p in problems:
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": p["problem"] + SUFFIX + extra})
        try:
            s = tok.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False, enable_thinking=enable_thinking
            )
        except TypeError:
            s = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        out.append(tok.encode(s, add_special_tokens=False))
    return out


def generate_batch(model, tok, prompts, max_tokens, seed=1000, bs=32, prefill_bs=16,
                   temp=0.6, top_p=0.95, top_k=20, verbose=True):
    mx.random.seed(seed)
    mx.reset_peak_memory()
    sampler = make_sampler(temp=temp, top_p=top_p, top_k=top_k)
    t0 = time.time()
    r = batch_generate(
        model, tok, prompts,
        max_tokens=max_tokens,
        sampler=sampler,
        completion_batch_size=bs,
        prefill_batch_size=prefill_bs,
        return_token_ids=True,
    )
    dt = max(1e-6, time.time() - t0)
    ntok = sum(len(x) for x in r.token_ids)
    peak_gb = mx.get_peak_memory() / (1 << 30)
    if verbose:
        print(f"  [gen] n={len(prompts)} cap={max_tokens} bs={bs} seed={seed}: "
              f"{ntok} tok in {dt:.1f}s ({ntok/dt:.1f} tok/s) peak={peak_gb:.1f}GB", flush=True)
    return r.texts, r.token_ids, dict(ntok=ntok, seconds=dt, tok_s=ntok / dt, peak_gb=peak_gb)


def eval_problems(model, tok, problems, max_tokens, seed=1000, bs=32, prefill_bs=16,
                  extra="", system_prompt=None, cache_path=None, temp=0.6, top_p=0.95, top_k=20,
                  enable_thinking=True, verbose=True):
    """Generate & grade problems with optional disk caching."""
    if cache_path and os.path.exists(cache_path):
        data = pickle.load(open(cache_path, "rb"))
        if len(data) == len(problems) and [r["pid"] for r in data] == [p["pid"] for p in problems]:
            if verbose:
                print(f"  [cache hit] {os.path.basename(cache_path)} (n={len(data)})", flush=True)
            return data
    prompts = build_prompts(tok, problems, extra=extra, system_prompt=system_prompt, enable_thinking=enable_thinking)
    texts, tids, perf = generate_batch(
        model, tok, prompts, max_tokens=max_tokens, seed=seed, bs=bs,
        prefill_bs=prefill_bs, temp=temp, top_p=top_p, top_k=top_k, verbose=verbose
    )
    recs = []
    for p, pr, txt, tid in zip(problems, prompts, texts, tids):
        g = grade_response(txt, len(tid), max_tokens, p)
        g.update(
            pid=p["pid"],
            source=p.get("source", ""),
            year=p.get("year", 0),
            level=p.get("level", 0),
            seed=seed,
            text=txt,
            prompt_ids=pr,
            out_ids=list(tid),
        )
        recs.append(g)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        tmp = cache_path + f".tmp.{os.getpid()}"
        with open(tmp, "wb") as f:
            pickle.dump(recs, f)
        os.replace(tmp, cache_path)
    return recs


# ---- Token Window Helpers (for surgical W->C representation extraction) ----

PIVOT_REGEX = re.compile(
    r"\b(wait|however|actually|hold on|check|verify|mistake|wrong|recheck|let me|alternatively|oops|error)\b",
    re.IGNORECASE,
)


def select_token_positions(prompt_len, out_ids, tok=None, window="all", max_tokens_per_seq=64):
    """Select response token indices (relative to full sequence) for representation / router readout.

    Windows:
      - 'all'   : uniformly subsampled across full response
      - 'early' : first 25% of response (initial strategy commitment!)
      - 'mid'   : middle 25%-75% of response (core derivation)
      - 'late'  : final 25% of response (verification & answer commitment!)
      - 'commit': last 128 tokens before </think> (or end of response)
      - 'pivot' : tokens near reasoning-pivot words ('Wait', 'Check', 'Actually', ...)
    """
    n_out = len(out_ids)
    if n_out == 0:
        return []
    lo, hi = 0, n_out
    if window == "early":
        hi = max(1, min(n_out, max(64, n_out // 4)))
    elif window == "mid":
        lo = n_out // 4
        hi = max(lo + 1, (3 * n_out) // 4)
    elif window == "late":
        lo = max(0, (3 * n_out) // 4)
    elif window == "commit":
        # Find </think> token boundary if present
        end_idx = n_out
        if tok is not None:
            try:
                end_think_ids = tok.encode("</think>", add_special_tokens=False)
                if len(end_think_ids) == 1:
                    eid = end_think_ids[0]
                    for j in range(n_out - 1, -1, -1):
                        if out_ids[j] == eid:
                            end_idx = j
                            break
            except Exception:
                pass
        hi = max(1, end_idx)
        lo = max(0, hi - 192)
    elif window == "pivot" and tok is not None:
        # Decode chunks and locate pivot token positions
        pivots = []
        step = 16
        for j in range(0, n_out, step):
            chunk_txt = tok.decode(out_ids[j:j + step])
            if PIVOT_REGEX.search(chunk_txt):
                pivots.extend(range(j, min(n_out, j + step)))
        if len(pivots) >= 4:
            idx = np.linspace(0, len(pivots) - 1, min(max_tokens_per_seq, len(pivots)), dtype=int)
            return [prompt_len + pivots[k] for k in idx]
        # Fallback to mid-late if no explicit pivot markers
        lo = n_out // 3
        hi = n_out

    span = hi - lo
    if span <= max_tokens_per_seq:
        rel = list(range(lo, hi))
    else:
        rel = np.linspace(lo, hi - 1, max_tokens_per_seq, dtype=int).tolist()
    return [prompt_len + r for r in rel]


def extract_layer_diagnostics(
    model,
    records,
    layers,
    tok=None,
    window="all",
    max_seq_len=6144,
    max_tokens_per_seq=48,
    collect_router=True,
    collect_mlp_in=False,
    collect_residual=True,
):
    """Single-pass forward extraction of residual states, MoE router logits/probs, and MLP inputs.

    Works natively on quantized Qwen3 / Qwen3.5 (dense & sparse MoE) with zero backprop.
    Returns dict keyed by layer index:
      - residual_mean : (N_recs, d) float32 mean residual over selected positions per record
      - residual_toks : (N_tok_total, d) float32 token-level residual states
      - tok_rec_idx   : (N_tok_total,) int record index for each token
      - router_logits : (N_tok_total, E) float32 (MoE layers only)
      - router_probs  : (N_tok_total, E) float32 (MoE layers only)
      - router_topk   : (N_tok_total, top_k) int32 (MoE layers only)
      - pre_mlp_norm  : (N_tok_total, d) float32 normalized input to blk.mlp (and blk.mlp.gate!)
      - mlp_in_toks   : (N_tok_total, d_ff) float32 SwiGLU input to down_proj (if collect_mlp_in)
    """
    backbone = get_text_model(model)
    all_layers = backbone.layers
    layers = sorted(set(int(l) for l in layers))
    Lmax = max(layers)
    layer_set = set(layers)

    res = {l: dict(
        residual_mean=[],
        residual_toks=[],
        tok_rec_idx=[],
        pre_mlp_norm=[],
        router_logits=[],
        router_probs=[],
        router_topk=[],
        mlp_in_toks=[],
    ) for l in layers}

    for r_idx, rec in enumerate(records):
        p_ids = list(rec["prompt_ids"])
        o_ids = list(rec["out_ids"])
        full = (p_ids + o_ids)[:max_seq_len]
        eff_out_len = max(0, len(full) - len(p_ids))
        if eff_out_len == 0:
            continue
        pos = select_token_positions(
            len(p_ids), o_ids[:eff_out_len], tok=tok, window=window, max_tokens_per_seq=max_tokens_per_seq
        )
        pos = [p for p in pos if p < len(full)]
        if not pos:
            continue
        pos_mx = mx.array(pos, dtype=mx.int32)

        x = mx.array(full, dtype=mx.int32)[None]
        h = backbone.embed_tokens(x)
        fa_mask = create_attention_mask(h, None)
        ssm_mask = create_ssm_mask(h, None) if any(getattr(b, "is_linear", False) for b in all_layers[:Lmax + 1]) else None

        eval_tensors = []
        pending = {}

        for i in range(Lmax + 1):
            blk = all_layers[i]
            mask = ssm_mask if getattr(blk, "is_linear", False) else fa_mask
            if i not in layer_set:
                h = blk(h, mask=mask, cache=None)
                continue

            # Step manually through block so we capture pre-MLP input (exact router & MLP input!)
            if getattr(blk, "is_linear", False):
                r_attn = blk.linear_attn(blk.input_layernorm(h), mask, None)
            else:
                r_attn = blk.self_attn(blk.input_layernorm(h), mask, None)
            h_mid = h + r_attn
            h_norm = blk.post_attention_layernorm(h_mid)
            h_norm_sel = h_norm[0, pos_mx].astype(mx.float32)

            p_dict = {}
            is_moe = hasattr(blk.mlp, "gate") and hasattr(blk.mlp, "switch_mlp")

            if collect_router and is_moe:
                g_logits = blk.mlp.gate(h_norm_sel).astype(mx.float32)
                g_probs = mx.softmax(g_logits, axis=-1, precise=True)
                k = blk.mlp.top_k
                topk = mx.argpartition(g_probs, kth=-k, axis=-1)[..., -k:]
                p_dict["pre_mlp_norm"] = h_norm_sel
                p_dict["router_logits"] = g_logits
                p_dict["router_probs"] = g_probs
                p_dict["router_topk"] = topk
                eval_tensors.extend([h_norm_sel, g_logits, g_probs, topk])

            if collect_mlp_in:
                # For dense MLP: blk.mlp; for MoE: blk.mlp.shared_expert
                mlp_mod = blk.mlp.shared_expert if is_moe else blk.mlp
                if hasattr(mlp_mod, "gate_proj") and hasattr(mlp_mod, "up_proj"):
                    mlp_act = (nn.silu(mlp_mod.gate_proj(h_norm_sel)) * mlp_mod.up_proj(h_norm_sel)).astype(mx.float32)
                    p_dict["mlp_in_toks"] = mlp_act
                    eval_tensors.append(mlp_act)
                    if "pre_mlp_norm" not in p_dict:
                        p_dict["pre_mlp_norm"] = h_norm_sel
                        eval_tensors.append(h_norm_sel)

            h = h_mid + blk.mlp(h_norm)

            if collect_residual:
                h_sel = h[0, pos_mx].astype(mx.float32)
                h_mean = h_sel.mean(axis=0)
                p_dict["residual_toks"] = h_sel
                p_dict["residual_mean"] = h_mean
                eval_tensors.extend([h_sel, h_mean])

            pending[i] = p_dict

        mx.eval(h, *eval_tensors)
        for l, p_dict in pending.items():
            d = res[l]
            n_t = len(pos)
            d["tok_rec_idx"].append(np.full(n_t, r_idx, dtype=np.int32))
            for k_name, arr in p_dict.items():
                d[k_name].append(np.array(arr))

        # Keep Metal peak memory low across long sequences
        mx.clear_cache()

    # Concatenate into clean numpy arrays
    final = {}
    for l, d in res.items():
        if not d["tok_rec_idx"]:
            continue
        out_l = {"tok_rec_idx": np.concatenate(d["tok_rec_idx"], axis=0)}
        for k_name in ("residual_toks", "pre_mlp_norm", "router_logits", "router_probs", "router_topk", "mlp_in_toks"):
            if d[k_name]:
                out_l[k_name] = np.concatenate(d[k_name], axis=0)
        if d["residual_mean"]:
            out_l["residual_mean"] = np.stack(d["residual_mean"], axis=0)
        final[l] = out_l
    return final
