"""Shared helpers: load Qwen3-1.7B in MLX, generate on MATH-500, grade, read hidden states, edit weights."""
import json, os, time, glob
import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import batch_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from math_verify import parse, verify

HF = os.path.expanduser("~/.cache/huggingface/hub")
MODEL_PATH = glob.glob(f"{HF}/models--Qwen--Qwen3-1.7B/snapshots/*")[0]
DATA_PATH = glob.glob(f"{HF}/datasets--HuggingFaceH4--MATH-500/snapshots/*/test.jsonl")[0]
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
CONCISE = "\nThink efficiently: keep your reasoning short and do not repeat yourself."


def load_data():
    rows = [json.loads(l) for l in open(DATA_PATH)]
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(rows))
    # fixed split: 100 practice (direction extraction / alpha choice), 400 held-out eval
    return [rows[i] for i in idx[:100]], [rows[i] for i in idx[100:]]


def load_model():
    return load(MODEL_PATH)


def build_prompts(tok, problems, extra=""):
    out = []
    for p in problems:
        msgs = [{"role": "user", "content": p["problem"] + SUFFIX + extra}]
        s = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=True)
        out.append(tok.encode(s, add_special_tokens=False))
    return out


def generate(model, tok, prompts, max_tokens, seed=0, bs=64, verbose=True):
    mx.random.seed(seed)
    sampler = make_sampler(temp=0.6, top_p=0.95, top_k=20)  # Qwen3 recommended thinking settings
    t = time.time()
    r = batch_generate(model, tok, prompts, max_tokens=max_tokens, sampler=sampler,
                       completion_batch_size=bs, prefill_batch_size=16, return_token_ids=True)
    dt = time.time() - t
    ntok = sum(len(x) for x in r.token_ids)
    msg = f"  generated {len(prompts)} x <= {max_tokens}: {ntok} tok in {dt:.0f}s ({ntok/dt:.0f} tok/s)"
    if verbose:
        print(msg, flush=True)
    else:
        import sys
        print(msg, file=sys.stderr, flush=True)
    return r.texts, r.token_ids


def last_boxed(s):
    """Content of the last balanced \\boxed{...} in s, or None."""
    i = s.rfind("\\boxed")
    while i >= 0:
        j = s.find("{", i)
        if j >= 0 and s[i + 6:j].strip() == "":
            depth = 0
            for k in range(j, len(s)):
                if s[k] == "{":
                    depth += 1
                elif s[k] == "}":
                    depth -= 1
                    if depth == 0:
                        return s[j + 1:k]
        i = s.rfind("\\boxed", 0, i)
    return None


def grade(text, ntok, max_tokens, gold):
    """Paper rule (Tab. 5): truncated -> wrong; else compare the LAST \\boxed{} after the thinking."""
    truncated = ntok >= max_tokens
    if truncated:
        return dict(truncated=True, correct=False, ntok=ntok)
    b = last_boxed(text.split("</think>")[-1])
    ok = False
    if b is not None:
        try:
            ok = bool(verify(parse("$" + gold + "$"), parse("$\\boxed{" + b + "}$")))
        except Exception:
            ok = False
    return dict(truncated=False, correct=ok, ntok=ntok)


def run_condition(model, tok, problems, max_tokens, n_samples, tag, extra="", seed0=0):
    prompts = build_prompts(tok, problems, extra)
    recs = []
    for s in range(n_samples):
        texts, ids = generate(model, tok, prompts, max_tokens, seed=seed0 + s)
        for i, (p, txt, tid) in enumerate(zip(problems, texts, ids)):
            g = grade(txt, len(tid), max_tokens, p["answer"])
            g.update(pid=p["unique_id"], sample=s, level=p["level"], text=txt, prompt_ids=prompts[i], out_ids=tid)
            recs.append(g)
    path = os.path.join(OUT, f"{tag}.jsonl")
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps({k: v for k, v in r.items() if k not in ("prompt_ids", "out_ids")}) + "\n")
    summarize(recs, tag)
    return recs


def summarize(recs, tag):
    n = len(recs)
    acc = np.mean([r["correct"] for r in recs])
    tr = np.mean([r["truncated"] for r in recs])
    fin = [r for r in recs if not r["truncated"]]
    accf = np.mean([r["correct"] for r in fin]) if fin else float("nan")
    ln = np.mean([r["ntok"] for r in recs])
    print(f"[{tag}] n={n} acc={acc:.3f} trunc={tr:.3f} acc|finished={accf:.3f} mean_len={ln:.0f}", flush=True)
    return dict(tag=tag, n=n, acc=acc, trunc=tr, acc_fin=accf, mean_len=ln)


def hidden_mean(model, prompt_ids, out_ids, layer, max_len=6000):
    """Mean residual-stream state after `layer` over the response tokens (one forward pass)."""
    ids = (list(prompt_ids) + list(out_ids))[:max_len]
    x = mx.array(ids)[None]
    m = model.model
    h = m.embed_tokens(x)
    mask = create_attention_mask(h, None)
    for i, blk in enumerate(m.layers):
        h = blk(h, mask, None)
        if i == layer:
            break
    v = h[0, len(prompt_ids):].astype(mx.float32).mean(axis=0)
    mx.eval(v)
    return np.array(v)


def out_mats(blk):
    """The two matrices of a block that write into the residual stream (Qwen3 and Qwen3.5 layouts)."""
    attn = blk.linear_attn.out_proj if getattr(blk, "is_linear", False) else blk.self_attn.o_proj
    return [attn, blk.mlp.down_proj]


def hidden_states_multi(model, ids, start, layers, max_len=6000):
    """Mean residual state over positions >= start after each layer in `layers` (Qwen3 and Qwen3.5)."""
    ids = list(ids)[:max_len]
    if start >= len(ids):
        raise ValueError("nothing to average")
    x = mx.array(ids)[None]
    m = model.model
    h = m.embed_tokens(x)
    fa, ssm = create_attention_mask(h, None), create_ssm_mask(h, None)
    out, Lmax = {}, max(layers)
    for i, blk in enumerate(m.layers):
        h = blk(h, ssm if getattr(blk, "is_linear", False) else fa, None)
        if i in layers:
            out[i] = h[0, start:].astype(mx.float32).mean(axis=0)
        if i == Lmax:
            break
    mx.eval(list(out.values()))
    return {k: np.array(v) for k, v in out.items()}


def edit_layer(model, layer, u, alpha):
    """W <- W - alpha * u u^T W on the two matrices of `layer` that write into the residual stream.
    Returns originals for restoring."""
    u = mx.array(u / np.linalg.norm(u), dtype=mx.float32)[:, None]
    mods = out_mats(model.model.layers[layer])
    orig = [mod.weight for mod in mods]
    for mod, o in zip(mods, orig):
        W = o.astype(mx.float32)  # shape (hidden, in): rows index the residual dimension
        mod.weight = (W - alpha * (u @ (u.T @ W))).astype(o.dtype)
    mx.eval([mod.weight for mod in mods])
    return orig


def restore_layer(model, layer, orig):
    for mod, o in zip(out_mats(model.model.layers[layer]), orig):
        mod.weight = o


def shift_dt_bias(model, delta):
    """Paper session 8: add `delta` (scalar or per-layer list of arrays) to every linear-attention dt_bias.
    Returns originals for restoring."""
    orig = []
    lin = [b for b in model.model.layers if getattr(b, "is_linear", False)]
    if not lin:
        raise ValueError("model has no linear-attention layers")
    for k, b in enumerate(lin):
        o = b.linear_attn.dt_bias
        d = delta[k] if isinstance(delta, (list, tuple)) else delta
        b.linear_attn.dt_bias = (o.astype(mx.float32) + mx.array(d, dtype=mx.float32)).astype(o.dtype)
        orig.append(o)
    mx.eval([b.linear_attn.dt_bias for b in lin])
    return orig


def restore_dt_bias(model, orig):
    lin = [b for b in model.model.layers if getattr(b, "is_linear", False)]
    for b, o in zip(lin, orig):
        b.linear_attn.dt_bias = o
