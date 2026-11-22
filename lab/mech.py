"""Mechanism experiment, any Qwen3 / Qwen3.5 size: does a weight edit help only by cutting truncation?

Arms (all on the same held-out problems, one answer each, same sampling seed as base):
  edit     termination-direction ablation W <- W - a u u^T W on the two residual-writing matrices of layer L,
           u = mean_h(truncated) - mean_h(finished) on practice answers; L, a from a 40-problem sweep or --fixed
  random   same layer and strength, random unit direction
  s8       the paper's session-8 edit (Qwen3.5 only): every linear-attention dt_bias += 0.5
  s8rand   same magnitude, random sign per element (Qwen3.5 only)
  concise  unedited model, prompt asks for efficient thinking
  reseed   unedited model, another seed (noise floor)
  + cont   base answers that were truncated, continued for another 4096 tokens (same sample)
Cached per model under out/mech_<model>_*.pkl, with a config sidecar that refuses mismatched reuse.
"""
import sys, os, json, glob, pickle, argparse
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (load_data, build_prompts, generate, grade, summarize, last_boxed, edit_layer, restore_layer,
                    shift_dt_bias, restore_dt_bias, hidden_states_multi, CONCISE, OUT, HF)
from agent8b import is_looping, paired_stats

MODELS = {"1.7b": "models--Qwen--Qwen3-1.7B", "4b": "models--Qwen--Qwen3-4B",
          "8b": "models--mlx-community--Qwen3-8B-8bit", "q35-4b": "models--mlx-community--Qwen3.5-4B-MLX-bf16"}
DEFAULT_ARMS = {"q35-4b": "s8,s8rand,concise,reseed"}
DEFAULT_BS = {"q35-4b": 128}  # only 8 of 32 layers keep a KV cache (handoff/codex/T8_throughput_out.md)
CAP = 4096
BS = {"n": 64}  # completion batch size; fixed per model so every arm is generated the same way


def dequantize_all(model):
    """Every QuantizedLinear inside the decoder blocks -> bf16 nn.Linear (same kernels for every arm)."""
    n = 0
    for blk in model.model.layers:
        for sub in ("self_attn", "mlp", "linear_attn"):
            if sub not in blk:
                continue
            par = blk[sub]
            for a, m in list(par.items()):
                if isinstance(m, nn.QuantizedLinear):
                    w = mx.dequantize(m.weight, m.scales, m.biases, group_size=m.group_size, bits=m.bits)
                    lin = nn.Linear(w.shape[1], w.shape[0], bias=False)
                    lin.weight = w.astype(mx.bfloat16)
                    mx.eval(lin.weight)
                    setattr(par, a, lin)
                    n += 1
    return n


def gen_grade(model, tok, probs, seed, extra=""):
    prompts = build_prompts(tok, probs, extra)
    texts, ids = generate(model, tok, prompts, CAP, seed=seed, bs=BS["n"])
    recs = []
    for p, pr, t, i in zip(probs, prompts, texts, ids):
        g = grade(t, len(i), CAP, p["answer"])
        g["looping"] = is_looping(t, g["truncated"])
        g.update(pid=p["unique_id"], level=p["level"], text=t, prompt_ids=pr, out_ids=i)
        recs.append(g)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--n_eval", type=int, default=400)
    ap.add_argument("--arms", default=None, help="comma list from edit,random,s8,s8rand,concise,reseed")
    ap.add_argument("--fixed", default=None, help="L,alpha for the edit arm: skip the sweep")
    ap.add_argument("--dirs", default=None, help="npz of precomputed directions keyed by layer")
    ap.add_argument("--practice", default=None, help="pkl of precomputed practice base records")
    ap.add_argument("--bs", type=int, default=None, help="completion batch size (same for all arms of a model)")
    ap.add_argument("--no-cont", action="store_true", help="skip the base-only 2x continuation (ext.py supersedes it)")
    a = ap.parse_args()
    if a.bs is None:
        a.bs = DEFAULT_BS.get(a.model, 64)
    BS["n"] = a.bs
    arms = (a.arms or DEFAULT_ARMS.get(a.model, "edit,random,concise,reseed")).split(",")

    path = glob.glob(f"{HF}/{MODELS[a.model]}/snapshots/*")[0]
    model, tok = load(path)
    nq = dequantize_all(model)
    nL = len(model.model.layers)
    print(f"model {a.model}: {nL} layers, dequantized {nq}, arms {arms}", flush=True)
    tag = f"mech_{a.model}"
    P = lambda n: f"{OUT}/{tag}_{n}.pkl"

    cfg = dict(n_eval=a.n_eval, cap=CAP, fixed=a.fixed, dirs=a.dirs, practice=a.practice, bs=a.bs)
    cp = f"{OUT}/{tag}_config.json"
    if os.path.exists(cp):
        old = json.load(open(cp))
        diff = {k: (old.get(k), v) for k, v in cfg.items() if old.get(k) != v}
        if diff:
            sys.exit(f"cached results under {tag} were made with a different config: {diff}. Move them away first.")
    json.dump(cfg, open(cp, "w"), indent=1)

    def cached(n, fn):
        if os.path.exists(P(n)):
            return pickle.load(open(P(n), "rb"))
        v = fn(); pickle.dump(v, open(P(n), "wb")); return v

    prac, ev = load_data()
    practice, ev = prac[:100], ev[:a.n_eval]
    S = {"model": a.model, "cap": CAP, "arms": arms, "arm_files": {}}
    sp = f"{OUT}/{tag}_summary.json"
    save = lambda: json.dump(S, open(sp, "w"), indent=1, default=float)

    if "edit" in arms or "random" in arms:
        if a.practice:
            base_prac = pickle.load(open(a.practice, "rb"))
        else:
            base_prac = cached("practice_base", lambda: gen_grade(model, tok, practice, 100))
        S["practice_base"] = summarize(base_prac, f"{tag}_practice_base")
        if a.dirs:
            dirs = {int(k): v for k, v in np.load(a.dirs).items()}
        else:
            def _dirs():
                layers = sorted({int(round(f * (nL - 1))) for f in (0.22, 0.52, 0.74)})
                H = {L: {True: [], False: []} for L in layers}
                for r in base_prac:
                    h = hidden_states_multi(model, list(r["prompt_ids"]) + list(r["out_ids"]), len(r["prompt_ids"]), layers)
                    for L in layers:
                        H[L][r["truncated"]].append(h[L])
                n_t = len(H[layers[0]][True]); n_f = len(H[layers[0]][False])
                if n_t < 5 or n_f < 5:
                    sys.exit(f"cannot build a truncated-vs-finished direction: {n_t} truncated, {n_f} finished")
                out = {}
                for L in layers:
                    u = np.mean(H[L][True], 0) - np.mean(H[L][False], 0)
                    out[L] = u / np.linalg.norm(u)
                return out
            dirs = cached("dirs", _dirs)
        print("direction layers:", sorted(dirs), flush=True)
        if a.fixed:
            L, A = a.fixed.split(","); L, A = int(L), float(A)
            S["chosen"] = dict(L=L, alpha=A, how="fixed")
        else:
            sub = practice[:40]
            subids = {p["unique_id"] for p in sub}
            S["sweep_base"] = summarize([r for r in base_prac if r["pid"] in subids], f"{tag}_sweep_base")
            Ls = sorted(dirs)
            sweep = []
            for L, A in [(Ls[0], 1.0), (Ls[1], 1.0), (Ls[2], 1.0), (Ls[1], 0.5)]:
                def _run(L=L, A=A):
                    orig = edit_layer(model, L, dirs[L], A)
                    try:
                        return gen_grade(model, tok, sub, 100)
                    finally:
                        restore_layer(model, L, orig)
                s = summarize(cached(f"sweep_L{L}_a{A}", _run), f"{tag}_sweep_L{L}_a{A}")
                s.update(L=L, alpha=A); sweep.append(s)
            best = max(sweep, key=lambda s: (s["acc"], -s["trunc"]))
            L, A = best["L"], best["alpha"]
            S["sweep"], S["chosen"] = sweep, dict(L=L, alpha=A, how="sweep on 40 practice problems")
        print("chosen", S["chosen"], flush=True); save()

    CHUNK = 200  # held-out set is generated and saved in chunks; chunk i uses seed + i in every arm

    def eval_arm(name, seed, extra="", apply=None, restore=None):
        if os.path.exists(P(name)):
            return pickle.load(open(P(name), "rb"))
        parts = []
        for ci in range(0, len(ev), CHUNK):
            pp = P(f"{name}.part{ci // CHUNK}")
            if os.path.exists(pp):
                parts.append(pickle.load(open(pp, "rb"))); continue
            st = apply() if apply else None
            try:
                recs = gen_grade(model, tok, ev[ci:ci + CHUNK], seed + ci // CHUNK, extra)
            finally:
                if restore:
                    restore(st)
            pickle.dump(recs, open(pp, "wb")); parts.append(recs)
            print(f"[{tag}] {name} chunk {ci // CHUNK} saved", flush=True)
        v = [r for part in parts for r in part]
        pickle.dump(v, open(P(name), "wb"))
        return v

    R = {"base": eval_arm("eval_base", 1000)}
    S["arm_files"]["base"] = "eval_base"

    def run_arm(arm):
        """arm spec: kind[:param][@seed]. kind in base, edit, random[K], s8, s8rand[K], concise, reseed.
        param: alpha for edit/random, dt_bias shift for s8/s8rand. Default spec names keep the original cache names."""
        spec, seed = (arm.split("@") + [None])[:2]
        seed = int(seed) if seed else 1000
        kind, _, param = spec.partition(":")
        sfx = (f"_p{param}" if param else "") + (f"_seed{seed}" if seed != 1000 else "")
        rid = int(kind[6:]) if kind.startswith("random") and kind[6:].isdigit() else 0
        if kind.startswith("s8rand") and kind[6:].isdigit():
            rid = int(kind[6:])
        apply = restore = None
        extra = ""
        if kind == "base":
            name = f"eval_base{sfx}"
        elif kind == "edit":
            al = float(param) if param else A
            name = f"eval_edit_L{L}_a{A}" if not sfx else f"eval_edit_L{L}{sfx}"
            apply, restore = (lambda: edit_layer(model, L, dirs[L], al)), (lambda o: restore_layer(model, L, o))
        elif kind.startswith("random"):
            al = float(param) if param else A
            rnd = np.random.default_rng(7 + rid).standard_normal(dirs[L].shape[0]); rnd /= np.linalg.norm(rnd)
            name = (f"eval_random_L{L}_a{A}" if not sfx and rid == 0 else f"eval_random{rid}_L{L}{sfx}")
            apply, restore = (lambda: edit_layer(model, L, rnd, al)), (lambda o: restore_layer(model, L, o))
        elif kind in ("s8", "s8concise"):  # s8concise: the edit and the concise prompt together
            d = float(param) if param else 0.5
            name = f"eval_{kind}{sfx}"
            extra = CONCISE if kind == "s8concise" else ""
            apply, restore = (lambda: shift_dt_bias(model, d)), (lambda o: restore_dt_bias(model, o))
        elif kind.startswith("s8rand"):
            d = float(param) if param else 0.5
            rng = np.random.default_rng(8 + rid)
            lin = [b for b in model.model.layers if getattr(b, "is_linear", False)]
            deltas = [d * rng.choice([-1.0, 1.0], size=b.linear_attn.dt_bias.shape) for b in lin]
            name = f"eval_s8rand{sfx}" if rid == 0 else f"eval_s8rand{rid}{sfx}"
            apply, restore = (lambda: shift_dt_bias(model, deltas)), (lambda o: restore_dt_bias(model, o))
        elif kind == "concise":
            name, extra = f"eval_concise{sfx}", CONCISE
        elif kind == "reseed":
            name, seed = "eval_reseed", 2000
        else:
            sys.exit(f"unknown arm {arm}")
        return name, seed, eval_arm(name, seed, extra=extra, apply=apply, restore=restore)

    for arm in arms:
        name, seed, R[arm] = run_arm(arm)
        S["arm_files"][arm] = name
        S.setdefault("arm_seeds", {})[arm] = seed
        print(f"[{tag}] arm {arm} done", flush=True)

    S["n_eval"] = len(R["base"])
    S["eval"] = {k: summarize(v, f"{tag}_eval_{k}") for k, v in R.items()}
    # each arm is paired with the base run that used the same seed (base@<seed> if it was generated)
    def base_for(k):
        sd = S.get("arm_seeds", {}).get(k, 1000)
        return R.get(f"base@{sd}", R["base"]) if sd not in (1000, 2000) else R["base"]
    S["paired_base"] = {k: ("base" if base_for(k) is R["base"] else f"base@{S['arm_seeds'][k]}") for k in R if k != "base"}
    S["paired_vs_base"] = {k: paired_stats(base_for(k), v) for k, v in R.items() if k != "base"}
    save()
    if a.no_cont:
        print("DONE (no continuation)", flush=True); return

    def _cont():
        tr = [r for r in R["base"] if r["truncated"]]
        # contexts here are ~2x as long as in the arms; a full batch ran out of GPU memory on 1.7B (bs 64)
        texts, ids = generate(model, tok, [list(r["prompt_ids"]) + list(r["out_ids"]) for r in tr], CAP, seed=3000,
                              bs=32 if a.model == "q35-4b" else 16)
        gold = {p["unique_id"]: p["answer"] for p in ev}
        out = []
        for r, t, i in zip(tr, texts, ids):
            g = grade(tok.decode(r["out_ids"]) + t, len(i), CAP, gold[r["pid"]])
            out.append(dict(pid=r["pid"], finished=not g["truncated"], correct=g["correct"], extra_tokens=len(i),
                            looping_at_cap=r["looping"], new_box_in_cont=last_boxed(t) is not None))
        return out
    cont = cached("cont", _cont)
    n_tr = sum(r["truncated"] for r in R["base"])
    assert len(cont) == n_tr, "continuation cache does not match the base answers"
    S["continuation"] = dict(n_truncated=n_tr, finished_with_2x=sum(c["finished"] for c in cont),
                             correct_with_2x=sum(c["correct"] for c in cont),
                             correct_with_new_box=sum(c["correct"] and c.get("new_box_in_cont", True) for c in cont),
                             looping_at_cap=sum(c["looping_at_cap"] for c in cont),
                             correct_among_looping=sum(c["correct"] for c in cont if c["looping_at_cap"]))
    S["continuation"]["base_acc_with_2x_budget"] = \
        (sum(r["correct"] for r in R["base"]) + S["continuation"]["correct_with_2x"]) / S["n_eval"]
    print("continuation", S["continuation"], flush=True)
    save()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
