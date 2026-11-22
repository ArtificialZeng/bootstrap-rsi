"""Extend finished mech.py arms to a larger token budget.

For every answer of an arm that was cut at the previous cap, generation is continued from its exact token
prefix up to the new cap WITH THE ARM'S WEIGHTS APPLIED, so the result is a sample of that arm at the new cap
(prefix and continuation both come from the same edited model; only the sampling seed changes at the join).
Answers that had already finished are kept as they are.

  ext.py --model q35-4b --arms base,s8,s8rand,concise --to 8192
  ext.py --model q35-4b --arms base,s8,s8rand,concise --to 16384      (needs the 8192 files)

Output: out/mech_<model>_<arm file>.cap<to>.pkl (same record format as mech.py), resumable per batch.
"""
import sys, os, glob, json, pickle, argparse, hashlib
import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (OUT, HF, load_data, generate, grade, edit_layer, restore_layer, shift_dt_bias, restore_dt_bias)
from mlx_lm import load
from mech import MODELS, dequantize_all
from agent8b import is_looping

BASE_CAP = 4096


def dump_atomic(v, path):
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(v, fh); fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp, path)


def check_recs(R, cap, what):
    for r in R:
        n = r["ntok"]
        assert n == len(r["out_ids"]) and 0 <= n <= cap, f"{what}: bad length for {r['pid']}"
        assert r["truncated"] == (n >= cap) and not (r["truncated"] and r["correct"]), f"{what}: bad flags for {r['pid']}"


def arm_edit(model, arm, S, dirs):
    """(apply, restore) for an arm spec as used by mech.py; (None, None) if the arm leaves the weights alone."""
    spec = arm.split("@")[0]
    kind, _, param = spec.partition(":")
    if kind in ("base", "concise", "reseed"):
        return None, None
    if kind in ("s8", "s8concise"):
        d = float(param) if param else 0.5
        return (lambda: shift_dt_bias(model, d)), (lambda o: restore_dt_bias(model, o))
    if kind.startswith("s8rand"):
        rid = int(kind[6:]) if kind[6:].isdigit() else 0
        d = float(param) if param else 0.5
        rng = np.random.default_rng(8 + rid)
        lin = [b for b in model.model.layers if getattr(b, "is_linear", False)]
        deltas = [d * rng.choice([-1.0, 1.0], size=b.linear_attn.dt_bias.shape) for b in lin]
        return (lambda: shift_dt_bias(model, deltas)), (lambda o: restore_dt_bias(model, o))
    L, A = S["chosen"]["L"], S["chosen"]["alpha"]
    al = float(param) if param else A
    if kind == "edit":
        return (lambda: edit_layer(model, L, dirs[L], al)), (lambda o: restore_layer(model, L, o))
    if kind.startswith("random"):
        rid = int(kind[6:]) if kind[6:].isdigit() else 0
        rnd = np.random.default_rng(7 + rid).standard_normal(dirs[L].shape[0]); rnd /= np.linalg.norm(rnd)
        return (lambda: edit_layer(model, L, rnd, al)), (lambda o: restore_layer(model, L, o))
    sys.exit(f"unknown arm {arm}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--arms", required=True)
    ap.add_argument("--to", type=int, required=True, help="new cap; the source is the file at half this cap")
    ap.add_argument("--bs", type=int, default=None)
    ap.add_argument("--dirs", default=None)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: continue only N answers, write *.test files")
    a = ap.parse_args()
    prev = {8192: 4096, 16384: 8192}.get(a.to)
    if prev is None:
        ap.error("--to must be 8192 or 16384")
    if a.limit is not None and a.limit <= 0:
        ap.error("--limit must be positive")
    bs = a.bs or 32
    tag = f"mech_{a.model}"
    sp = f"{OUT}/{tag}_summary.json"
    # arm -> cache file; the default names are fixed by mech.py, so the summary is only needed for edit/random
    S = json.load(open(sp)) if os.path.exists(sp) else {"arm_files": {}}
    for k in ("base", "s8", "s8rand", "concise", "reseed"):
        S["arm_files"].setdefault(k, f"eval_{k}")
    dirs = None
    if a.dirs:
        z = np.load(a.dirs); dirs = {int(k): z[k] for k in z.files}

    model, tok = load(glob.glob(f"{HF}/{MODELS[a.model]}/snapshots/*")[0])
    dequantize_all(model)
    mx.set_cache_limit(8 << 30)
    _, ev = load_data()
    gold = {p["unique_id"]: p["answer"] for p in ev}
    sfx = lambda cap: "" if cap == BASE_CAP else f".cap{cap}"

    for arm in a.arms.split(","):
        f = S["arm_files"][arm]
        src, dst = f"{OUT}/{tag}_{f}{sfx(prev)}.pkl", f"{OUT}/{tag}_{f}{sfx(a.to)}.pkl"
        if os.path.exists(dst):
            print(f"[ext {a.model}] {arm} cap {a.to}: cached", flush=True); continue
        R = pickle.load(open(src, "rb"))
        check_recs(R, prev, f"{arm} source")
        assert len({r["pid"] for r in R}) == len(R) == len(ev)
        todo = [i for i, r in enumerate(R) if r["truncated"]]
        if a.limit:
            todo, dst = todo[:a.limit], dst[:-4] + f".test{a.limit}.pkl"
        # batches written by another batch size / source file / limit are refused, not silently merged
        meta = dict(src_sha=hashlib.sha256(open(src, "rb").read()).hexdigest(), to=a.to, bs=bs, limit=a.limit,
                    mode="continued from the stored token prefix (re-prefilled), seed 3000+cap+batch")
        bad = sum(tok.decode(list(R[i]["out_ids"])) != R[i]["text"] for i in todo[:20])
        if bad:
            print(f"[ext {a.model}] {arm}: {bad}/20 stored texts differ from decode(out_ids)", flush=True)
        apply, restore = arm_edit(model, arm, S, dirs)
        new = {}
        for bi in range(0, len(todo), bs):
            pp = f"{dst[:-4]}.part{bi // bs}.pkl"
            idx = todo[bi:bi + bs]
            if os.path.exists(pp):
                pm, part = pickle.load(open(pp, "rb"))
                assert pm == meta and set(part) == set(idx), f"stale batch file {pp}: move it away"
                new.update(part); continue
            st = apply() if apply else None
            try:
                _, ids = generate(model, tok, [list(R[i]["prompt_ids"]) + list(R[i]["out_ids"]) for i in idx],
                                  a.to - prev, seed=3000 + a.to + bi // bs, bs=bs)
            finally:
                if restore:
                    restore(st)
            part = {}
            assert len(ids) == len(idx)
            for i, extra in zip(idx, ids):
                r = R[i]
                out_ids = list(r["out_ids"]) + list(extra)
                text = tok.decode(out_ids)
                g = grade(text, len(out_ids), a.to, gold[r["pid"]])
                g["looping"] = is_looping(text, g["truncated"])
                g.update(pid=r["pid"], level=r["level"], text=text, prompt_ids=r["prompt_ids"], out_ids=out_ids)
                part[i] = g
            dump_atomic((meta, part), pp); new.update(part)
            mx.clear_cache()
            print(f"[ext {a.model}] {arm} cap {a.to}: batch {bi // bs + 1}/{-(-len(todo) // bs)} saved", flush=True)
        assert set(new) == set(todo)
        out = [new.get(i, r) for i, r in enumerate(R)]
        if not a.limit:
            check_recs(out, a.to, f"{arm} result")
        dump_atomic(out, dst)
        json.dump(meta, open(dst[:-4] + ".meta.json", "w"), indent=1)
        acc = np.mean([r["correct"] for r in out]); tr = np.mean([r["truncated"] for r in out])
        print(f"[ext {a.model}] {arm} cap {a.to}: acc {acc:.3f} trunc {tr:.3f} (continued {len(todo)})", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
