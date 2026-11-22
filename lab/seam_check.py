"""Does continuing an answer from a stored token prefix behave like uninterrupted generation?

Every answer of an arm (generated in one pass up to 4096 tokens) that is longer than CUT tokens is cut back to its
first CUT tokens and continued to 4096 exactly as ext.py does it (re-prefill, new seed, the arm's weights applied).
Original and re-continued answers share the same CUT-token prefix, so if the continuation is unbiased the two have
the same chance of finishing (and being correct) by 4096. Paired by problem.

  seam_check.py --model q35-4b --arms base,s8 [--cut 2048] [--bs 128]   -> out/seam_<model>_<arm>.pkl, out/seam_<model>.json
"""
import sys, os, glob, json, pickle, argparse
import numpy as np
import mlx.core as mx
from scipy.stats import binomtest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, HF, load_data, generate, grade
from mlx_lm import load
from mech import MODELS, dequantize_all
from ext import arm_edit, dump_atomic

CAP = 4096


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="q35-4b", choices=list(MODELS))
    ap.add_argument("--arms", default="base,s8")
    ap.add_argument("--cut", type=int, default=2048)
    ap.add_argument("--bs", type=int, default=128)
    a = ap.parse_args()
    tag = f"mech_{a.model}"
    S = json.load(open(f"{OUT}/{tag}_summary.json"))
    model, tok = load(glob.glob(f"{HF}/{MODELS[a.model]}/snapshots/*")[0])
    dequantize_all(model)
    mx.set_cache_limit(8 << 30)
    _, ev = load_data()
    gold = {p["unique_id"]: p["answer"] for p in ev}
    res = {}
    jp = f"{OUT}/seam_{a.model}.json"
    if os.path.exists(jp):
        res = json.load(open(jp))
    for arm in a.arms.split(","):
        R = pickle.load(open(f"{OUT}/{tag}_{S['arm_files'][arm]}.pkl", "rb"))
        idx = [i for i, r in enumerate(R) if r["ntok"] > a.cut]
        dst = f"{OUT}/seam_{a.model}_{S['arm_files'][arm]}_cut{a.cut}.pkl"
        if os.path.exists(dst):
            new = pickle.load(open(dst, "rb"))
        else:
            apply, restore = arm_edit(model, arm, S, None)
            new = {}
            for bi in range(0, len(idx), a.bs):
                pp = f"{dst[:-4]}.part{bi // a.bs}.pkl"
                b = idx[bi:bi + a.bs]
                if os.path.exists(pp):
                    part = pickle.load(open(pp, "rb")); assert set(part) == set(b); new.update(part); continue
                st = apply() if apply else None
                try:
                    _, ids = generate(model, tok, [list(R[i]["prompt_ids"]) + list(R[i]["out_ids"][:a.cut]) for i in b],
                                      CAP - a.cut, seed=5000 + a.cut + bi // a.bs, bs=a.bs)
                finally:
                    if restore:
                        restore(st)
                part = {}
                for i, extra in zip(b, ids):
                    out_ids = list(R[i]["out_ids"][:a.cut]) + list(extra)
                    g = grade(tok.decode(out_ids), len(out_ids), CAP, gold[R[i]["pid"]])
                    g.update(pid=R[i]["pid"], ntok=len(out_ids))
                    part[i] = g
                dump_atomic(part, pp); new.update(part); mx.clear_cache()
                print(f"[seam {a.model}] {arm}: batch {bi // a.bs + 1}/{-(-len(idx) // a.bs)} saved", flush=True)
            assert set(new) == set(idx)
            dump_atomic(new, dst)
        o = np.array([bool(R[i]["correct"]) for i in idx]); c = np.array([bool(new[i]["correct"]) for i in idx])
        of = np.array([not R[i]["truncated"] for i in idx]); cf = np.array([not new[i]["truncated"] for i in idx])
        w, l = int((c & ~o).sum()), int((o & ~c).sum())
        d = c.astype(float) - o
        rng = np.random.default_rng(0); m = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
        res[arm] = dict(cut=a.cut, n=len(idx), n_all=len(R), orig_correct=int(o.sum()), cont_correct=int(c.sum()),
                        orig_finished=int(of.sum()), cont_finished=int(cf.sum()),
                        cont_only=w, orig_only=l, p_sign=float(binomtest(w, w + l, 0.5).pvalue) if w + l else 1.0,
                        delta_within=float(d.mean()), ci=[float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))],
                        delta_on_all_400=float(d.sum() / len(R)),
                        orig_mean_ntok=float(np.mean([R[i]["ntok"] for i in idx])),
                        cont_mean_ntok=float(np.mean([new[i]["ntok"] for i in idx])))
        print(f"[seam {a.model}] {arm}: {json.dumps(res[arm])}", flush=True)
        json.dump(res, open(jp, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
