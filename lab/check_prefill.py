"""Does re-prefilling a stored prefix in one pass give the same next-token distribution as the original
token-by-token decoding? (ext.py continues answers from a re-prefilled prefix.) Single sequence, bf16.

  check_prefill.py --model q35-4b [--n 6] [--k 700]
Prints, per sampled answer: max |dlogit|, KL(step || prefill) of the temperature-0.6 distributions, and whether
the top-20 candidate set (the sampler's top_k) is the same.
"""
import sys, os, glob, pickle, argparse, json
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, HF
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mech import MODELS, dequantize_all

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="q35-4b"); ap.add_argument("--n", type=int, default=6)
ap.add_argument("--k", type=int, default=700, help="output tokens replayed step by step")
a = ap.parse_args()
model, tok = load(glob.glob(f"{HF}/{MODELS[a.model]}/snapshots/*")[0]); dequantize_all(model)
mx.set_cache_limit(4 << 30)
R = [r for r in pickle.load(open(f"{OUT}/mech_{a.model}_eval_base.pkl", "rb")) if r["truncated"]]
rows = []
for r in R[:a.n]:
    p, o = list(r["prompt_ids"]), list(r["out_ids"])[:a.k]
    c1 = make_prompt_cache(model)
    lg1 = model(mx.array([p + o]), cache=c1)[0, -1].astype(mx.float32)          # one-pass prefill
    c2 = make_prompt_cache(model)
    lg = model(mx.array([p]), cache=c2)
    for t in o:                                                                 # original path: one token at a time
        lg = model(mx.array([[t]]), cache=c2); mx.eval(lg)
    lg2 = lg[0, -1].astype(mx.float32)
    a1, a2 = np.array(lg1), np.array(lg2)
    ls = lambda x: (x / 0.6) - np.log(np.exp(x / 0.6 - (x / 0.6).max()).sum()) - (x / 0.6).max()
    q1, q2 = ls(a1), ls(a2)
    kl = float((np.exp(q2) * (q2 - q1)).sum())
    t1, t2 = set(np.argsort(-a1)[:20].tolist()), set(np.argsort(-a2)[:20].tolist())
    row = dict(pid=r["pid"], n_ctx=len(p) + len(o), max_abs_dlogit=float(np.abs(a1 - a2).max()), kl=kl,
               same_top20=t1 == t2, same_argmax=int(a1.argmax()) == int(a2.argmax()),
               p_top_step=float(np.exp(q2).max()), p_top_prefill=float(np.exp(q1).max()))
    rows.append(row); print(row, flush=True)
json.dump(rows, open(f"{OUT}/check_prefill_{a.model}.json", "w"), indent=1)
print("median max|dlogit| %.4f, max KL %.2e, same top-20 in %d/%d" % (
    np.median([x["max_abs_dlogit"] for x in rows]), max(x["kl"] for x in rows), sum(x["same_top20"] for x in rows), len(rows)))
