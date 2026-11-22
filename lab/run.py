"""Does a 'termination-direction' weight edit help only by reducing truncation?

1. Base on 100 practice problems -> finished vs truncated groups.
2. u_L = mean_h(truncated) - mean_h(finished) at layer L (response tokens).
3. Sweep (L, alpha) on 40 practice problems, pick best by accuracy.
4. Held-out 160 problems: base / edit / random-direction edit / concise prompt.
5. Counterfactual: let base's truncated answers continue +CAP tokens.
"""
import sys, pickle, json
sys.path.insert(0, ".")
from common import *

CAP = 4096
N_EVAL = 160
model, tok = load_model()
prac, ev = load_data()
ev = ev[:N_EVAL]
log = {}

print("== step 1: base on practice", flush=True)
pr = run_condition(model, tok, prac, CAP, 1, "practice_base")
pickle.dump(pr, open(f"{OUT}/practice_base.pkl", "wb"))

print("== step 2: directions", flush=True)
LAYERS = [6, 14, 20]
dirs = {}
for L in LAYERS:
    H = {True: [], False: []}
    for r in pr:
        H[r["truncated"]].append(hidden_mean(model, r["prompt_ids"], r["out_ids"], L))
    u = np.mean(H[True], 0) - np.mean(H[False], 0)
    dirs[L] = u / np.linalg.norm(u)
    print(f"  L={L}: n_trunc={len(H[True])} n_fin={len(H[False])} |diff|={np.linalg.norm(u):.3f}", flush=True)
np.savez(f"{OUT}/dirs.npz", **{str(k): v for k, v in dirs.items()})

print("== step 3: sweep on 40 practice problems", flush=True)
sub = prac[:40]
base_sub = summarize([r for r in pr if r["pid"] in {p["unique_id"] for p in sub}], "practice40_base")
sweep = []
for L, a in [(6, 1.0), (14, 1.0), (20, 1.0), (14, 0.5)]:
    orig = edit_layer(model, L, dirs[L], a)
    recs = run_condition(model, tok, sub, CAP, 1, f"sweep_L{L}_a{a}", seed0=100)
    restore_layer(model, L, orig)
    s = summarize(recs, f"sweep_L{L}_a{a}"); s.update(L=L, alpha=a); sweep.append(s)
best = max(sweep, key=lambda s: (s["acc"], -s["trunc"]))
L, A = best["L"], best["alpha"]
print(f"  chosen L={L} alpha={A}", flush=True)
log.update(sweep=sweep, base_sub=base_sub, chosen=dict(L=L, alpha=A))

print("== step 4: held-out eval", flush=True)
res = {}
recs = run_condition(model, tok, ev, CAP, 1, "eval_base", seed0=1000)
pickle.dump(recs, open(f"{OUT}/eval_base.pkl", "wb")); res["base"] = recs

orig = edit_layer(model, L, dirs[L], A)
res["edit"] = run_condition(model, tok, ev, CAP, 1, "eval_edit", seed0=1000)
restore_layer(model, L, orig)

rng = np.random.default_rng(7)
rnd = rng.standard_normal(dirs[L].shape[0]); rnd /= np.linalg.norm(rnd)
orig = edit_layer(model, L, rnd, A)
res["random"] = run_condition(model, tok, ev, CAP, 1, "eval_random", seed0=1000)
restore_layer(model, L, orig)

res["concise"] = run_condition(model, tok, ev, CAP, 1, "eval_concise", extra=CONCISE, seed0=1000)

print("== step 5: continue base's truncated answers by +CAP", flush=True)
tr = [r for r in res["base"] if r["truncated"]]
prompts = [list(r["prompt_ids"]) + list(r["out_ids"]) for r in tr]
texts, ids = generate(model, tok, prompts, CAP, seed=2000)
gold = {p["unique_id"]: p["answer"] for p in ev}
cont = []
for r, t, i in zip(tr, texts, ids):
    full = tok.decode(r["out_ids"]) + t
    g = grade(full, len(i), CAP, gold[r["pid"]])
    cont.append(dict(pid=r["pid"], truncated_again=g["truncated"], correct_if_extended=g["correct"]))
json.dump(cont, open(f"{OUT}/continuation.json", "w"), indent=1)
print(f"  {len(cont)} truncated base answers: finished with 2xCAP: {sum(not c['truncated_again'] for c in cont)}, "
      f"correct: {sum(c['correct_if_extended'] for c in cont)}", flush=True)

log["eval"] = {k: summarize(v, "eval_" + k) for k, v in res.items()}
log["continuation"] = dict(n=len(cont), finished=sum(not c["truncated_again"] for c in cont),
                           correct=sum(c["correct_if_extended"] for c in cont))
json.dump(log, open(f"{OUT}/summary.json", "w"), indent=1, default=float)
print("DONE", flush=True)
