"""Pre-specified analysis of the recursive chain (out/prereg_chain.md). CPU only; reads the stored answers.

  chain_stats.py [--key 8b_c4096_p60_e400]   -> out/<key>_chain_stats.json and a printed table

Primary test: last generation that kept an edit vs generation 0 on the held-out problems, seeds 1000 and 2000 for
both, d_i = mean over seeds of (last - gen0), paired t-test over problems. Everything else is exploratory.
"""
import os, json, pickle, argparse, collections
import numpy as np
from cap_table import pooled, paired, boot

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def load(key, name):
    p = f"{OUT}/{key}_{name}.pkl"
    if not os.path.exists(p):
        return None
    R = pickle.load(open(p, "rb"))
    D = {r["pid"]: dict(c=bool(r["correct"]), t=bool(r["truncated"]), w=(not r["correct"]) and not r["truncated"],
                        n=r["ntok"], loop=bool(r.get("looping"))) for r in R}
    assert len(D) == len(R), f"duplicate problem ids in {p}"
    return D


def arm(D):
    x = list(D.values())
    return dict(n=len(x), acc=float(np.mean([r["c"] for r in x])), trunc=float(np.mean([r["t"] for r in x])),
                fin_wrong=int(sum(r["w"] for r in x)), looping=float(np.mean([r["loop"] for r in x])),
                med_len_finished=float(np.median([r["n"] for r in x if not r["t"]])) if any(not r["t"] for r in x) else None)


def trans(A, B):
    st = lambda r: "T" if r["t"] else ("C" if r["c"] else "W")
    M = collections.Counter(st(B[p]) + st(A[p]) for p in B)  # reference state, then arm state
    return {a + b: M.get(a + b, 0) for a in "TWC" for b in "TWC"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="8b_c4096_p60_e400")
    ap.add_argument("--chain-name", default="", help="suffix of a further chain (agent8b.py --chain-name)")
    a = ap.parse_args()
    cn = "chain" + a.chain_name
    res = json.load(open(f"{OUT}/{a.key}_results.json"))
    ch = res.get(cn, {})
    gens = sorted(int(g) for g in ch.get("generations", {}))
    D = {"gen0@1000": load(a.key, "eval_base"), "gen0@2000": load(a.key, "eval_reseed")}
    # a generation whose held-out answers exist but whose record is not written yet (run stopped in between)
    g = (gens[-1] if gens else 0) + 1
    while os.path.exists(f"{OUT}/{a.key}_{cn}_gen{g}_eval_s1000.pkl"):
        gens.append(g); g += 1
    for g in gens:
        D[f"gen{g}@1000"] = load(a.key, f"{cn}_gen{g}_eval_s1000")
    last = gens[-1] if gens else 0
    if last:
        D[f"gen{last}@2000"] = load(a.key, f"{cn}_gen{last}_eval_s2000")
    D = {k: v for k, v in D.items() if v}
    o = dict(key=a.key, chain=cn, generations=gens, stopped=ch.get("stopped"), arms={k: arm(v) for k, v in D.items()},
             attempts={k: dict(decision=v["decision"], edits=v.get("edits"), check=v.get("check"))
                       for k, v in ch.get("attempts", {}).items()}, exploratory={})
    if "gen0@1000" in D and "gen0@2000" in D:
        o["exploratory"]["noise floor: gen0@2000 - gen0@1000"] = paired(D["gen0@2000"], D["gen0@1000"])
    for g in gens:
        k, prev = f"gen{g}@1000", f"gen{g-1}@1000"
        if k in D:
            o["exploratory"][f"gen{g} - gen0 (seed 1000)"] = dict(paired(D[k], D["gen0@1000"]), transitions=trans(D[k], D["gen0@1000"]))
            if g > 1 and prev in D:
                o["exploratory"][f"gen{g} - gen{g-1} (seed 1000)"] = dict(paired(D[k], D[prev]), transitions=trans(D[k], D[prev]))
    need = [f"gen{last}@1000", f"gen{last}@2000", "gen0@1000", "gen0@2000"]
    if last and all(k in D for k in need):
        o["primary"] = dict(pooled(D, (need[0], need[1]), (need[2], need[3])), generation=last,
                            seed1000=paired(D[need[0]], D[need[2]]), seed2000=paired(D[need[1]], D[need[3]]),
                            transitions_seed2000=trans(D[need[1]], D[need[3]]))
    elif not gens:
        o["primary"] = "the chain did not start: no generation kept an edit" if ch.get("stopped") else "no generation finished yet"
    else:
        o["primary"] = "pending: " + ", ".join(k for k in need if k not in D)
    json.dump(o, open(f"{OUT}/{a.key}_{cn}_stats.json", "w"), indent=1, default=str)
    for k, v in o["arms"].items():
        print(f"{k:12s} acc {100*v['acc']:.2f}  truncated {100*v['trunc']:.2f}  finished wrong {v['fin_wrong']}  looping {100*v['looping']:.1f}")
    for k, v in o["exploratory"].items():
        print(f"{k}: {100*v['delta']:+.2f} [{100*v['ci'][0]:+.2f}, {100*v['ci'][1]:+.2f}] {v['wins']}/{v['losses']} p={v['p']:.3g}", v.get("transitions", ""))
    print("PRIMARY:", json.dumps(o["primary"], default=str)[:900])


if __name__ == "__main__":
    main()
