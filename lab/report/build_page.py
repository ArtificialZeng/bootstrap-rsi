"""Build report/index.html from the stored results (CPU only, no model code).

  ../.venv/bin/python -B cap_table.py --caps 4096,5120,...,16384 --json report/curve.json --md report/curve.md
  ../.venv/bin/python -B report/build_page.py
"""
import os, json, pickle, collections, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out")
F = {"base": "eval_base", "reseed": "eval_reseed", "s8": "eval_s8", "s8@2000": "eval_s8_seed2000",
     "s8rand": "eval_s8rand", "concise": "eval_concise", "concise@2000": "eval_concise_seed2000",
     "s8concise": "eval_s8concise"}


def recs(arm, cap=16384):
    p = f"{OUT}/mech_q35-4b_{F[arm]}.cap{cap}.pkl"
    return {r["pid"]: r for r in pickle.load(open(p, "rb"))} if os.path.exists(p) else None


def state(r):
    return "T" if r["truncated"] else ("C" if r["correct"] else "W")


def transitions(a, b):
    """Counts of base-state -> arm-state over problems (T cut at the cap, W finished wrong, C correct)."""
    M = collections.Counter(state(b[p]) + state(a[p]) for p in b)
    return {k: M.get(k, 0) for k in ("TC", "CT", "WC", "CW", "TW", "WT")}


def main():
    curve = json.load(open(f"{HERE}/curve.json"))
    caps = sorted(int(c) for c in curve)
    D = {"caps": caps, "arms": {}, "pairs": {}}
    for c in caps:
        for k, v in curve[str(c)]["arms"].items():
            D["arms"].setdefault(k, {})[c] = dict(acc=v["acc"], trunc=v["trunc"], fw=v["fin_wrong"], tpc=v["tok_per_correct"])
        for k, v in curve[str(c)]["pairs"].items():
            D["pairs"].setdefault(k, {})[c] = dict(d=v["delta"], lo=v["ci"][0], hi=v["ci"][1], p=v["p"],
                                                    w=v.get("wins", v.get("pos")), l=v.get("losses", v.get("neg")))
    R = {k: recs(k) for k in F}
    tr = {}
    for name, a, b in (("s8 seed 1000", "s8", "base"), ("s8 seed 2000", "s8@2000", "reseed"), ("concise", "concise", "base"),
                       ("s8rand", "s8rand", "base")):
        if R.get(a) and R.get(b):
            tr[name] = transitions(R[a], R[b])
    D["transitions"] = tr
    lv = {}
    for k in ("base", "s8", "concise"):
        if R.get(k):
            n5 = sum(r["level"] == 5 for r in R[k].values())
            lv[k] = dict(l5=sum(r["truncated"] and r["level"] == 5 for r in R[k].values()), n5=n5,
                         rest=sum(r["truncated"] and r["level"] != 5 for r in R[k].values()), nrest=len(R[k]) - n5,
                         loop=sum(bool(r.get("looping")) for r in R[k].values() if r["truncated"]),
                         trunc=sum(r["truncated"] for r in R[k].values()),
                         med=sorted(r["ntok"] for r in R[k].values() if not r["truncated"])[
                             sum(not r["truncated"] for r in R[k].values()) // 2])
    D["levels"] = lv
    small = json.load(open(f"{OUT}/results.json")).get("1.7b", {})
    D["small"] = {k: dict(acc=v["edit_acc"], d=v["delta"], lo=v["ci95"][0], hi=v["ci95"][1], w=v["wins"], l=v["losses"])
                  for k, v in small.get("conditions", {}).items()}
    D["built"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    html = open(f"{HERE}/template.html").read().replace("__DATA__", json.dumps(D, separators=(",", ":")))
    open(f"{HERE}/index.html", "w").write(html)
    print("wrote", f"{HERE}/index.html", len(html), "bytes; arms", list(D["arms"]), "transitions", tr, "levels", lv, "small", D["small"])


if __name__ == "__main__":
    main()
