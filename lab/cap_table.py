"""Paired comparisons of Qwen3.5-4B arms at several caps (CPU only; reads the stored answers).

  cap_table.py [--caps 4096,5120,6144,7168,8192] [--md out/q35_caps.md]   -> out/q35_caps.json, a markdown table

Arms are generated at 4096 and continued to 8192 / 16384 by ext.py. At a cap below a file's cap the answers are
re-cut offline (an answer counts only if it finished within the cap). Each cap uses the smallest file that covers it.

Statistics (per problem, n = 400):
  single-run pairs   exact sign test over discordant problems, bootstrap CI over problems
  two-seed rows      d_i = mean over seeds of (arm - reference); paired t-test with its CI (the pre-specified
                     statistic, out/prereg_16384.md), plus the exact sign-flip p and the bootstrap CI as checks
"""
import os, sys, json, pickle, argparse
import numpy as np
from scipy.stats import binomtest, ttest_1samp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")  # no common import: keeps this script off MLX

ARMS = {"base": "eval_base", "reseed": "eval_reseed", "s8": "eval_s8", "s8@2000": "eval_s8_seed2000",
        "s8rand": "eval_s8rand", "s8rand1": "eval_s8rand1", "concise": "eval_concise", "concise@2000": "eval_concise_seed2000", "s8concise": "eval_s8concise"}
LABEL = {"base": "base (seed 1000)", "reseed": "base (seed 2000)", "s8": "s8 edit (seed 1000)",
         "s8@2000": "s8 edit (seed 2000)", "s8rand": "random-sign control", "s8rand1": "random-sign control 2",
         "concise": "concise prompt (seed 1000)", "concise@2000": "concise prompt (seed 2000)", "s8concise": "s8 edit + concise prompt"}
PAIRS = [("s8", "base"), ("s8@2000", "reseed"), ("reseed", "base"), ("s8rand", "base"), ("s8rand1", "base"),
         ("s8", "s8rand"), ("s8", "s8rand1"), ("concise", "base"), ("concise", "s8"), ("concise", "reseed"),
         ("concise", "s8@2000"), ("concise@2000", "reseed"), ("concise@2000", "s8@2000"), ("s8concise", "base"), ("s8concise", "s8"), ("s8concise", "concise")]
# name -> (arms averaged, reference arms averaged)
POOLED = {"s8 (2 seeds) - base (2 seeds)": (("s8", "s8@2000"), ("base", "reseed")),
          "concise - base (2 seeds)": (("concise",), ("base", "reseed")),
          "concise - s8 (2 seeds)": (("concise",), ("s8", "s8@2000")),
          "s8concise - s8 (2 seeds)": (("s8concise",), ("s8", "s8@2000")),
          "concise (2 seeds) - base (2 seeds)": (("concise", "concise@2000"), ("base", "reseed")),
          "s8 (2 seeds) - concise (2 seeds)": (("s8", "s8@2000"), ("concise", "concise@2000"))}
FILE_CAPS = (4096, 8192, 16384)


def load(arm, cap, m="q35-4b"):
    for c in (c for c in FILE_CAPS if c >= cap):
        p = f"{OUT}/mech_{m}_{ARMS[arm]}{'' if c == 4096 else f'.cap{c}'}.pkl"
        if os.path.exists(p):
            R = pickle.load(open(p, "rb"))
            D = {r["pid"]: dict(c=bool(r["correct"]) and r["ntok"] < cap, t=r["ntok"] >= cap,
                                n=min(r["ntok"], cap), w=(not r["correct"]) and r["ntok"] < cap) for r in R}
            assert len(D) == len(R), f"duplicate problem ids in {p}"
            return D
    return None


def boot(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def flip_p(d, unit):
    """Exact two-sided sign-flip p for differences that are multiples of `unit` (dynamic programme over sums)."""
    w = [abs(int(round(x / unit))) for x in d if x]
    counts = np.array([1.0])
    for k in w:
        new = np.zeros(len(counts) + k); new[:len(counts)] += counts; new[k:] += counts; counts = new
    tot, obs = sum(w), abs(int(round(d.sum() / unit)))
    s = np.arange(len(counts))
    return float(counts[np.abs(2 * s - tot) >= obs].sum() / 2.0 ** len(w)) if w else 1.0


def paired(A, B):
    assert set(A) == set(B), "arms cover different problems"
    pids = sorted(A)
    d = np.array([int(A[p]["c"]) - int(B[p]["c"]) for p in pids], float)
    w, l = int((d > 0).sum()), int((d < 0).sum())
    return dict(delta=float(d.mean()), ci=boot(d), wins=w, losses=l,
                p=float(binomtest(w, w + l, 0.5).pvalue) if w + l else 1.0,
                wc=sum(B[p]["w"] and A[p]["c"] for p in pids), cw=sum(B[p]["c"] and A[p]["w"] for p in pids))


def pooled(D, xs, ys):
    pids = sorted(D[xs[0]])
    assert all(set(D[k]) == set(pids) for k in xs + ys)
    m = lambda ks: np.array([np.mean([D[k][p]["c"] for k in ks]) for p in pids])
    d = m(xs) - m(ys)
    t = ttest_1samp(d, 0); ci = t.confidence_interval()
    unit = 1.0 / (len(xs) * len(ys))
    return dict(delta=float(d.mean()), ci=(float(ci.low), float(ci.high)), p=float(t.pvalue), ci_boot=boot(d),
                p_flip=flip_p(d, unit), pos=int((d > 0).sum()), neg=int((d < 0).sum()), pooled=True)


def fmt(s):
    wl = f"+{s['pos']}/-{s['neg']}" if s.get("pooled") else f"{s['wins']}/{s['losses']}"
    return f"{100*s['delta']:+.1f} [{100*s['ci'][0]:+.1f}, {100*s['ci'][1]:+.1f}], {wl}, p={s['p']:.2g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caps", default="4096,5120,6144,7168,8192")
    ap.add_argument("--md", default=f"{OUT}/q35_caps.md")
    ap.add_argument("--json", default=f"{OUT}/q35_caps.json")
    a = ap.parse_args()
    caps = [int(c) for c in a.caps.split(",")]
    res = {}
    for cap in caps:
        D = {k: v for k, v in ((k, load(k, cap)) for k in ARMS) if v}
        o = res[cap] = dict(arms={}, pairs={})
        for k, v in D.items():
            x = list(v.values()); nc = sum(r["c"] for r in x)
            o["arms"][k] = dict(acc=nc / len(x), trunc=float(np.mean([r["t"] for r in x])), n=len(x),
                                fin_wrong=sum(r["w"] for r in x),
                                tok_per_correct=sum(r["n"] for r in x) / nc if nc else None)
        for x, y in PAIRS:
            if x in D and y in D:
                o["pairs"][f"{x} - {y}"] = paired(D[x], D[y])
        for name, (xs, ys) in POOLED.items():
            if all(k in D for k in xs + ys):
                o["pairs"][name] = pooled(D, xs, ys)
    json.dump(res, open(a.json, "w"), indent=1)

    arms = [k for k in ARMS if any(k in res[c]["arms"] for c in caps)]
    names = [f"{x} - {y}" for x, y in PAIRS] + list(POOLED)
    names = [k for k in names if any(k in res[c]["pairs"] for c in caps)]
    cell = lambda c, f: f(c) if f(c) is not None else "—"
    head = ["| " + " | ".join(["arm"] + [str(c) for c in caps]) + " |", "|---|" + "---|" * len(caps)]
    L = ["# Qwen3.5-4B, 400 held-out MATH-500 problems, one answer per problem and seed", "",
         "Arms were generated at a 4096-token cap; truncated answers were then continued from their exact token prefix to",
         "8192 (and 16384 where shown) with the arm's weights applied. Caps in between are offline re-cuts of the same",
         "trajectories. s8 = the paper's session-8 edit (dt_bias +0.5 on all linear-attention layers).", "",
         "## Accuracy by cap", ""] + head
    for k in arms:
        L.append(f"| {LABEL[k]} | " + " | ".join(
            cell(c, lambda c: f"{res[c]['arms'][k]['acc']:.3f}" if k in res[c]["arms"] else None) for c in caps) + " |")
    L += ["", "## Finished-but-wrong answers (count out of 400)", ""] + head
    for k in arms:
        L.append(f"| {LABEL[k]} | " + " | ".join(
            cell(c, lambda c: str(res[c]['arms'][k]['fin_wrong']) if k in res[c]["arms"] else None) for c in caps) + " |")
    L += ["", "## Output tokens per correct answer", ""] + head
    for k in arms:
        L.append(f"| {LABEL[k]} | " + " | ".join(
            cell(c, lambda c: f"{res[c]['arms'][k]['tok_per_correct']:.0f}"
                 if k in res[c]["arms"] and res[c]["arms"][k]["tok_per_correct"] else None) for c in caps) + " |")
    L += ["", "## Differences in accuracy points [95% CI], problems won/lost, two-sided p (nominal, uncorrected)", "",
          "| comparison | " + " | ".join(str(c) for c in caps) + " |", "|---|" + "---|" * len(caps)]
    for k in names:
        L.append(f"| {k} | " + " | ".join(
            cell(c, lambda c: fmt(res[c]["pairs"][k]) if k in res[c]["pairs"] else None) for c in caps) + " |")
    wc = sum(s.get("wc", 0) + s.get("cw", 0) for c in caps for s in res[c]["pairs"].values())
    L += ["", f"Wrong<->correct flips summed over every single-run comparison and cap: {wc}.",
          "Single-run rows: exact sign test, bootstrap CI. Two-seed rows: per-problem mean over seeds 1000 and 2000,",
          "paired t-test and its CI (the exact sign-flip p and bootstrap CI are in q35_caps.json as `p_flip`, `ci_boot`).",
          "The p values are not corrected for the number of comparisons in this table; the one pre-specified test is",
          "the two-seed edit vs base difference at 16384 (prereg_16384.md).",
          "The same unedited model at another seed differs by several points at the larger caps (row `reseed - base`),",
          "so single-run differences of that size are not reliable on their own."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
