"""Collect results into out/results.md and out/results.json.

Exact decomposition of an accuracy change (acc = (1 - trunc) * acc_finished):
  delta = (t_base - t_edit) * a_base            <- answers rescued from truncation, at base's finished accuracy
        + (1 - t_edit) * (a_edit - a_base)      <- finished answers becoming more/less accurate
"""
import os, sys, json, pickle
import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT

COND = {"edit": "termination-direction edit", "random": "random-direction edit (same layer, strength)",
        "concise": "prompt: think efficiently", "reseed": "no edit, different seed (noise floor)",
        "s8": "paper session-8 edit: linear-attention dt_bias +0.5", "s8rand": "dt_bias ±0.5, random sign per element"}


def decompose(s):
    tb, te, ab, ae = s["base_trunc"], s["edit_trunc"], s["base_acc_finished"], s["edit_acc_finished"]
    return dict(from_less_truncation=(tb - te) * ab, from_finished_accuracy=(1 - te) * (ae - ab))


CAPS = [1024, 2048, 3072, 4096]


def state(r):
    return "T" if r["truncated"] else ("C" if r["correct"] else "W")


def transitions(base, edit):
    """3x3 counts base-state -> edit-state (T truncated, W finished wrong, C correct) and the exact split
    delta = [n(T->C) - n(C->T)]/N + [n(W->C) - n(C->W)]/N."""
    M = {a: {b: 0 for b in "TWC"} for a in "TWC"}
    for b, e in zip(base, edit):
        M[state(b)][state(e)] += 1
    N = len(base)
    return dict(matrix=M, via_truncation=(M["T"]["C"] - M["C"]["T"]) / N,
                via_finished_answers=(M["W"]["C"] - M["C"]["W"]) / N)


def budget_curve(R):
    """Re-cut stored trajectories offline: at cap c an answer counts only if it finished within c tokens."""
    out = {}
    for c in CAPS:
        acc = np.mean([bool(r["correct"]) and r["ntok"] < c for r in R])
        tr = np.mean([r["ntok"] >= c for r in R])
        out[c] = dict(acc=float(acc), trunc=float(tr))
    return out


def tokens(R):
    n = np.array([r["ntok"] for r in R]); k = sum(r["correct"] for r in R)
    return dict(mean=float(n.mean()), p50=float(np.median(n)), p90=float(np.percentile(n, 90)),
                tokens_per_correct=float(n.sum() / k) if k else float("nan"))


def bootstrap_ci(base, edit, n=10000, seed=0):
    d = np.array([int(e["correct"]) - int(b["correct"]) for b, e in zip(base, edit)], float)
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def model_block(m):
    sp = f"{OUT}/mech_{m}_summary.json"
    if not os.path.exists(sp):
        return None, []
    S = json.load(open(sp))
    out = dict(model=m, chosen=S.get("chosen"), practice=S.get("practice_base"), continuation=S.get("continuation"),
               conditions={})
    name = "Qwen3.5-4B" if m == "q35-4b" else f"Qwen3-{m.upper()}"
    lines = [f"\n## {name} (held-out n={S.get('n_eval', '?')}, cap {S['cap']})\n"]
    if S.get("chosen"):
        lines.append(f"Direction edit: layer {S['chosen']['L']}, alpha {S['chosen']['alpha']} ({S['chosen']['how']}).\n")
    if "eval" not in S:
        lines.append("_evaluation not finished yet_\n")
        return out, lines
    base_recs = pickle.load(open(f"{OUT}/mech_{m}_eval_base.pkl", "rb"))
    b = S["eval"]["base"]
    lines.append("| condition | acc | truncated | looping | acc on finished | Δacc vs base [95% CI] | wins/losses | p (sign, 1-sided) | Δ from less truncation | Δ from finished acc |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    lines.append(f"| base | {b['acc']:.3f} | {b['trunc']:.3f} | | {b['acc_fin']:.3f} | | | | | |")
    for k, s in S.get("paired_vs_base", {}).items():
        if S.get("arm_files"):
            f = S["arm_files"][k]
        else:
            L, A = S["chosen"]["L"], S["chosen"]["alpha"]
            f = {"edit": f"eval_edit_L{L}_a{A}", "random": f"eval_random_L{L}_a{A}"}.get(k, f"eval_{k}")
        pb = S.get("paired_base", {}).get(k, "base")
        b_recs = base_recs if pb == "base" else pickle.load(open(f"{OUT}/mech_{m}_{S['arm_files'][pb]}.pkl", "rb"))
        ci = bootstrap_ci(b_recs, pickle.load(open(f"{OUT}/mech_{m}_{f}.pkl", "rb")))
        dec = decompose(s)
        arm_recs = pickle.load(open(f"{OUT}/mech_{m}_{f}.pkl", "rb"))
        out["conditions"][k] = dict(s, ci95=ci, **dec, transitions=transitions(b_recs, arm_recs), paired_base=pb,
                                    budget_curve=budget_curve(arm_recs), tokens=tokens(arm_recs))
        lines.append(f"| {COND.get(k, k)} | {s['edit_acc']:.3f} | {s['edit_trunc']:.3f} | {s['edit_looping']:.3f} | "
                     f"{s['edit_acc_finished']:.3f} | {s['delta']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | "
                     f"{s['wins']}/{s['losses']} | {s['p_sign_one_sided']:.3g} | {dec['from_less_truncation']:+.3f} | "
                     f"{dec['from_finished_accuracy']:+.3f} |")
    out["base_budget_curve"] = budget_curve(base_recs)
    out["base_tokens"] = tokens(base_recs)
    lines.append("\nTransitions base→arm (T truncated, W finished wrong, C correct) and token use:\n")
    lines.append("| condition | T→C | C→T | W→C | C→W | T→W | W→T | Δ via truncation | Δ via finished answers | mean tokens | tokens per correct |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    bt = out["base_tokens"]
    lines.append(f"| base | | | | | | | | | {bt['mean']:.0f} | {bt['tokens_per_correct']:.0f} |")
    for k, o in out["conditions"].items():
        M = o["transitions"]["matrix"]
        lines.append(f"| {COND.get(k, k)} | {M['T']['C']} | {M['C']['T']} | {M['W']['C']} | {M['C']['W']} | "
                     f"{M['T']['W']} | {M['W']['T']} | {o['transitions']['via_truncation']:+.3f} | "
                     f"{o['transitions']['via_finished_answers']:+.3f} | {o['tokens']['mean']:.0f} | "
                     f"{o['tokens']['tokens_per_correct']:.0f} |")
    lines.append("\nAccuracy when the same trajectories are cut at a smaller budget (offline re-cut):\n")
    lines.append("| condition | " + " | ".join(f"{c} tok" for c in CAPS) + " |")
    lines.append("|---|" + "---|" * len(CAPS))
    lines.append("| base | " + " | ".join(f"{out['base_budget_curve'][c]['acc']:.3f}" for c in CAPS) + " |")
    for k, o in out["conditions"].items():
        lines.append(f"| {COND.get(k, k)} | " + " | ".join(f"{o['budget_curve'][c]['acc']:.3f}" for c in CAPS) + " |")
    c = S.get("continuation")
    if c:
        lines.append(f"\nContinuation: of {c['n_truncated']} base answers cut at {S['cap']} tokens, "
                     f"{c['finished_with_2x']} finish within another {S['cap']} and {c['correct_with_2x']} are then correct "
                     f"({c['looping_at_cap']} were looping at the cut; {c['correct_among_looping']} of those end correct). "
                     f"Base accuracy with a 2x budget: **{c['base_acc_with_2x_budget']:.3f}** (vs {b['acc']:.3f}).\n")
    return out, lines


def agent_block():
    p = f"{OUT}/8b_c4096_p60_e400_results.json"
    if not os.path.exists(p):
        return None, []
    R = json.load(open(p))
    lines = ["\n## Qwen3-8B self-editing sessions (protocol 2)\n",
             "| session | decision | edits (relative change) | check on reserved: Δ (base→edit acc) | held-out Δ [p] | trunc base→edit | acc on finished base→edit |",
             "|---|---|---|---|---|---|---|"]
    for k in sorted(x for x in R if x.startswith("session")):
        r = R[k]
        ch = r.get("check")
        chs = f"{ch['delta']:+.3f} ({ch['base_acc']:.2f}→{ch['edit_acc']:.2f})" if isinstance(ch, dict) else "—"
        ev = r.get("eval")
        evs = f"{ev['delta']:+.3f} [p={ev['p_sign_one_sided']:.3g}]" if ev else "—"
        tr = f"{ev['base_trunc']:.3f}→{ev['edit_trunc']:.3f}" if ev else "—"
        af = f"{ev['base_acc_finished']:.3f}→{ev['edit_acc_finished']:.3f}" if ev else "—"
        lines.append(f"| {k} | {r['decision']} | {r.get('edits')} | {chs} | {evs} | {tr} | {af} |")
    return R, lines


def main():
    res, lines = {}, ["# Self-rewriting replication: results\n",
                      "Accuracy = finished within the cap AND last \\boxed{} correct (paper's rule). "
                      "One answer per problem, same sampling seed across arms; sign test over discordant problems; "
                      "bootstrap CI over problems.\n"]
    for m in ["1.7b", "4b", "8b", "q35-4b"]:
        o, l = model_block(m)
        if o:
            res[m] = o; lines += l
    a, l = agent_block()
    if a:
        res["agent8b"] = a; lines += l
    open(f"{OUT}/results.md", "w").write("\n".join(lines) + "\n")
    json.dump(res, open(f"{OUT}/results.json", "w"), indent=1, default=float)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
