"""Rigorous statistical evaluation: 3x3 T/W/C transition decomposition, multi-seed pooled tests, strict gates."""
import collections
import numpy as np
from scipy import stats as sp_stats


def summarize_records(recs, label=""):
    n = len(recs)
    if n == 0:
        return dict(label=label, n=0)
    c = sum(1 for r in recs if r["correct"])
    t = sum(1 for r in recs if r["truncated"])
    w = sum(1 for r in recs if not r["truncated"] and not r["correct"])
    fin = n - t
    loop = sum(1 for r in recs if r.get("looping"))
    toks = [r["ntok"] for r in recs]
    fin_toks = [r["ntok"] for r in recs if not r["truncated"]]
    tot_tok = sum(toks)
    return dict(
        label=label,
        n=n,
        acc=c / n,
        trunc=t / n,
        wrong_rate=w / n,
        acc_finished=(c / fin) if fin > 0 else float("nan"),
        n_C=c,
        n_W=w,
        n_T=t,
        looping=loop / n,
        mean_tok=float(np.mean(toks)),
        med_tok=float(np.median(toks)),
        med_tok_finished=float(np.median(fin_toks)) if fin_toks else float("nan"),
        tok_per_correct=(tot_tok / c) if c > 0 else float("inf"),
    )


def _index_by_pid(recs):
    d = {}
    for r in recs:
        pid = r["pid"]
        assert pid not in d, f"Duplicate pid {pid}"
        d[pid] = r
    return d


def _state(r):
    return "T" if r["truncated"] else ("C" if r["correct"] else "W")


def bootstrap_ci(diffs, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs, dtype=np.float64)
    n = len(diffs)
    if n == 0:
        return [float("nan"), float("nan")]
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [float(lo), float(hi)]


def compare_paired(arm_edit, arm_base):
    """Exact paired comparison (edit - base) with full 3x3 T/W/C decomposition & W<->C significance."""
    E = _index_by_pid(arm_edit)
    B = _index_by_pid(arm_base)
    pids = sorted(set(E) & set(B))
    n = len(pids)
    assert n > 0, "No overlapping problem IDs"

    trans = collections.Counter(_state(B[p]) + _state(E[p]) for p in pids)
    T = {a + b: int(trans.get(a + b, 0)) for a in "TWC" for b in "TWC"}

    diffs = np.array([int(E[p]["correct"]) - int(B[p]["correct"]) for p in pids], dtype=np.float64)
    wins = int(np.sum(diffs > 0))
    losses = int(np.sum(diffs < 0))
    delta = float(diffs.mean())
    se = float(diffs.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    p_sign_two = float(sp_stats.binomtest(wins, wins + losses, 0.5).pvalue) if (wins + losses) > 0 else 1.0
    p_sign_one = float(sp_stats.binomtest(wins, wins + losses, 0.5, alternative="greater").pvalue) if (wins + losses) > 0 else 1.0
    p_t_two = float(sp_stats.ttest_1samp(diffs, 0.0).pvalue) if (n > 1 and diffs.std() > 0) else 1.0
    ci = bootstrap_ci(diffs)

    # Exact additive decomposition of accuracy gain:
    # delta == delta_completion (T<->C) + delta_reasoning (W<->C)
    wc, cw = T["WC"], T["CW"]
    tc, ct = T["TC"], T["CT"]
    delta_completion = (tc - ct) / n
    delta_reasoning = (wc - cw) / n
    p_wc_two = float(sp_stats.binomtest(wc, wc + cw, 0.5).pvalue) if (wc + cw) > 0 else 1.0
    p_wc_one = float(sp_stats.binomtest(wc, wc + cw, 0.5, alternative="greater").pvalue) if (wc + cw) > 0 else 1.0

    # Length delta on jointly finished problems
    both_fin = [E[p]["ntok"] - B[p]["ntok"] for p in pids if not E[p]["truncated"] and not B[p]["truncated"]]
    mean_dtok_fin = float(np.mean(both_fin)) if both_fin else float("nan")
    mean_dtok_all = float(np.mean([E[p]["ntok"] - B[p]["ntok"] for p in pids]))

    return dict(
        n=n,
        base_acc=float(np.mean([B[p]["correct"] for p in pids])),
        edit_acc=float(np.mean([E[p]["correct"] for p in pids])),
        delta=delta,
        se=se,
        ci95=ci,
        wins=wins,
        losses=losses,
        p_sign_two=p_sign_two,
        p_sign_one=p_sign_one,
        p_t_two=p_t_two,
        delta_completion=delta_completion,
        delta_reasoning=delta_reasoning,
        wc=wc,
        cw=cw,
        tc=tc,
        ct=ct,
        p_wc_two=p_wc_two,
        p_wc_one=p_wc_one,
        transitions=T,
        mean_dtok_all=mean_dtok_all,
        mean_dtok_both_finished=mean_dtok_fin,
    )


def compare_multiseed(edit_runs, base_runs):
    """Pool multiple independent seeds per problem: d_i = mean_s(edit_{i,s} - base_{i,s})."""
    assert len(edit_runs) == len(base_runs) and len(edit_runs) >= 1
    S = len(edit_runs)
    E_list = [_index_by_pid(r) for r in edit_runs]
    B_list = [_index_by_pid(r) for r in base_runs]
    pids = sorted(set.intersection(*(set(d) for d in E_list + B_list)))
    n = len(pids)
    per_seed = [compare_paired(e, b) for e, b in zip(edit_runs, base_runs)]

    d_prob = np.array([
        np.mean([int(E_list[s][p]["correct"]) - int(B_list[s][p]["correct"]) for s in range(S)])
        for p in pids
    ], dtype=np.float64)
    d_wc = np.array([
        np.mean([
            (1 if (_state(B_list[s][p]) == "W" and _state(E_list[s][p]) == "C") else
             -1 if (_state(B_list[s][p]) == "C" and _state(E_list[s][p]) == "W") else 0)
            for s in range(S)
        ])
        for p in pids
    ], dtype=np.float64)
    d_tc = np.array([
        np.mean([
            (1 if (_state(B_list[s][p]) == "T" and _state(E_list[s][p]) == "C") else
             -1 if (_state(B_list[s][p]) == "C" and _state(E_list[s][p]) == "T") else 0)
            for s in range(S)
        ])
        for p in pids
    ], dtype=np.float64)

    delta = float(d_prob.mean())
    se = float(d_prob.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    p_t_two = float(sp_stats.ttest_1samp(d_prob, 0.0).pvalue) if (n > 1 and d_prob.std() > 0) else 1.0
    p_t_one = (p_t_two / 2.0 if delta > 0 else 1.0 - p_t_two / 2.0)
    p_wc_t_two = float(sp_stats.ttest_1samp(d_wc, 0.0).pvalue) if (n > 1 and d_wc.std() > 0) else 1.0

    # Aggregate 3x3 counts across all seeds
    T_sum = collections.Counter()
    for ps in per_seed:
        T_sum.update(ps["transitions"])

    return dict(
        n_problems=n,
        n_seeds=S,
        base_acc=float(np.mean([ps["base_acc"] for ps in per_seed])),
        edit_acc=float(np.mean([ps["edit_acc"] for ps in per_seed])),
        delta=delta,
        se=se,
        ci95=bootstrap_ci(d_prob),
        p_t_two=p_t_two,
        p_t_one=p_t_one,
        delta_completion=float(d_tc.mean()),
        delta_reasoning=float(d_wc.mean()),
        ci95_reasoning=bootstrap_ci(d_wc),
        p_reasoning_t_two=p_wc_t_two,
        transitions_sum=dict(T_sum),
        per_seed=per_seed,
    )


def strict_gate_decision(comp, min_delta=0.02, min_z=1.65, max_p_one=0.05, require_non_negative_reasoning=False):
    """Strict statistical gate preventing Goodhart retention on noise."""
    z = comp["delta"] / comp["se"] if comp["se"] > 0 else (99.0 if comp["delta"] > 0 else -99.0)
    p_one = comp.get("p_t_one", comp.get("p_sign_one", 1.0))
    ok = (comp["delta"] >= min_delta) and (z >= min_z) and (p_one <= max_p_one)
    if require_non_negative_reasoning:
        ok = ok and (comp["delta_reasoning"] >= -0.01)
    return dict(passed=bool(ok), z=float(z), delta=comp["delta"], p_one=float(p_one),
                delta_reasoning=comp["delta_reasoning"], delta_completion=comp["delta_completion"])


def format_comparison(name, c):
    if "per_seed" in c:
        return (f"[{name}] n={c['n_problems']}x{c['n_seeds']}s "
                f"base={100*c['base_acc']:.2f}% edit={100*c['edit_acc']:.2f}% "
                f"delta={100*c['delta']:+.2f}% CI=[{100*c['ci95'][0]:+.2f},{100*c['ci95'][1]:+.2f}] "
                f"p={c['p_t_two']:.4g} | T->C:{100*c['delta_completion']:+.2f}% "
                f"W->C:{100*c['delta_reasoning']:+.2f}% (p_wc={c['p_reasoning_t_two']:.4g})")
    return (f"[{name}] n={c['n']} base={100*c['base_acc']:.2f}% edit={100*c['edit_acc']:.2f}% "
            f"delta={100*c['delta']:+.2f}% CI=[{100*c['ci95'][0]:+.2f},{100*c['ci95'][1]:+.2f}] "
            f"W/L={c['wins']}/{c['losses']} p={c['p_sign_two']:.4g} | "
            f"T->C:{c['tc']}-{c['ct']}({100*c['delta_completion']:+.2f}%) "
            f"W->C:{c['wc']}-{c['cw']}({100*c['delta_reasoning']:+.2f}%, p={c['p_wc_two']:.4g})")
