"""Recompute the paired Qwen3-14B audit from the supplied outcome files."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from scipy import stats


DATA = Path(__file__).resolve().parent / "data"
ARMS = {
    "gen0@1000": "14b_c4096_p60_e400_eval_base.pkl",
    "gen0@2000": "14b_c4096_p60_e400_eval_reseed.pkl",
    "gen1@1000": "14b_c4096_p60_e400_chain_gen1_eval_s1000.pkl",
    "gen1@2000": "14b_c4096_p60_e400_chain_gen1_eval_s2000.pkl",
}
CONTRASTS = [(1, 0), (2, 0), (3, 1)]
CAP = 4096
RESAMPLES = 100_000
BOOTSTRAP_SEED = 20261127


def load_arms():
    indexed = []
    for name, filename in ARMS.items():
        with (DATA / filename).open("rb") as file:
            rows = pickle.load(file)
        by_pid = {row["pid"]: row for row in rows}
        if len(rows) != 400 or len(by_pid) != 400:
            raise ValueError(f"{name} does not have 400 distinct rows")
        if indexed and by_pid.keys() != indexed[0].keys():
            raise ValueError(f"{name} has a different set of problem IDs")
        for row in rows:
            if any(type(row[field]) is not bool for field in ("correct", "truncated")):
                raise ValueError(f"{name} has non-boolean grading fields")
        indexed.append(by_pid)

    pids = sorted(indexed[0])
    return pids, [[arm[pid] for pid in pids] for arm in indexed]


def bootstrap_intervals(effects, *, resamples=RESAMPLES, seed=BOOTSTRAP_SEED,
                        index_dtype=np.int64):
    """Draw full problem rows so that all seed and model pairings stay intact."""
    rng = np.random.default_rng(seed)
    n, columns = effects.shape
    means = np.empty((resamples, columns))
    batch_size = 1_000
    for start in range(0, resamples, batch_size):
        stop = min(start + batch_size, resamples)
        sample = rng.integers(n, size=(stop - start, n), dtype=index_dtype)
        means[start:stop] = effects[sample].mean(axis=1)
    return np.quantile(means, [0.025, 0.975], axis=0).T


def sign_flip_p(twice_effect):
    """Count all Rademacher sign assignments exactly, ignoring zero effects."""
    ways = [1]
    radius = 0
    for value in twice_effect:
        step = abs(int(value))
        if step:
            updated = [0] * (2 * (radius + step) + 1)
            for i, count in enumerate(ways):
                updated[i] += count
                updated[i + 2 * step] += count
            ways = updated
            radius += step
    observed = abs(int(sum(twice_effect)))
    return sum(count for i, count in enumerate(ways) if abs(i - radius) >= observed) / sum(ways)


def interval_text(interval):
    low, high = np.asarray(interval) * 100
    return f"[{low:+.2f}, {high:+.2f}]"


def bootstrap_limit_interval(effects):
    """Evaluate the percentile bootstrap distribution via convolution."""
    steps = (2 * effects).astype(int)
    n = len(steps)
    low = steps.min()
    frequencies = np.bincount(steps - low) / n
    distribution = np.array([1.0])
    for _ in range(n):
        distribution = np.convolve(distribution, frequencies)
    quantiles = np.searchsorted(distribution.cumsum(), [0.025, 0.975])
    return (quantiles + n * low) / (2 * n)


def comparison_checks(effects, arms):
    print("Contrasts in Part 1 order, then the primary test")
    print("10,000 resamples, seed 39, int16 indices (post hoc reproduction)")
    for interval in bootstrap_intervals(effects, resamples=10_000, seed=39,
                                        index_dtype=np.int16):
        print(interval_text(interval))
    print("Percentile intervals from the full resampling distribution")
    for contrast in effects.T:
        print(interval_text(bootstrap_limit_interval(contrast)))

    with (DATA / "8b_c4096_p60_e400_results.json").open() as file:
        attempts = json.load(file)["chainB"]["attempts"]
    print("Largest recorded relative edit in each 8B attempt")
    for name, attempt in attempts.items():
        matrix = max(attempt["edits"], key=attempt["edits"].get)
        print(name, attempt["decision"], matrix, attempt["edits"][matrix])

    gap = effects[:, 1] - effects[:, 2]
    result = stats.ttest_1samp(gap, 0)
    print(f"Seed 1000 minus seed 2000 edit effect: {gap.mean() * 100:+.2f} pp, "
          f"95% t CI {interval_text(result.confidence_interval())} pp, p={result.pvalue:.8f}")
    print("Evaluation records flagged as looping, in arm order")
    print(*(sum(row["looping"] for row in arm) for arm in arms))


def main(compare=False):
    pids, arms = load_arms()
    correct = np.array([[row["correct"] for row in arm] for arm in arms], dtype=int)
    truncated = np.array([[row["truncated"] for row in arm] for arm in arms], dtype=bool)
    ntok = np.array([[row["ntok"] for row in arm] for arm in arms])
    wrong_finished = ~truncated & (correct == 0)
    differences = np.stack([correct[new] - correct[old] for new, old in CONTRASTS])
    twice_primary = differences[1] + differences[2]
    primary = twice_primary / 2
    effects = np.column_stack((*differences, primary))
    if compare:
        comparison_checks(effects, arms)
        return
    intervals = bootstrap_intervals(effects)

    print("# T18 independent audit of the Qwen3-14B chain")
    print()
    print("## Part 1. Recomputed results")
    print()
    print(f"Each arm contains the same {len(pids)} distinct problem IDs, each appearing "
          "once. All comparisons pair rows by `pid`. The denominator for each arm is 400.")
    print()
    print("| Arm | Correct | Accuracy | Truncated | Truncation rate | Finished wrong |")
    print("|---|---:|---:|---:|---:|---:|")
    for i, name in enumerate(ARMS):
        print(f"| {name} | {correct[i].sum()} | {correct[i].mean():.2%} | "
              f"{truncated[i].sum()} | {truncated[i].mean():.2%} | "
              f"{wrong_finished[i].sum()} |")
    print()
    print("### Paired comparisons")
    print()
    print(f"All intervals in the table are 95% percentile bootstrap intervals from "
          f"{RESAMPLES:,} resamples of 400 problems with replacement. NumPy's "
          f"`default_rng` used seed {BOOTSTRAP_SEED} and `int64` index draws. "
          "The same problem resamples "
          "were used across contrasts and kept all outcomes for a problem together.")
    print()
    print("| Comparison | Difference (pp) | Gained | Lost | Exact sign-test p | "
          "95% bootstrap CI (pp) |")
    print("|---|---:|---:|---:|---:|---:|")
    names = list(ARMS)
    for i, (new, old) in enumerate(CONTRASTS):
        delta = differences[i]
        gained = np.count_nonzero(delta > 0)
        lost = np.count_nonzero(delta < 0)
        p = stats.binomtest(gained, gained + lost, p=0.5).pvalue
        print(f"| {names[new]} − {names[old]} | {delta.mean() * 100:+.2f} | "
              f"{gained} | {lost} | {p:.8f} | {interval_text(intervals[i])} |")
    print()
    print("### Pre-specified primary test")
    print()
    t_test = stats.ttest_1samp(primary, 0)
    print(f"The mean of the two seed-matched correctness differences per problem "
          f"was **{primary.mean() * 100:+.3f} percentage points**. The 95% t interval "
          f"was {interval_text(t_test.confidence_interval())} pp. The two-sided paired "
          f"t-test gave t({len(pids) - 1}) = {t_test.statistic:.6f}, "
          f"p = {t_test.pvalue:.8f}. The exact two-sided sign-flip permutation p "
          f"was {sign_flip_p(twice_primary):.8f}, using the absolute mean as the "
          "statistic and counting all sign assignments of nonzero problem differences. "
          f"The 95% percentile bootstrap interval was {interval_text(intervals[3])} pp "
          "using the resampling described above.")
    print()
    print(f"Across problems, {np.count_nonzero(primary > 0)} had dᵢ > 0, "
          f"{np.count_nonzero(primary < 0)} had dᵢ < 0, and "
          f"{np.count_nonzero(primary == 0)} had dᵢ = 0.")
    print()
    print("### Transitions from gen0 to gen1")
    print()
    print("Rows are gen0 outcomes and columns are gen1 outcomes. T means truncated, "
          "W means finished wrong, and C means correct. A changed score means that "
          "correctness changed.")
    if np.any(truncated & (correct == 1)):
        raise ValueError("A row marked both correct and truncated prevents assigning states")
    states = np.where(truncated, 0, np.where(correct, 2, 1))
    for new, old in CONTRASTS[1:]:
        table = np.zeros((3, 3), dtype=int)
        np.add.at(table, (states[old], states[new]), 1)
        print()
        print(f"**Seed {names[new].split('@')[1]}**")
        print()
        print("| gen0 → gen1 | T | W | C |")
        print("|---|---:|---:|---:|")
        for label, (t, w, c) in zip("TWC", table):
            print(f"| {label} | {t} | {w} | {c} |")
        changed = np.count_nonzero(correct[new] != correct[old])
        print()
        print(f"{changed} changed scores, of which {table[0, 2] + table[2, 0]} were T ↔ C.")
    changed_either = (differences[1:] != 0).any(axis=0)
    tc_pairs = (((states[:2] == 0) & (states[2:] == 2)) |
                ((states[:2] == 2) & (states[2:] == 0)))
    print()
    print(f"Across both seeds, {np.count_nonzero(differences[1:])} problem–seed "
          "pairs changed score, including "
          f"{tc_pairs.sum()} "
          f"T ↔ C transitions. Counting each problem once, {changed_either.sum()} "
          f"problems changed score in at least one seed and {tc_pairs.any(axis=0).sum()} had "
          "a T ↔ C transition in at least one seed.")
    print()
    print("### Grading field consistency")
    print()
    print(f"Across {correct.size:,} records, {np.count_nonzero(truncated & (correct == 1))} "
          "were marked both correct and truncated. All grading flags are Boolean. "
          f"There were {np.count_nonzero(truncated != (ntok >= CAP))} discrepancies "
          f"between the truncation flag and whether `ntok` reached the {CAP:,}-token cap.")
    token_lengths = np.array([[len(row["out_ids"]) for row in arm] for arm in arms])
    print(f"The recorded `ntok` equaled the length of `out_ids` in "
          f"{np.count_nonzero(ntok == token_lengths):,}/{correct.size:,} records.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare", action="store_true", help="print additional claim checks")
    main(parser.parse_args().compare)
