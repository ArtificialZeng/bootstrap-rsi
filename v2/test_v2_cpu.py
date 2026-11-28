"""CPU-only unit tests for exp/v2 (benchmarks, stats, null-space ridge solver, router contrast, LoRA algebra)."""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from v2.benchmarks import load_benchmark, grade_response
from v2.stats import summarize_records, compare_paired, compare_multiseed, strict_gate_decision, format_comparison
from v2.compilers import solve_nullspace_ridge, compute_router_contrast, compile_router_bias, compile_router_nullspace_lora


def test_benchmarks():
    for name in ("aime_practice", "aime_gate", "aime_test", "aime_24_25", "amc23", "oly_math_eval", "oly_phys", "math500_eval"):
        ds = load_benchmark(name)
        pids = [r["pid"] for r in ds]
        assert len(ds) > 0 and len(set(pids)) == len(pids), f"Bad benchmark {name}"
        print(f"  benchmark {name:16s}: n={len(ds):4d} sample_ans={ds[0]['answer']!r}")

    # Verify zero overlap between AIME practice, gate, and test!
    p_ids = {r["pid"] for r in load_benchmark("aime_practice")}
    g_ids = {r["pid"] for r in load_benchmark("aime_gate")}
    t_ids = {r["pid"] for r in load_benchmark("aime_test")}
    assert not (p_ids & g_ids) and not (p_ids & t_ids) and not (g_ids & t_ids), "AIME split leakage!"
    print("  AIME temporal split isolation verified (0 overlap)!")

    # Verify grading edge cases
    prob_aime = dict(pid="t1", source="aime", answer="42")
    g1 = grade_response("<think>foo</think> Thus \\boxed{042}.", 100, 4096, prob_aime)
    assert g1["correct"] and g1["state"] == "C"
    g2 = grade_response("<think>foo</think> Thus \\boxed{43}.", 100, 4096, prob_aime)
    assert (not g2["correct"]) and g2["finished_wrong"] and g2["state"] == "W"
    g3 = grade_response("<think>foo \\boxed{42}", 4096, 4096, prob_aime)
    assert (not g3["correct"]) and g3["truncated"] and g3["state"] == "T" and g3["boxed_ok_ignoring_trunc"]
    print("  grading edge cases passed!")


def test_stats():
    # Exact additive identity: delta == delta_completion + delta_reasoning
    base = [
        dict(pid=f"p{i}", correct=(i < 10), truncated=(10 <= i < 20), ntok=1000)
        for i in range(30)
    ]
    # Edit fixes 3 truncated (T->C) and 4 finished-wrong (W->C), breaks 1 correct (C->W)
    edit = []
    for i in range(30):
        if i == 0:
            edit.append(dict(pid=f"p{i}", correct=False, truncated=False, ntok=1000))  # C->W
        elif i < 10:
            edit.append(dict(pid=f"p{i}", correct=True, truncated=False, ntok=1000))   # C->C
        elif i < 13:
            edit.append(dict(pid=f"p{i}", correct=True, truncated=False, ntok=1000))   # T->C
        elif i < 20:
            edit.append(dict(pid=f"p{i}", correct=False, truncated=True, ntok=4096))   # T->T
        elif i < 24:
            edit.append(dict(pid=f"p{i}", correct=True, truncated=False, ntok=1000))   # W->C
        else:
            edit.append(dict(pid=f"p{i}", correct=False, truncated=False, ntok=1000))  # W->W
    c = compare_paired(edit, base)
    assert abs(c["delta"] - (c["delta_completion"] + c["delta_reasoning"])) < 1e-12
    assert c["tc"] == 3 and c["ct"] == 0 and c["wc"] == 4 and c["cw"] == 1
    gate = strict_gate_decision(c, min_delta=0.05, min_z=1.5, max_p_one=0.05)
    assert gate["passed"]
    print("  stats:", format_comparison("synthetic", c), "gate:", gate["passed"])


def test_nullspace_solver_and_router():
    rng = np.random.default_rng(7)
    # Primal branch (d_in <= N) and Dual branch (d_in > N)
    for d_in, nf, nk in [(32, 80, 80), (256, 40, 40)]:
        d_out = 16
        # Disjoint subspaces for fail vs keep -> high selectivity expected!
        X_fail = np.hstack([rng.normal(size=(nf, d_in // 2)), np.zeros((nf, d_in - d_in // 2))])
        X_keep = np.hstack([np.zeros((nk, d_in // 2)), rng.normal(size=(nk, d_in - d_in // 2))])
        target_dir = rng.normal(size=(1, d_out))
        Y_target = np.repeat(target_dir, nf, axis=0)
        A, B, diag = solve_nullspace_ridge(X_fail, Y_target, X_keep=X_keep, lam_keep=10.0, lam_reg=0.1, rank=4)
        assert A.shape == (d_in, 4) and B.shape == (4, d_out)
        assert diag["selectivity"] > 100.0, f"Low selectivity: {diag}"
        print(f"  nullspace (d_in={d_in}, nf={nf}): selectivity={diag['selectivity']:.1f} fit_err={diag['fit_rel_err']:.3f}")

    # Test MoE router contrast + state-conditional router LoRA
    N_tok, d, E = 200, 64, 32
    rec_idx = np.repeat(np.arange(10), 20)
    H = rng.normal(size=(N_tok, d)).astype(np.float32)
    # Make expert 3 strongly enriched in pos (0..4) and expert 7 enriched in neg (5..9)
    logits = rng.normal(scale=0.5, size=(N_tok, E)).astype(np.float32)
    logits[rec_idx < 5, 3] += 2.5
    logits[rec_idx >= 5, 7] += 2.5
    probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    diag_layer = dict(tok_rec_idx=rec_idx, pre_mlp_norm=H, router_logits=logits, router_probs=probs)
    con = compute_router_contrast(diag_layer, pos_rec_ids=range(5), neg_rec_ids=range(5, 10))
    assert np.argmax(con["t_stat"]) == 3 and np.argmin(con["t_stat"]) == 7
    bias = compile_router_bias(con, alpha=0.5, mode="t_stat", top_m=8)
    assert bias[3] > 0 and bias[7] < 0 and np.sum(bias != 0) == 8
    la, lb, rd = compile_router_nullspace_lora(diag_layer, con, alpha=0.5, rank=4)
    assert la.shape == (d, 4) and lb.shape == (4, E)
    print(f"  router contrast OK: top+ expert={np.argmax(con['t_stat'])}, top- expert={np.argmin(con['t_stat'])}, "
          f"snr={con['layer_snr']:.2f}, router_lora selectivity={rd['selectivity']:.2f}")


if __name__ == "__main__":
    print("Running exp/v2 CPU unit tests...")
    test_benchmarks()
    test_stats()
    test_nullspace_solver_and_router()
    print("ALL CPU UNIT TESTS PASSED!")
