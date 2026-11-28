"""Publication-grade figure generator for bootstrap-rsi v2 (NeurIPS/ICML vector PDF + PNG)."""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from v2.engine import V2_OUT

FIG_DIR = os.path.join(V2_OUT, "figs")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def plot_gate_phase_diagram():
    path = os.path.join(V2_OUT, "gate_phase_transition.json")
    if not os.path.exists(path):
        return
    data = json.load(open(path))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))

    n_vals = [40, 80, 120, 256]
    colors = {0.0: "#d62728", 1.0: "#ff7f0e", 1.65: "#1f77b4", 2.0: "#2ca02c"}
    labels = {0.0: "z >= 0 (unthresholded)", 1.0: "z >= 1.0", 1.65: "z >= 1.65 (p<0.05)", 2.0: "z >= 2.0"}

    # Panel A: Weak/Null editor Goodhart Gap & 4-Gen False Survival
    ax = axes[0]
    for z in (0.0, 1.0, 1.65, 2.0):
        rows = [r for r in data if r["regime"] == "weak_editor_null" and abs(r["z_min"] - z) < 1e-6]
        rows.sort(key=lambda r: r["n_g"])
        ax.plot([r["n_g"] for r in rows], [100 * r["goodhart_gap"] for r in rows],
                marker="o", lw=2.0, color=colors[z], label=labels[z])
    # Overlay empirical Qwen3-8B Chain B point (n_g=40, z=0: apparent gate +2.5%..-2.5%, heldout -1.9%)
    ax.scatter([40], [9.57], s=90, marker="*", color="black", zorder=5, label="Qwen3-8B/14B regime (n=40, z=0)")
    ax.set_xlabel("Reserved Gate Size $n_g$")
    ax.set_ylabel("Expected Goodhart Gap (Gate $-$ Held-Out, pp)")
    ax.set_title("(a) Goodhart Drift Under Weak/Null Editor")
    ax.set_xticks(n_vals)
    ax.grid(alpha=0.25, ls="--")
    ax.legend(frameon=False, loc="upper right")

    # Panel B: Competent editor true gain vs false retention rate tradeoff
    ax = axes[1]
    for z in (0.0, 1.0, 1.65, 2.0):
        rows = [r for r in data if r["regime"] == "competent_editor" and abs(r["z_min"] - z) < 1e-6]
        rows.sort(key=lambda r: r["n_g"])
        ax.plot([100 * r["false_retention_rate"] for r in rows],
                [100 * r["mean_final_true"] for r in rows],
                marker="s", lw=2.0, color=colors[z], label=labels[z])
        for r in rows:
            if r["n_g"] in (40, 256):
                ax.annotate(f"n={r['n_g']}", (100 * r["false_retention_rate"], 100 * r["mean_final_true"]),
                            textcoords="offset points", xytext=(4, 3), fontsize=7.5, color=colors[z])
    ax.set_xlabel("False Retention Rate (% Kept Edits with $\\Delta_{true} \\leq 0$)")
    ax.set_ylabel("Expected Cumulative True Held-Out Gain (pp)")
    ax.set_title("(b) Detection-Power Cliff vs. False Retention")
    ax.grid(alpha=0.25, ls="--")
    ax.legend(frameon=False, loc="lower right")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"fig1_gate_phase.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("Saved fig1_gate_phase.pdf/png")


def plot_pilot_decomposition(pilot_summary_path, out_name="fig3_decomposition"):
    if not os.path.exists(pilot_summary_path):
        return
    s = json.load(open(pilot_summary_path))
    comps = s.get("comparisons", {})
    names = [k for k in comps if k != "noise_floor_reseed"]
    if not names:
        return

    fig, ax = plt.subplots(figsize=(10.5, max(4.2, 0.45 * len(names) + 1.2)))
    y = np.arange(len(names))
    d_tc = [100 * comps[k]["delta_completion"] for k in names]
    d_wc = [100 * comps[k]["delta_reasoning"] for k in names]
    d_tot = [100 * comps[k]["delta"] for k in names]
    se = [100 * comps[k].get("se", 0.0) for k in names]

    ax.barh(y - 0.18, d_tc, height=0.34, color="#4c78a8", alpha=0.85, label="Completion Gain ($T \\leftrightarrow C$)")
    ax.barh(y + 0.18, d_wc, height=0.34, color="#e45756", alpha=0.90, label="Reasoning Gain ($W \\leftrightarrow C$)")
    ax.errorbar(d_tot, y, xerr=[1.96 * e for e in se], fmt="ko", markersize=4.5, capsize=3, lw=1.2,
                label="Total $\\Delta$ Acc (95% CI)")
    ax.axvline(0, color="black", lw=0.9)
    if "noise_floor_reseed" in comps:
        nf = abs(100 * comps["noise_floor_reseed"]["delta"])
        ax.axvspan(-nf, nf, color="gray", alpha=0.12, label=f"Reseed Noise Floor ($\\pm${nf:.1f} pp)")

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy Change vs. Unedited Base (percentage points)")
    ax.set_title(f"Additive Accuracy Decomposition ($T \\leftrightarrow C$ vs. $W \\leftrightarrow C$): {s.get('model')} on {s.get('bench')}")
    ax.grid(axis="x", alpha=0.25, ls="--")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"{out_name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_name}.pdf/png")


if __name__ == "__main__":
    plot_gate_phase_diagram()
    for p in sys.argv[1:]:
        plot_pilot_decomposition(p)
