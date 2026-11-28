"""Training-free closed-form weight compilers & reversible model editor for MLX Qwen3 / Qwen3.5 (dense & MoE).

Core mechanisms (ZERO gradient descent, ZERO backpropagation, ZERO full-model BF16 dequantization):
  1. MoERouterCompiler:
       - Expert Contrastive Bias (log-odds / Fisher t-stat over 256 experts on paired C vs W trajectories)
       - State-Conditional Null-Space Router Weight Compilation (closed-form ridge LoRA on mlp.gate
         pushing failure-state routing toward correct-state experts while preserving correct-state routing!)
  2. NullSpaceLoRACompiler:
       - Exact closed-form Woodbury/Primal Null-Space Ridge LoRA for mlp.down_proj / shared_expert.down_proj / o_proj
       - Supports both Contrastive Trajectory Direction Compilation AND Context-to-Weight Internalization!
       - Supports exact rank-1 orthogonal projection W <- W - alpha u u^T W natively on 4-bit weights via rank-1 LoRA.
  3. DeltaNetCompiler:
       - Uniform & per-head data-driven Gated DeltaNet dt_bias / A_log editing.
  4. SteeringWrapper (Control):
       - Inference-time residual activation steering (TGAS-style) for direct apples-to-apples comparison.
"""
import copy
import numpy as np
import mlx.core as mx
import mlx.nn as nn

from .engine import get_layers


# ---- Lightweight Native-Quantized Wrappers (0% base dequantization) ----

class RouterGateWrapper(nn.Module):
    """Wraps MoE mlp.gate with reversible bias and/or low-rank state-conditional delta."""
    def __init__(self, base_gate, bias=None, lora_a=None, lora_b=None, scale=1.0):
        super().__init__()
        self.base = base_gate
        self.bias = mx.array(bias, dtype=mx.float32) if bias is not None else None
        self.lora_a = mx.array(lora_a, dtype=mx.float32) if lora_a is not None else None
        self.lora_b = mx.array(lora_b, dtype=mx.float32) if lora_b is not None else None
        self.scale = float(scale)

    def __call__(self, x):
        y = self.base(x)
        if self.lora_a is not None and self.lora_b is not None:
            dx = (x.astype(mx.float32) @ self.lora_a) @ self.lora_b
            y = y + (self.scale * dx).astype(y.dtype)
        if self.bias is not None:
            y = y + self.bias.astype(y.dtype)
        return y


class FastLoRAWrapper(nn.Module):
    """Wraps nn.QuantizedLinear or nn.Linear with exact low-rank delta y = base(x) + scale * (x @ A) @ B."""
    def __init__(self, base_linear, lora_a, lora_b, scale=1.0):
        super().__init__()
        self.base = base_linear
        self.lora_a = mx.array(lora_a, dtype=mx.bfloat16)  # (in_dim, r)
        self.lora_b = mx.array(lora_b, dtype=mx.bfloat16)  # (r, out_dim)
        self.scale = float(scale)

    def __call__(self, x):
        y = self.base(x)
        z = (x.astype(mx.bfloat16) @ self.lora_a) @ self.lora_b
        return y + (self.scale * z).astype(y.dtype)


class ResidualSteeringWrapper(nn.Module):
    """Inference-time activation steering control (adds vector delta to layer output without modifying weights)."""
    def __init__(self, base_layer, delta_vec):
        super().__init__()
        self.base = base_layer
        self.delta_vec = mx.array(delta_vec, dtype=mx.bfloat16)  # (d,)
        self.is_linear = getattr(base_layer, "is_linear", False)

    def __call__(self, x, mask=None, cache=None):
        out = self.base(x, mask=mask, cache=cache)
        return out + self.delta_vec.astype(out.dtype)


# ---- Closed-Form Linear Algebra Solvers (Pure CPU float64/float32, <0.2s) ----

def solve_nullspace_ridge(X_fail, Y_target, X_keep=None, lam_keep=2.0, lam_reg=1.0, rank=None):
    """Solve min_M (1/n_f)||X_f M - Y||_F^2 + (lam_keep/n_k)||X_k M||_F^2 + lam_reg||M||_F^2.

    Args:
      X_fail   : (n_f, d_in) failure-state inputs
      Y_target : (n_f, d_out) desired output shifts on failure states
      X_keep   : (n_k, d_in) optional preserve-state inputs (from already-correct trajectories!)
      lam_keep : null-space preservation weight on X_keep
      lam_reg  : L2 ridge regularization
      rank     : optional target rank for truncated SVD factorization M ≈ A @ B

    Returns:
      lora_a (d_in, r), lora_b (r, d_out), diagnostics dict
    """
    Xf = np.asarray(X_fail, dtype=np.float64)
    Yt = np.asarray(Y_target, dtype=np.float64)
    nf, d_in = Xf.shape
    d_out = Yt.shape[1]
    assert Yt.shape[0] == nf

    if X_keep is not None and len(X_keep) > 0 and lam_keep > 0:
        Xk = np.asarray(X_keep, dtype=np.float64)
        nk = Xk.shape[0]
        X_stack = np.vstack([Xf / np.sqrt(nf), np.sqrt(lam_keep / nk) * Xk])
        Y_stack = np.vstack([Yt / np.sqrt(nf), np.zeros((nk, d_out), dtype=np.float64)])
    else:
        Xk = None
        nk = 0
        X_stack = Xf / np.sqrt(nf)
        Y_stack = Yt / np.sqrt(nf)

    N = X_stack.shape[0]
    # Choose primal (d_in x d_in) or dual Woodbury (N x N), whichever is smaller!
    if d_in <= N:
        G = X_stack.T @ X_stack + lam_reg * np.eye(d_in, dtype=np.float64)
        RHS = X_stack.T @ Y_stack
        M = np.linalg.solve(G, RHS)
    else:
        K = X_stack @ X_stack.T + lam_reg * np.eye(N, dtype=np.float64)
        alpha = np.linalg.solve(K, Y_stack)
        M = X_stack.T @ alpha

    # Factorize M into balanced low-rank factors (A, B) with A @ B ≈ M
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    if rank is not None and rank < len(S):
        r = int(rank)
        U, S, Vt = U[:, :r], S[:r], Vt[:r, :]
    else:
        # Keep non-trivial singular values
        nz = S > (1e-7 * (S[0] if len(S) else 1.0))
        r = max(1, int(np.sum(nz)))
        U, S, Vt = U[:, :r], S[:r], Vt[:r, :]

    s_sqrt = np.sqrt(np.maximum(S, 0.0))
    lora_a = (U * s_sqrt[None, :]).astype(np.float32)
    lora_b = (s_sqrt[:, None] * Vt).astype(np.float32)

    # Diagnostics: how well did we fit Y_target vs disturb X_keep?
    M_approx = lora_a.astype(np.float64) @ lora_b.astype(np.float64)
    fail_pred = Xf @ M_approx
    fit_rel = float(np.linalg.norm(fail_pred - Yt) / (np.linalg.norm(Yt) + 1e-12))
    fail_rms = float(np.sqrt(np.mean(fail_pred ** 2)))
    keep_rms = float(np.sqrt(np.mean((Xk @ M_approx) ** 2))) if Xk is not None else 0.0
    selectivity = fail_rms / (keep_rms + 1e-9) if Xk is not None else float("inf")

    diag = dict(
        rank=int(r),
        frob_norm=float(np.linalg.norm(M_approx)),
        top_sv=float(S[0]) if len(S) else 0.0,
        fit_rel_err=fit_rel,
        fail_rms=fail_rms,
        keep_rms=keep_rms,
        selectivity=float(selectivity),
    )
    return lora_a, lora_b, diag


# ---- Method 1: MoE Router Self-Rewiring Compiler ----

def compute_router_contrast(diag_layer, pos_rec_ids, neg_rec_ids, eps=1e-4):
    """Compute per-expert statistics comparing positive (e.g. C) vs negative (e.g. W or T) trajectories.

    Uses record-level aggregation first so long trajectories don't dominate short ones!
    Returns dict with per-expert log_odds, diff_prob, t_stat, logit_diff, pos_mask, neg_mask.
    """
    rec_idx = diag_layer["tok_rec_idx"]
    probs = diag_layer["router_probs"]    # (N_tok, E)
    logits = diag_layer["router_logits"]  # (N_tok, E)
    E = probs.shape[1]

    pos_set = set(pos_rec_ids)
    neg_set = set(neg_rec_ids)

    pos_rec_probs, neg_rec_probs = [], []
    pos_rec_logits, neg_rec_logits = [], []
    for rid in sorted(pos_set):
        m = rec_idx == rid
        if np.any(m):
            pos_rec_probs.append(probs[m].mean(axis=0))
            pos_rec_logits.append(logits[m].mean(axis=0))
    for rid in sorted(neg_set):
        m = rec_idx == rid
        if np.any(m):
            neg_rec_probs.append(probs[m].mean(axis=0))
            neg_rec_logits.append(logits[m].mean(axis=0))

    if len(pos_rec_probs) < 2 or len(neg_rec_probs) < 2:
        raise ValueError(f"Need >=2 pos and >=2 neg records, got {len(pos_rec_probs)} and {len(neg_rec_probs)}")

    P_pos = np.stack(pos_rec_probs, axis=0)   # (n_pos, E)
    P_neg = np.stack(neg_rec_probs, axis=0)   # (n_neg, E)
    L_pos = np.stack(pos_rec_logits, axis=0)
    L_neg = np.stack(neg_rec_logits, axis=0)

    mean_p_pos = P_pos.mean(axis=0)
    mean_p_neg = P_neg.mean(axis=0)
    diff_p = mean_p_pos - mean_p_neg
    log_odds = np.log((mean_p_pos + eps) / (mean_p_neg + eps))

    # Welch t-statistic per expert on record-level router logits (measures consistent signal across problems!)
    diff_l = L_pos.mean(axis=0) - L_neg.mean(axis=0)
    var_l = L_pos.var(axis=0, ddof=1) / len(L_pos) + L_neg.var(axis=0, ddof=1) / len(L_neg)
    t_stat = diff_l / np.sqrt(var_l + 1e-8)

    # Layer SNR: root-mean-square of top-16 absolute t-stats
    top16_t = np.sort(np.abs(t_stat))[-16:]
    layer_snr = float(np.sqrt(np.mean(top16_t ** 2)))

    return dict(
        n_pos=len(L_pos),
        n_neg=len(L_neg),
        num_experts=E,
        mean_p_pos=mean_p_pos,
        mean_p_neg=mean_p_neg,
        diff_prob=diff_p,
        log_odds=log_odds,
        diff_logit=diff_l,
        t_stat=t_stat,
        layer_snr=layer_snr,
        pos_tok_mask=np.isin(rec_idx, list(pos_set)),
        neg_tok_mask=np.isin(rec_idx, list(neg_set)),
    )


def compile_router_bias(contrast, alpha=0.3, mode="t_stat", top_m=32, clip_std=None):
    """Build a sparse or smooth router logit bias vector (E,) from expert contrast statistics."""
    if mode == "t_stat":
        raw = contrast["t_stat"].copy()
    elif mode == "log_odds":
        raw = contrast["log_odds"].copy()
    elif mode == "diff_logit":
        raw = contrast["diff_logit"].copy()
    else:
        raise ValueError(f"Unknown router bias mode: {mode}")

    # Zero-mean so softmax temperature/scale doesn't drift
    raw = raw - np.mean(raw)
    if top_m is not None and top_m < len(raw):
        keep_idx = np.argsort(np.abs(raw))[-top_m:]
        masked = np.zeros_like(raw)
        masked[keep_idx] = raw[keep_idx]
        raw = masked
        # Re-center active entries slightly or keep sparse zero-background
    norm = np.std(raw[raw != 0]) if np.any(raw != 0) else 1.0
    vec = (raw / (norm + 1e-8)) * alpha
    if clip_std is not None:
        vec = np.clip(vec, -clip_std * alpha, clip_std * alpha)
    return vec.astype(np.float32)


def compile_router_nullspace_lora(diag_layer, contrast, alpha=0.5, rank=8, lam_keep=3.0, lam_reg=5.0,
                                  top_m=48, target_mode="t_stat"):
    """Compile a STATE-CONDITIONAL null-space router weight update ΔW_gate = B A!

    On negative-trajectory states H_neg: shifts router logits toward positive-enriched experts
    (weighted by t_stat or logit_diff).
    On positive-trajectory states H_pos: preserves original routing via null-space penalty (ΔW_gate H_pos ≈ 0)!
    """
    H = diag_layer["pre_mlp_norm"]
    H_neg = H[contrast["neg_tok_mask"]]
    H_pos = H[contrast["pos_tok_mask"]]

    direction = compile_router_bias(contrast, alpha=1.0, mode=target_mode, top_m=top_m)  # (E,)
    if target_mode == "logit_shift":
        # Push each failure token's router logits toward the mean positive logit profile!
        mean_l_pos = diag_layer["router_logits"][contrast["pos_tok_mask"]].mean(axis=0, keepdims=True)
        Y_target = mean_l_pos - diag_layer["router_logits"][contrast["neg_tok_mask"]]
        # Normalize per-token scale
        rms = np.sqrt(np.mean(Y_target ** 2)) + 1e-8
        Y_target = (Y_target / rms) * alpha
    else:
        # Rank-1 target direction across experts, solved with state-selective null-space input vector!
        Y_target = np.broadcast_to((alpha * direction)[None, :], (len(H_neg), len(direction))).copy()

    lora_a, lora_b, diag = solve_nullspace_ridge(
        H_neg, Y_target, X_keep=H_pos, lam_keep=lam_keep, lam_reg=lam_reg, rank=rank
    )
    return lora_a, lora_b, diag


# ---- Method 2: Closed-Form Null-Space Residual / MLP LoRA Compiler ----

def compile_contrastive_direction(diag_layer, pos_rec_ids, neg_rec_ids, fisher_reg=1e-2):
    """Compute Fisher-regularized contrastive residual direction d* = (Sigma_W + reg I)^{-1} (mu+ - mu-)."""
    rec_idx = diag_layer["tok_rec_idx"]
    H = diag_layer["residual_toks"]
    pos_m = np.isin(rec_idx, list(pos_rec_ids))
    neg_m = np.isin(rec_idx, list(neg_rec_ids))
    H_pos = H[pos_m].astype(np.float64)
    H_neg = H[neg_m].astype(np.float64)
    mu_pos = H_pos.mean(axis=0)
    mu_neg = H_neg.mean(axis=0)
    diff = mu_pos - mu_neg

    # Diagonal shrinkage Fisher LDA (stable in high dim d=2048..5120 with moderate N)
    var_w = 0.5 * (H_pos.var(axis=0) + H_neg.var(axis=0))
    scale = var_w.mean()
    d_fisher = diff / (var_w + fisher_reg * scale + 1e-8)
    d_fisher = d_fisher / (np.linalg.norm(d_fisher) + 1e-12)
    d_diff = diff / (np.linalg.norm(diff) + 1e-12)

    # Separation AUC proxy (Cohen's d on projections)
    p_pos = H_pos @ d_diff
    p_neg = H_neg @ d_diff
    cohen_d = float((p_pos.mean() - p_neg.mean()) / (np.sqrt(0.5 * (p_pos.var() + p_neg.var())) + 1e-8))
    return dict(
        d_diff=d_diff.astype(np.float32),
        d_fisher=d_fisher.astype(np.float32),
        mean_norm=float(np.linalg.norm(diff)),
        cohen_d=cohen_d,
        pos_tok_mask=pos_m,
        neg_tok_mask=neg_m,
    )


def compile_nullspace_mlp_lora(diag_layer, pos_rec_ids, neg_rec_ids, alpha=0.5, rank=4,
                               lam_keep=3.0, lam_reg=5.0, use_fisher=False):
    """Compile a null-space-preserving LoRA for mlp.down_proj (or shared_expert.down_proj).

    Maps failure MLP activations X_neg -> +alpha * d_contrastive (residual push toward correct subspace)
    while forcing X_pos -> 0 (null-space protection against C->W regression!).
    """
    c_info = compile_contrastive_direction(diag_layer, pos_rec_ids, neg_rec_ids)
    X = diag_layer["mlp_in_toks"]
    X_neg = X[c_info["neg_tok_mask"]]
    X_pos = X[c_info["pos_tok_mask"]]
    d_vec = c_info["d_fisher"] if use_fisher else c_info["d_diff"]
    # Scale target to match natural residual delta magnitude (alpha * mean_norm)
    step = alpha * c_info["mean_norm"]
    Y_target = np.broadcast_to((step * d_vec)[None, :], (len(X_neg), len(d_vec))).copy()

    lora_a, lora_b, diag = solve_nullspace_ridge(
        X_neg, Y_target, X_keep=X_pos, lam_keep=lam_keep, lam_reg=lam_reg, rank=rank
    )
    diag["cohen_d"] = c_info["cohen_d"]
    diag["mean_diff_norm"] = c_info["mean_norm"]
    return lora_a, lora_b, diag


def compile_context_internalization_lora(diag_bare, diag_ctx, preserve_diag=None, alpha=1.0,
                                         rank=8, lam_keep=2.0, lam_reg=5.0):
    """Compile the behavioral effect of a verification/reasoning prompt into LoRA weights (zero backprop!).

    Aligns per-problem bare MLP activations X_bare -> alpha * (mean_H_ctx - mean_H_bare)
    so the model reproduces the scaffolded hidden trajectory WITHOUT the prompt in context!
    """
    X_bare_list, dY_list = [], []
    n_recs = min(len(diag_bare["residual_mean"]), len(diag_ctx["residual_mean"]))
    rec_idx = diag_bare["tok_rec_idx"]
    X_all = diag_bare["mlp_in_toks"]

    for r in range(n_recs):
        m = rec_idx == r
        if not np.any(m):
            continue
        dh = diag_ctx["residual_mean"][r] - diag_bare["residual_mean"][r]  # (d,)
        X_r = X_all[m]
        X_bare_list.append(X_r)
        dY_list.append(np.broadcast_to((alpha * dh)[None, :], (len(X_r), len(dh))))

    X_fail = np.vstack(X_bare_list)
    Y_target = np.vstack(dY_list)
    X_keep = preserve_diag["mlp_in_toks"] if preserve_diag is not None else None

    return solve_nullspace_ridge(
        X_fail, Y_target, X_keep=X_keep, lam_keep=lam_keep, lam_reg=lam_reg, rank=rank
    )


def dequantize_single_matrix(mod):
    """Temporarily dequantize ONE linear module to float32 numpy without touching the model."""
    if isinstance(mod, nn.QuantizedLinear):
        w = mx.dequantize(
            mod.weight, mod.scales, mod.biases,
            group_size=mod.group_size, bits=mod.bits, mode=getattr(mod, "mode", "affine")
        ).astype(mx.float32)
        mx.eval(w)
        return np.array(w)
    w = mod.weight.astype(mx.float32)
    mx.eval(w)
    return np.array(w)


def compile_rank1_ablation_lora(mod, u_vec, alpha):
    """Exact rank-1 orthogonal projection W <- W - alpha u u^T W via rank-1 LoRA (0% model dequantization!)."""
    u = np.asarray(u_vec, dtype=np.float32)
    u = u / (np.linalg.norm(u) + 1e-12)
    W = dequantize_single_matrix(mod)  # (out_dim, in_dim)
    v = W.T @ u                        # (in_dim,)
    lora_a = v[:, None].astype(np.float32)
    lora_b = (-float(alpha) * u)[None, :].astype(np.float32)
    return lora_a, lora_b


# ---- Transactional Reversible Model Editor (supports multi-generation stacking!) ----

class ModelEditor:
    """Reversible, transactional weight editor supporting native 4-bit/8-bit LoRA, MoE Router, dt_bias & steering."""

    def __init__(self, model):
        self.model = model
        self.layers = get_layers(model)
        # Stack of committed generations; each entry is list of undo closures + metadata
        self._committed = []
        self._staged = []

    def _get_target_parent_and_attr(self, layer_idx, target):
        blk = self.layers[layer_idx]
        if target == "router":
            return blk.mlp, "gate"
        if target == "shared_gate":
            return blk.mlp, "shared_expert_gate"
        if target == "mlp_down":
            if hasattr(blk.mlp, "shared_expert"):
                return blk.mlp.shared_expert, "down_proj"
            return blk.mlp, "down_proj"
        if target == "attn_out":
            if getattr(blk, "is_linear", False):
                return blk.linear_attn, "out_proj"
            return blk.self_attn, "o_proj"
        raise ValueError(f"Unknown edit target: {target}")

    def apply_router_edit(self, layer_idx, bias=None, lora_a=None, lora_b=None, scale=1.0, meta=None):
        par, attr = self._get_target_parent_and_attr(layer_idx, "router")
        old_mod = getattr(par, attr)
        wrapper = RouterGateWrapper(old_mod, bias=bias, lora_a=lora_a, lora_b=lora_b, scale=scale)
        setattr(par, attr, wrapper)
        mx.eval(wrapper.parameters())
        self._staged.append((
            lambda p=par, a=attr, o=old_mod: setattr(p, a, o),
            dict(kind="router", layer=layer_idx, has_bias=bias is not None, has_lora=lora_a is not None, meta=meta or {})
        ))

    def apply_lora_edit(self, layer_idx, target, lora_a, lora_b, scale=1.0, meta=None):
        par, attr = self._get_target_parent_and_attr(layer_idx, target)
        old_mod = getattr(par, attr)
        wrapper = FastLoRAWrapper(old_mod, lora_a=lora_a, lora_b=lora_b, scale=scale)
        setattr(par, attr, wrapper)
        mx.eval(wrapper.parameters())
        self._staged.append((
            lambda p=par, a=attr, o=old_mod: setattr(p, a, o),
            dict(kind=f"lora:{target}", layer=layer_idx, rank=int(lora_a.shape[1]), scale=scale, meta=meta or {})
        ))

    def apply_dt_bias_edit(self, delta, layer_indices=None, meta=None):
        """Edit Gated DeltaNet dt_bias (scalar, per-layer dict, or per-head vector)."""
        lin_idx = [i for i, b in enumerate(self.layers) if getattr(b, "is_linear", False)]
        if layer_indices is not None:
            lin_idx = [i for i in lin_idx if i in set(layer_indices)]
        if not lin_idx:
            raise ValueError("No linear-attention layers to edit")
        originals = []
        updated = []
        for i in lin_idx:
            b = self.layers[i]
            orig = b.linear_attn.dt_bias
            originals.append((b, orig))
            d = delta[i] if isinstance(delta, dict) else delta
            new_dt = (orig.astype(mx.float32) + mx.array(d, dtype=mx.float32)).astype(orig.dtype)
            b.linear_attn.dt_bias = new_dt
            updated.append(new_dt)
        mx.eval(updated)

        def _undo(origs=originals):
            for b, o in origs:
                b.linear_attn.dt_bias = o

        self._staged.append((
            _undo,
            dict(kind="dt_bias", n_layers=len(lin_idx), layers=lin_idx, meta=meta or {})
        ))

    def apply_steering_control(self, layer_idx, delta_vec, meta=None):
        """Inference-time activation steering control (TGAS-style)."""
        old_layer = self.layers[layer_idx]
        wrapper = ResidualSteeringWrapper(old_layer, delta_vec)
        self.layers[layer_idx] = wrapper
        mx.eval(wrapper.delta_vec)

        def _undo(idx=layer_idx, orig=old_layer):
            self.layers[idx] = orig

        self._staged.append((
            _undo,
            dict(kind="steering_control", layer=layer_idx, norm=float(np.linalg.norm(delta_vec)), meta=meta or {})
        ))

    def rollback_staged(self):
        """Atomically revert all uncommitted staged edits in LIFO order."""
        for undo, _ in reversed(self._staged):
            undo()
        self._staged.clear()
        mx.clear_cache()

    def commit_staged(self, tag="gen"):
        """Lock staged edits as a committed generation (still reversible via rollback_all)."""
        self._committed.append((tag, list(self._staged)))
        self._staged.clear()

    def rollback_all(self):
        """Revert ALL staged and committed edits back to exact base model."""
        self.rollback_staged()
        for _, entries in reversed(self._committed):
            for undo, _ in reversed(entries):
                undo()
        self._committed.clear()
        mx.clear_cache()

    def summary(self):
        return dict(
            committed=[dict(tag=t, edits=[m for _, m in es]) for t, es in self._committed],
            staged=[m for _, m in self._staged],
        )
