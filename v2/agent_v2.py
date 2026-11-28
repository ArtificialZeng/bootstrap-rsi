"""Autonomous Self-Rewriting Scientist v2 (fixes ALL 5 harness defects of agent8b.py + unlocks compiled primitives).

Fixes over agent8b.py:
  1. Read-only `tools` namespace object (impossible to accidentally overwrite tool functions like gen2_try2!).
  2. Safe `type`, `hasattr`, and `.format` restored in sandbox.
  3. Atomic transactional weight edits (`tools.rollback()`, automatic rollback on exception inside a code block!).
  4. Cached practice_base and reserved_base (zero redundant base recomputation in paired tests and gates!).
  5. Strict statistical gate (prevents Goodhart retention on noisy 40-problem checks) + exact W->C vs T->C tracking.
  6. Exposes training-free closed-form compilation primitives (MoE Router Self-Rewiring, Null-Space Ridge LoRA,
     Gated DeltaNet dt_bias, Context Internalization) alongside raw diagnostics, AND supports multi-generation
     recursive persistence (Gen g edits are inherited by Gen g+1!).
"""
import argparse
import ast
import builtins
import contextlib
import io
import json
import os
import pickle
import re
import sys
import time
import types
import numpy as np
import mlx.core as mx
from mlx_lm.generate import generate as single_generate
from mlx_lm.sample_utils import make_sampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from v2.gpu_lock import gpu_lock
from v2.benchmarks import load_benchmark
from v2.engine import (
    load_native_model, eval_problems, extract_layer_diagnostics, get_layers, is_moe_model, V2_OUT
)
from v2.compilers import (
    ModelEditor, compute_router_contrast, compile_router_bias, compile_router_nullspace_lora,
    compile_contrastive_direction, compile_nullspace_mlp_lora
)
from v2.stats import summarize_records, compare_paired, strict_gate_decision, format_comparison


# ---- Safe Sandbox (fixed: allows type, hasattr, .format; immutable tools facade) ----

SAFE_MODULES = {"numpy", "math", "statistics", "collections", "itertools", "re", "json", "functools", "random"}
SAFE_BUILTINS = [
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "format", "int",
    "isinstance", "issubclass", "type", "hasattr", "len", "list", "map", "max", "min", "print",
    "range", "repr", "reversed", "round", "set", "slice", "sorted", "str", "sum", "tuple", "zip",
    "divmod", "pow", "iter", "next", "callable", "chr", "ord", "frozenset", "complex",
    "Exception", "ValueError", "KeyError", "IndexError", "TypeError", "RuntimeError",
    "ZeroDivisionError", "AssertionError", "StopIteration", "NotImplementedError", "__build_class__",
]
BLOCK_ATTR = {
    "save", "savez", "savez_compressed", "savetxt", "load", "loadtxt", "genfromtxt", "fromfile",
    "tofile", "fromregex", "memmap", "DataSource", "npyio", "ctypeslib", "lib", "f2py", "testing",
    "distutils", "open_memmap",
}
BLOCK_NAME = {
    "eval", "exec", "compile", "getattr", "setattr", "delattr", "vars", "dir", "open",
    "globals", "locals", "breakpoint", "input", "memoryview",
}


def refuse_reason(code):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and (node.attr.startswith("__") or node.attr in BLOCK_ATTR):
            return f"attribute '{node.attr}' is not allowed"
        if isinstance(node, ast.Name) and (node.id.startswith("__") or node.id in BLOCK_NAME):
            return f"name '{node.id}' is not allowed"
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [al.name for al in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for m in mods:
                if m.split(".")[0] not in SAFE_MODULES:
                    return f"import of '{m}' is not allowed"
    if "__" in code:
        return "double-underscore names are not allowed"
    return None


def _safe_import(name, *a, **k):
    if name.split(".")[0] not in SAFE_MODULES:
        raise ImportError(f"import of '{name}' not allowed; allowed={sorted(SAFE_MODULES)}")
    return __import__(name, *a, **k)


def extract_python_block(text):
    """Extract the last ```python ... ``` block after </think> (or whole text)."""
    body = text.split("</think>")[-1] if "</think>" in text else text
    matches = re.findall(r"```(?:python)?\s*\n(.*?)```", body, flags=re.DOTALL)
    return matches[-1].strip() if matches else None


class ReadOnlyTools:
    """Immutable attribute facade so the model can NEVER shadow or overwrite tool functions."""
    def __init__(self, funcs):
        object.__setattr__(self, "_funcs", dict(funcs))

    def __getattr__(self, name):
        f = object.__getattribute__(self, "_funcs")
        if name in f:
            return f[name]
        raise AttributeError(f"tools has no method '{name}'. Available: {sorted(f)}")

    def __setattr__(self, name, value):
        raise AttributeError("tools is read-only; assign results to a normal variable instead")


class SelfRewriteEnvV2:
    """Clean, cached, atomic environment exposed via `tools.*` to the self-rewriting model."""

    def __init__(self, model, tok, editor, practice_probs, gate_probs, practice_base, gate_base,
                 cap=4096, bs=32, budget_units=8, strict_gate=True, eval_extra="", eval_thinking=True):
        self.model = model
        self.tok = tok
        self.editor = editor
        self.practice_probs = practice_probs
        self.gate_probs = gate_probs
        self.prac_by_pid = {p["pid"]: p for p in practice_probs}
        self.base_prac_by_pid = {r["pid"]: r for r in practice_base}
        self.practice_base = practice_base
        self.gate_base = gate_base
        self.cap = cap
        self.bs = bs
        self.budget_left = budget_units
        self.strict_gate = strict_gate
        self.eval_extra = eval_extra
        self.eval_thinking = eval_thinking
        self.nL = len(get_layers(model))
        self.is_moe = is_moe_model(model)
        self.has_linear = any(getattr(b, "is_linear", False) for b in get_layers(model))

        self.diag_cache = {}
        self.gate_result = None
        self.decision = None
        self.used_diagnostic = False
        self.history_log = []
        self._test_seed = 5000

    def _spend(self, cost, op):
        if self.budget_left < cost:
            raise RuntimeError(f"Budget exhausted ({self.budget_left} left, {op} needs {cost}). Call tools.submit() or tools.withdraw().")
        self.budget_left -= cost

    def status(self):
        st = summarize_records(self.practice_base, "practice_base")
        return dict(
            n_layers=self.nL,
            is_moe=self.is_moe,
            has_linear_attn=self.has_linear,
            budget_left=self.budget_left,
            practice_summary=dict(
                n=st["n"], acc=round(st["acc"], 4), trunc=round(st["trunc"], 4),
                n_C=st["n_C"], n_W=st["n_W"], n_T=st["n_T"],
                acc_finished=round(st["acc_finished"], 4),
            ),
            staged_edits=self.editor.summary()["staged"],
            committed_gens=len(self.editor.summary()["committed"]),
            gate_result=self.gate_result,
            decision=self.decision,
        )

    def list_pids(self, state="all"):
        """Return practice problem IDs filtered by state ('C', 'W', 'T', or 'all')."""
        if state == "all":
            return [r["pid"] for r in self.practice_base]
        return [r["pid"] for r in self.practice_base if r["state"] == state]

    def inspect_failure(self, pid, tail_chars=800):
        """Inspect problem statement, reference answer, and the model's base response tail for a practice pid."""
        if pid not in self.prac_by_pid:
            raise KeyError(f"Unknown practice pid '{pid}'")
        self.used_diagnostic = True
        p = self.prac_by_pid[pid]
        r = self.base_prac_by_pid[pid]
        return dict(
            pid=pid,
            state=r["state"],
            ntok=r["ntok"],
            looping=r["looping"],
            boxed=r["boxed"],
            gold=p["answer"],
            problem=p["problem"][:600],
            response_tail=r["text"][-tail_chars:],
        )

    def _get_diag(self, pos_pids, neg_pids, layers, window="commit", need_mlp=False):
        key = (tuple(pos_pids), tuple(neg_pids), tuple(sorted(layers)), window, need_mlp)
        if key in self.diag_cache:
            return self.diag_cache[key]
        recs = [self.base_prac_by_pid[p] for p in pos_pids] + [self.base_prac_by_pid[p] for p in neg_pids]
        # Extract under current committed weights (rollback staged temporarily if needed? No, read base_prac traces)
        d = extract_layer_diagnostics(
            self.model, recs, layers, tok=self.tok, window=window,
            max_seq_len=min(self.cap + 512, 6144), max_tokens_per_seq=40,
            collect_router=self.is_moe, collect_mlp_in=need_mlp, collect_residual=True,
        )
        pos_idx = list(range(len(pos_pids)))
        neg_idx = list(range(len(pos_pids), len(recs)))
        self.diag_cache[key] = (d, pos_idx, neg_idx)
        return d, pos_idx, neg_idx

    def scan_layers(self, contrast="CW", window="commit", step=4):
        """Scan layers to measure C-vs-W (or C-vs-T) MoE Router SNR and Residual Cohen's d. Costs 1 budget unit."""
        self.used_diagnostic = True
        pos_pids = self.list_pids("C")[:24]
        neg_pids = self.list_pids("W" if contrast == "CW" else "T")[:24]
        if len(pos_pids) < 3 or len(neg_pids) < 3:
            raise ValueError(f"Not enough {contrast} practice records: pos={len(pos_pids)}, neg={len(neg_pids)}")
        self._spend(1, "scan_layers")
        layers = list(range(2, self.nL - 1, step))
        d, pos_idx, neg_idx = self._get_diag(pos_pids, neg_pids, layers, window=window, need_mlp=False)
        out = {}
        for l in layers:
            cd = compile_contrastive_direction(d[l], pos_idx, neg_idx)
            entry = dict(cohen_d=round(cd["cohen_d"], 3), mean_diff_norm=round(cd["mean_norm"], 3))
            if self.is_moe and "router_logits" in d[l]:
                rc = compute_router_contrast(d[l], pos_idx, neg_idx)
                entry["router_snr"] = round(rc["layer_snr"], 3)
                entry["max_abs_t"] = round(float(np.max(np.abs(rc["t_stat"]))), 2)
            out[l] = entry
        self.history_log.append(dict(op="scan_layers", contrast=contrast, window=window, result=out))
        return out

    def compile_router_edit(self, layers, contrast="CW", mode="bias", alpha=0.4, top_m=32,
                            rank=8, lam_keep=4.0, window="commit"):
        """Compile a training-free MoE Router edit on `layers` from practice C vs W (or C vs T) trajectories."""
        if not self.is_moe:
            raise RuntimeError("This model is not an MoE model.")
        self.used_diagnostic = True
        layers = [int(l) for l in layers]
        pos_pids = self.list_pids("C")[:30]
        neg_pids = self.list_pids("W" if contrast == "CW" else "T")[:30]
        d, pos_idx, neg_idx = self._get_diag(pos_pids, neg_pids, layers, window=window, need_mlp=False)
        report = {}
        for l in layers:
            con = compute_router_contrast(d[l], pos_idx, neg_idx)
            if mode == "bias":
                b = compile_router_bias(con, alpha=float(alpha), mode="t_stat", top_m=int(top_m))
                self.editor.apply_router_edit(l, bias=b, meta=dict(contrast=contrast, mode="bias", alpha=alpha))
                report[l] = dict(snr=round(con["layer_snr"], 3), l2=round(float(np.linalg.norm(b)), 3))
            elif mode == "nullspace_lora":
                la, lb, rd = compile_router_nullspace_lora(
                    d[l], con, alpha=float(alpha), rank=int(rank), lam_keep=float(lam_keep), top_m=int(top_m)
                )
                self.editor.apply_router_edit(l, lora_a=la, lora_b=lb, meta=dict(contrast=contrast, mode="nullspace_lora", alpha=alpha))
                report[l] = dict(snr=round(con["layer_snr"], 3), selectivity=round(rd["selectivity"], 2))
            else:
                raise ValueError("mode must be 'bias' or 'nullspace_lora'")
        self.history_log.append(dict(op="compile_router_edit", layers=layers, contrast=contrast, mode=mode, alpha=alpha))
        return report

    def compile_nullspace_lora(self, layers, contrast="CW", alpha=0.4, rank=4, lam_keep=4.0, window="commit"):
        """Compile closed-form Null-Space Ridge LoRA on mlp.down_proj / shared_expert.down_proj."""
        self.used_diagnostic = True
        layers = [int(l) for l in layers]
        pos_pids = self.list_pids("C")[:30]
        neg_pids = self.list_pids("W" if contrast == "CW" else "T")[:30]
        d, pos_idx, neg_idx = self._get_diag(pos_pids, neg_pids, layers, window=window, need_mlp=True)
        report = {}
        for l in layers:
            la, lb, rd = compile_nullspace_mlp_lora(
                d[l], pos_idx, neg_idx, alpha=float(alpha), rank=int(rank), lam_keep=float(lam_keep)
            )
            self.editor.apply_lora_edit(l, "mlp_down", la, lb, meta=dict(contrast=contrast, alpha=alpha, rank=rank))
            report[l] = dict(cohen_d=round(rd["cohen_d"], 3), selectivity=round(rd["selectivity"], 2),
                             frob=round(rd["frob_norm"], 4))
        self.history_log.append(dict(op="compile_nullspace_lora", layers=layers, contrast=contrast, alpha=alpha))
        return report

    def edit_dt_bias(self, delta, layers=None):
        """Shift Gated DeltaNet dt_bias by scalar `delta` on linear-attention layers."""
        if not self.has_linear:
            raise RuntimeError("This model has no Gated DeltaNet linear-attention layers.")
        self.editor.apply_dt_bias_edit(float(delta), layer_indices=layers, meta=dict(delta=delta))
        self.history_log.append(dict(op="edit_dt_bias", delta=delta, layers=layers))
        return dict(applied="dt_bias", delta=delta, staged_count=len(self.editor.summary()["staged"]))

    def rollback(self):
        """Atomically undo all uncommitted staged edits."""
        self.editor.rollback_staged()
        return "staged edits rolled back"

    def test_practice(self, pids):
        """Evaluate staged edits on up to 24 practice problems, comparing against CACHED practice_base! Costs 1 unit."""
        pids = list(pids)
        if not pids or len(pids) > 24:
            raise ValueError(f"Pass 1..24 practice pids (got {len(pids)})")
        probs = [self.prac_by_pid[p] for p in pids]
        self._spend(1, "test_practice")
        self._test_seed += 1
        edit_recs = eval_problems(self.model, self.tok, probs, self.cap, seed=self._test_seed,
                                  bs=min(self.bs, len(probs)), extra=self.eval_extra,
                                  enable_thinking=self.eval_thinking, verbose=False)
        base_sub = [self.base_prac_by_pid[p] for p in pids]
        c = compare_paired(edit_recs, base_sub)
        out = dict(
            n=c["n"], base_acc=round(c["base_acc"], 4), edit_acc=round(c["edit_acc"], 4),
            delta=round(c["delta"], 4), wins=c["wins"], losses=c["losses"],
            tc=c["tc"], ct=c["ct"], wc=c["wc"], cw=c["cw"],
            delta_completion=round(c["delta_completion"], 4),
            delta_reasoning=round(c["delta_reasoning"], 4),
        )
        self.history_log.append(dict(op="test_practice", result=out))
        return out

    def submit(self):
        """Submit staged edits to the STRICT held-out gate (cached gate_base). One-shot per session!"""
        if self.decision is not None:
            raise RuntimeError(f"Already decided: {self.decision}")
        if not self.editor.summary()["staged"]:
            raise RuntimeError("No staged edits to submit; call tools.withdraw() instead.")
        if self.gate_result is not None:
            return self.gate_result
        edit_gate = eval_problems(self.model, self.tok, self.gate_probs, self.cap, seed=1000, bs=self.bs,
                                  extra=self.eval_extra, enable_thinking=self.eval_thinking, verbose=True)
        comp = compare_paired(edit_gate, self.gate_base)
        gate = strict_gate_decision(comp, min_delta=0.02, min_z=1.5, max_p_one=0.07) if self.strict_gate else dict(
            passed=comp["delta"] > 0, z=0.0, delta=comp["delta"], p_one=comp["p_sign_one"]
        )
        self.gate_result = dict(
            passed=gate["passed"],
            delta=round(comp["delta"], 4),
            se=round(comp["se"], 4),
            z=round(gate["z"], 2),
            p_one=round(gate["p_one"], 4),
            wins=comp["wins"], losses=comp["losses"],
            tc=comp["tc"], ct=comp["ct"], wc=comp["wc"], cw=comp["cw"],
            delta_completion=round(comp["delta_completion"], 4),
            delta_reasoning=round(comp["delta_reasoning"], 4),
        )
        if gate["passed"]:
            self.decision = "accepted"
        else:
            self.decision = "rejected_by_gate"
            self.editor.rollback_staged()
        return self.gate_result

    def withdraw(self):
        if self.decision is not None:
            raise RuntimeError(f"Already decided: {self.decision}")
        self.editor.rollback_staged()
        self.decision = "withdrawn"
        return "withdrawn and rolled back"

    def make_namespace(self):
        bi = {k: getattr(builtins, k) for k in SAFE_BUILTINS}
        bi["__import__"] = _safe_import
        tools = ReadOnlyTools({
            "status": self.status,
            "list_pids": self.list_pids,
            "inspect_failure": self.inspect_failure,
            "scan_layers": self.scan_layers,
            "compile_router_edit": self.compile_router_edit,
            "compile_nullspace_lora": self.compile_nullspace_lora,
            "edit_dt_bias": self.edit_dt_bias,
            "rollback": self.rollback,
            "test_practice": self.test_practice,
            "submit": self.submit,
            "withdraw": self.withdraw,
        })
        return {"__builtins__": bi, "__name__": "session", "np": np, "tools": tools}


SYSTEM_PROMPT_V2 = """You are an autonomous ML scientist improving YOUR OWN neural network weights WITHOUT gradient descent.
In each turn, output a brief hypothesis followed by ONE ```python ... ``` code block using the read-only `tools` object.

Key scientific objective:
- Break beyond mere truncation reduction (T->C) and achieve genuine finished-reasoning accuracy gains (W->C) OR orthogonal multi-generation compounding.
- ALWAYS inspect your failures and scan layers BEFORE editing. NEVER apply blind random edits.
- When you submit (`tools.submit()`), the framework runs a strict statistical gate on unseen reserved problems. Only statistically meaningful gains pass.

Available methods on `tools`:
- `tools.status()` : model architecture, practice C/W/T counts, budget left, staged edits
- `tools.list_pids(state="all"|"C"|"W"|"T")` : practice problem IDs by outcome
- `tools.inspect_failure(pid)` : read problem, gold answer, and your failed response tail
- `tools.scan_layers(contrast="CW"|"CT", window="commit"|"early"|"mid"|"late"|"pivot", step=4)` : measure per-layer MoE Router SNR & Residual Cohen's d (costs 1 unit)
- `tools.compile_router_edit(layers, contrast="CW"|"CT", mode="bias"|"nullspace_lora", alpha=0.4, top_m=32, rank=8, lam_keep=4.0, window="commit")` : training-free MoE Router Self-Rewiring (MoE models)
- `tools.compile_nullspace_lora(layers, contrast="CW"|"CT", alpha=0.4, rank=4, lam_keep=4.0, window="commit")` : closed-form Null-Space Ridge LoRA on MLP down_proj / shared_expert
- `tools.edit_dt_bias(delta, layers=None)` : edit Gated DeltaNet dt_bias on linear-attention layers
- `tools.rollback()` : atomically undo staged edits
- `tools.test_practice(pids)` : fast paired test on 1..24 practice pids vs cached base (costs 1 unit)
- `tools.submit()` : run strict held-out gate & finalize (accepted if passes strict statistical gate, else rejected)
- `tools.withdraw()` : rollback and end session without submitting

Rules:
- `tools` is read-only; NEVER assign to `tools.*`.
- If a code block raises an error, any staged edits created inside that block are automatically rolled back.
- Keep your reasoning concise and decisive. Aim to diagnose, test 1-2 well-motivated edits, and submit within 6-10 turns.
"""


def run_self_session(model, tok, env, max_turns=12, turn_max_tokens=2048, session_seed=777):
    """Run one autonomous self-rewriting session where the model edits its own weights via `tools.*`."""
    ns = env.make_namespace()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_V2},
        {"role": "user", "content": f"Begin your self-improvement session. Current status:\n{json.dumps(env.status(), indent=2)}"},
    ]
    transcript = []
    n_errors = 0
    sampler = make_sampler(temp=0.6, top_p=0.95, top_k=20)

    for turn in range(1, max_turns + 1):
        mx.random.seed(session_seed + turn)
        t0 = time.time()
        try:
            prompt_str = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=True)
        except TypeError:
            prompt_str = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        reply = single_generate(model, tok, prompt=prompt_str, max_tokens=turn_max_tokens, sampler=sampler, verbose=False)
        dt = time.time() - t0
        code = extract_python_block(reply)
        messages.append({"role": "assistant", "content": reply})

        if not code:
            obs = "No ```python``` code block found. Write one python block calling `tools.*`."
            n_errors += 1
            transcript.append(dict(turn=turn, seconds=round(dt, 1), code=None, error=True, obs=obs))
            messages.append({"role": "user", "content": obs})
            continue

        ref = refuse_reason(code)
        if ref:
            obs = f"Sandbox refused code: {ref}"
            n_errors += 1
            transcript.append(dict(turn=turn, seconds=round(dt, 1), code=code, error=True, obs=obs))
            messages.append({"role": "user", "content": obs})
            continue

        # Atomic transaction per turn: remember staged count so partial edits rollback cleanly on error!
        n_staged_before = len(env.editor._staged)
        buf = io.StringIO()
        err_flag = False
        try:
            tree = ast.parse(code, mode="exec")
            last_expr = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last_expr = ast.Expression(tree.body.pop().value)
            with contextlib.redirect_stdout(buf):
                if tree.body:
                    exec(compile(tree, "<session>", "exec"), ns)
                if last_expr is not None:
                    val = eval(compile(last_expr, "<session>", "eval"), ns)
                    if val is not None:
                        print(repr(val))
            obs = buf.getvalue().strip() or "(ok, no output)"
        except Exception as e:
            err_flag = True
            n_errors += 1
            # Rollback ONLY edits staged during this failed turn!
            while len(env.editor._staged) > n_staged_before:
                undo, _ = env.editor._staged.pop()
                undo()
            obs = f"ERROR ({type(e).__name__}): {e}\n(Any partial edits from this turn were atomically rolled back.)"

        if len(obs) > 3000:
            obs = obs[:3000] + "\n...[truncated]"
        print(f"  [agent_v2 t{turn}] {dt:.1f}s err={err_flag} budget={env.budget_left} "
              f"staged={len(env.editor._staged)} dec={env.decision}", flush=True)
        transcript.append(dict(turn=turn, seconds=round(dt, 1), code=code, error=err_flag, obs=obs))
        messages.append({"role": "user", "content": f"Execution output:\n{obs}"})
        if env.decision is not None:
            break

    err_rate = n_errors / max(1, len(transcript))
    return dict(
        decision=env.decision,
        gate_result=env.gate_result,
        used_diagnostic=env.used_diagnostic,
        n_turns=len(transcript),
        n_errors=n_errors,
        error_rate=round(err_rate, 3),
        edits=env.editor.summary(),
        history=env.history_log,
        transcript=transcript,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="q35-35b", choices=["q35-9b", "q35-27b", "q35-35b"])
    ap.add_argument("--bench", default="supergpqa_brief", choices=["aime", "oly_math", "supergpqa_brief"])
    ap.add_argument("--cap", type=int, default=768)
    ap.add_argument("--n-prac", type=int, default=64)
    ap.add_argument("--n-gate", type=int, default=64)
    ap.add_argument("--n-eval", type=int, default=128)
    ap.add_argument("--gens", type=int, default=3)
    ap.add_argument("--attempts", type=int, default=2)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--strict-gate", action="store_true", default=True)
    ap.add_argument("--loose-gate", action="store_true", help="Ablation: unthresholded gate to measure Retention Illusion")
    args = ap.parse_args()
    strict = not args.loose_gate

    BRIEF_EXTRA = "\nBe concise: give only a brief 3-5 sentence justification (under 200 words) and immediately put your final single capital letter in \\boxed{}."
    eval_extra = BRIEF_EXTRA if args.bench == "supergpqa_brief" else ""
    eval_thinking = False if args.bench == "supergpqa_brief" else True

    tag = f"agentv2_{args.model}_{args.bench}_c{args.cap}_{'strict' if strict else 'loose'}"
    run_dir = os.path.join(V2_OUT, tag)
    os.makedirs(run_dir, exist_ok=True)

    with gpu_lock(tag, wait=True):
        model, tok = load_native_model(args.model)
        ed = ModelEditor(model)
        if args.bench == "supergpqa_brief":
            prac_probs = load_benchmark("supergpqa_sci_hard_practice")[:args.n_prac]
            gate_probs = load_benchmark("supergpqa_sci_hard_gate")[:args.n_gate]
            eval_probs = load_benchmark("supergpqa_sci_hard_test")[:args.n_eval]
        else:
            prac_probs = load_benchmark("aime_practice" if args.bench == "aime" else "oly_math_practice")[:args.n_prac]
            gate_probs = load_benchmark("aime_gate" if args.bench == "aime" else "oly_math_eval")[:args.n_gate]
            eval_probs = load_benchmark("aime_test" if args.bench == "aime" else "oly_phys")[:args.n_eval]

        gen_prac = eval_problems(model, tok, prac_probs, args.cap, seed=100, bs=args.bs,
                                 extra=eval_extra, enable_thinking=eval_thinking,
                                 cache_path=os.path.join(run_dir, "gen0_prac.pkl"))
        gen_gate = eval_problems(model, tok, gate_probs, args.cap, seed=1000, bs=args.bs,
                                 extra=eval_extra, enable_thinking=eval_thinking,
                                 cache_path=os.path.join(run_dir, "gen0_gate.pkl"))
        gen0_eval_s1 = eval_problems(model, tok, eval_probs, args.cap, seed=1000, bs=args.bs,
                                     extra=eval_extra, enable_thinking=eval_thinking,
                                     cache_path=os.path.join(run_dir, "gen0_eval_s1000.pkl"))
        gen0_eval_s2 = eval_problems(model, tok, eval_probs, args.cap, seed=2000, bs=args.bs,
                                     extra=eval_extra, enable_thinking=eval_thinking,
                                     cache_path=os.path.join(run_dir, "gen0_eval_s2000.pkl"))

        chain_log = dict(tag=tag, model=args.model, strict_gate=strict, generations=[])
        for g in range(1, args.gens + 1):
            kept = False
            for att in range(1, args.attempts + 1):
                print(f"\n=== {tag}: Generation {g}, Attempt {att} ===", flush=True)
                env = SelfRewriteEnvV2(model, tok, ed, prac_probs, gate_probs, gen_prac, gen_gate,
                                      cap=args.cap, bs=args.bs, budget_units=8, strict_gate=strict,
                                      eval_extra=eval_extra, eval_thinking=eval_thinking)
                sess = run_self_session(model, tok, env, max_turns=10, turn_max_tokens=1536, session_seed=7000 + 100 * g + att)
                json.dump(sess, open(os.path.join(run_dir, f"sess_g{g}_a{att}.json"), "w"), indent=2, default=str)
                chain_log["generations"].append(dict(gen=g, attempt=att, decision=sess["decision"],
                                                     gate=sess["gate_result"], error_rate=sess["error_rate"],
                                                     n_turns=sess["n_turns"], n_errors=sess["n_errors"],
                                                     used_diagnostic=sess["used_diagnostic"]))
                if sess["decision"] == "accepted":
                    ed.commit_staged(tag=f"gen{g}")
                    kept = True
                    # Refresh practice & gate baseline under new inherited weights for Gen g+1!
                    gen_prac = eval_problems(model, tok, prac_probs, args.cap, seed=100, bs=args.bs,
                                             extra=eval_extra, enable_thinking=eval_thinking,
                                             cache_path=os.path.join(run_dir, f"gen{g}_prac.pkl"))
                    gen_gate = eval_problems(model, tok, gate_probs, args.cap, seed=1000, bs=args.bs,
                                             extra=eval_extra, enable_thinking=eval_thinking,
                                             cache_path=os.path.join(run_dir, f"gen{g}_gate.pkl"))
                    break
                else:
                    ed.rollback_staged()
            if not kept:
                print(f"[{tag}] Chain stopped cleanly at generation {g} (no edit passed gate).", flush=True)
                break

        # Final held-out multi-seed evaluation of last committed generation (two-stage funnel!)
        n_kept = len(ed._committed)
        if n_kept > 0:
            ev1 = eval_problems(model, tok, eval_probs, args.cap, seed=1000, bs=args.bs,
                                extra=eval_extra, enable_thinking=eval_thinking,
                                cache_path=os.path.join(run_dir, f"gen{n_kept}_eval_s1000.pkl"))
            ev2 = eval_problems(model, tok, eval_probs, args.cap, seed=2000, bs=args.bs,
                                extra=eval_extra, enable_thinking=eval_thinking,
                                cache_path=os.path.join(run_dir, f"gen{n_kept}_eval_s2000.pkl"))
            from v2.stats import compare_multiseed
            comp = compare_multiseed([ev1, ev2], [gen0_eval_s1, gen0_eval_s2])
            chain_log["final_heldout"] = comp
            print("\nFINAL HELD-OUT:", format_comparison(f"{tag}_gen{n_kept}", comp), flush=True)
        chain_log["final_edits"] = ed.summary()
        json.dump(chain_log, open(os.path.join(run_dir, "chain_summary.json"), "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
