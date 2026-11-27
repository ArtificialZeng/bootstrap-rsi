"""Self-rewriting at 8B, small scale.

Qwen3-8B edits a copy of its own weights through a persistent Python session, following the
paper's second protocol (App. E.9): edit -> one framework check on reserved problems -> the model
keeps (confirm) or discards (withdraw) the edit.  The model always reasons with its unedited
weights; tools run the edited copy.  No gradients, no optimizer.

All projection matrices are dequantized from the MLX 8-bit checkpoint to bf16 once at load, so the
base arm and the edited arm run the same kernels (review BUG-03).

Phases (cached in out/8b_*):
  practice  base answers on the practice pool (the agent reads them)
  evalbase  base on the held-out eval set
  script    non-agentic baseline: fixed termination-direction ablation (recipe from the 1.7B sweep)
  reseed    noise floor: the unedited model with another seed, scored like an edit
  session   N agent sessions, all reported; each kept edit is evaluated on the held-out set
  chain     recursive self-improvement: generation g starts from the weights generation g-1 kept, so the model
            that reasons, diagnoses and edits is itself the product of its earlier edits (--chain G --attempts K)
"""
import sys, os, re, io, ast, json, math, time, glob, signal, pickle, contextlib, traceback, argparse, builtins, zlib
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.generate import generate as lm_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models.base import create_attention_mask
from scipy.stats import binomtest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_data, build_prompts, generate, grade, summarize, OUT, HF

MODELS = {"8b": ("models--mlx-community--Qwen3-8B-8bit", "Qwen3-8B"),
          "14b": ("models--mlx-community--Qwen3-14B-8bit", "Qwen3-14B")}
MTAG = "8b"  # --model: prefix of the results key and of every output file
BS = 64  # --bs: answers generated at once when a problem set is scored
PROJ = {"self_attn": ["q_proj", "k_proj", "v_proj", "o_proj"], "mlp": ["gate_proj", "up_proj", "down_proj"]}


def dequantize_projections(model):
    """Replace every QuantizedLinear projection with a bf16 nn.Linear holding the dequantized weight."""
    n = 0
    for blk in model.model.layers:
        for sub, names in PROJ.items():
            par = getattr(blk, sub)
            for a in names:
                m = getattr(par, a)
                if isinstance(m, nn.QuantizedLinear):
                    w = mx.dequantize(m.weight, m.scales, m.biases, group_size=m.group_size, bits=m.bits)
                    lin = nn.Linear(w.shape[1], w.shape[0], bias=False)
                    lin.weight = w.astype(mx.bfloat16)
                    mx.eval(lin.weight)
                    setattr(par, a, lin)
                    n += 1
    return n


# --------------------------------------------------------------------------- weights
class Subject:
    """Edits live as separate nn.Linear modules; they are installed only inside edited()."""

    def __init__(self, model):
        self.model = model
        self.orig = {}  # path -> (parent, attr, original module), snapshotted once
        for i, blk in enumerate(model.model.layers):
            for sub, names in PROJ.items():
                par = getattr(blk, sub)
                for a in names:
                    self.orig[f"layers.{i}.{sub}.{a}"] = (par, a, getattr(par, a))
        self.edits, self.rel = {}, {}
        self.active = False
        self.locked = False

    def _check(self, path):
        if path not in self.orig:
            raise ValueError(f"'{path}' is not editable. Editable: layers.<0-{len(self.model.model.layers) - 1}>."
                             "self_attn.<q|k|v|o>_proj and layers.<i>.mlp.<gate|up|down>_proj")

    def base_weight(self, path):
        self._check(path)
        return np.array(self.orig[path][2].weight.astype(mx.float32))

    def get(self, path):
        self._check(path)
        if path in self.edits:
            return np.array(self.edits[path].weight.astype(mx.float32))
        return self.base_weight(path)

    def _guard(self):
        if self.locked:
            raise RuntimeError("weights are locked after the check; call confirm() or withdraw()")
        if self.active:
            raise RuntimeError("cannot change edits while they are installed")

    def set(self, path, W):
        self._check(path)
        self._guard()
        W = np.asarray(W, dtype=np.float32)
        B0 = self.base_weight(path)
        if W.shape != B0.shape:
            raise ValueError(f"shape {W.shape} != {B0.shape}")
        if not np.isfinite(W).all():
            raise ValueError("weights contain NaN/inf")
        c = float((W * B0).sum() / (B0 * B0).sum())
        if abs(c - 1) > 1e-4 and np.linalg.norm(W - c * B0) / np.linalg.norm(B0) < 1e-4:
            raise ValueError("refused: an edit that only rescales a whole tensor is not allowed")
        lin = nn.Linear(W.shape[1], W.shape[0], bias=False)
        lin.weight = mx.array(W).astype(mx.bfloat16)
        mx.eval(lin.weight)
        self.edits[path] = lin
        self.rel[path] = float(np.linalg.norm(W - B0) / np.linalg.norm(B0))
        return self.rel[path]

    def reset(self, path=None):
        self._guard()
        if path is None:
            self.edits.clear(); self.rel.clear(); return
        self._check(path)
        if path not in self.edits:
            raise ValueError(f"no edit on '{path}'. Current edits: {sorted(self.edits)}")
        self.edits.pop(path); self.rel.pop(path)

    def summary(self):
        return {p: round(r, 6) for p, r in self.rel.items()}

    @contextlib.contextmanager
    def edited(self):
        if self.active:
            raise RuntimeError("edited() is not re-entrant")
        self.active = True
        installed = []
        try:
            for p, lin in self.edits.items():
                par, a, _ = self.orig[p]
                setattr(par, a, lin)
                installed.append(p)
            yield
        finally:
            for p in installed:
                par, a, m = self.orig[p]
                setattr(par, a, m)
            self.active = False

    def assert_clean(self):
        bad = [p for p, (par, a, m) in self.orig.items() if getattr(par, a) is not m]
        if bad or self.active:
            raise RuntimeError(f"editor would run with edited weights: {bad[:3]}")

    def save(self, path):
        if self.edits:
            mx.save_safetensors(path, {p: l.weight for p, l in self.edits.items()})


def hidden_multi(model, ids, start, layers, max_len=6000):
    """Mean residual state over positions >= start, after each layer in `layers`."""
    ids = list(ids)[:max_len]
    if start >= len(ids):
        raise ValueError("nothing to average: the text is empty or longer prompt than the 6000-token window")
    x = mx.array(ids)[None]
    m = model.model
    h = m.embed_tokens(x)
    mask = create_attention_mask(h, None)
    out, Lmax = {}, max(layers)
    for i, blk in enumerate(m.layers):
        h = blk(h, mask, None)
        if i in layers:
            out[i] = h[0, start:].astype(mx.float32).mean(axis=0)
        if i == Lmax:
            break
    mx.eval(list(out.values()))
    return {k: np.array(v) for k, v in out.items()}


def is_looping(text, truncated):
    tail = text[-6000:].encode()
    return bool(truncated and tail and len(zlib.compress(tail)) < 0.15 * len(tail))


def gen_grade(model, tok, probs, cap, seed, verbose=True):
    prompts = build_prompts(tok, probs)
    texts, ids = generate(model, tok, prompts, cap, seed=seed, bs=BS, verbose=verbose)
    print(f"  peak memory {mx.get_peak_memory() / 2**30:.1f} GB", file=sys.stderr, flush=True)
    recs = []
    for p, pr, t, i in zip(probs, prompts, texts, ids):
        g = grade(t, len(i), cap, p["answer"])
        g["looping"] = is_looping(t, g["truncated"])
        g.update(pid=p["unique_id"], level=p["level"], text=t, prompt_ids=pr, out_ids=i)
        recs.append(g)
    return recs


def acc_f(R):
    F = [r for r in R if not r["truncated"]]
    return float(np.mean([r["correct"] for r in F])) if F else float("nan")


def paired_stats(base, edit):
    """base/edit: records aligned by problem, one answer each."""
    assert len(base) == len(edit) and [r["pid"] for r in base] == [r["pid"] for r in edit], "misaligned pairs"
    d = np.array([int(e["correct"]) - int(b["correct"]) for b, e in zip(base, edit)], dtype=float)
    n = len(d)
    se = float(np.sqrt(((d - d.mean()) ** 2).sum() / (n * (n - 1)))) if n > 1 else float("nan")
    up, down = int((d > 0).sum()), int((d < 0).sum())
    p = binomtest(up, up + down, 0.5, alternative="greater").pvalue if up + down else 1.0
    mean = lambda R, k: float(np.mean([r[k] for r in R]))
    return dict(n=n, base_acc=mean(base, "correct"), edit_acc=mean(edit, "correct"), delta=float(d.mean()), se=se,
                wins=up, losses=down, p_sign_one_sided=float(p),
                base_trunc=mean(base, "truncated"), edit_trunc=mean(edit, "truncated"),
                base_looping=mean(base, "looping"), edit_looping=mean(edit, "looping"),
                base_acc_finished=acc_f(base), edit_acc_finished=acc_f(edit))


# --------------------------------------------------------------------------- python session
class BudgetError(Exception):
    pass


class _Timeout(BaseException):  # BaseException: cannot be swallowed by `except Exception` in session code
    pass


def _alarm_handler(*_):
    raise _Timeout()


_CLOCK = {"deadline": None}  # monotonic deadline of the running code block; None outside exec_code


def _arm():
    rem = _CLOCK["deadline"] - time.monotonic()
    if rem <= 0:
        raise _Timeout()
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, rem)


def _tool(fn):
    """Forward passes and other framework work do not count against the code-run time limit:
    the deadline is pushed back by exactly the time spent inside the tool."""
    def w(*a, **k):
        if _CLOCK["deadline"] is None:
            return fn(*a, **k)
        signal.setitimer(signal.ITIMER_REAL, 0)
        t = time.monotonic()
        try:
            return fn(*a, **k)
        finally:
            _CLOCK["deadline"] += time.monotonic() - t
            _arm()
    return w


class Env:
    def __init__(self, model, tok, subj, practice, base_prac, cfg):
        self.model, self.tok, self.subj, self.cfg = model, tok, subj, cfg
        self.practice = {p["unique_id"]: p for p in practice}
        self.base_prac = {r["pid"]: r for r in base_prac}
        self.B_left = cfg.B
        self.want_check = False
        self.checked = None
        self.decision = None
        self.final = False
        self.seed = 5000
        self.events = []
        self.last = {}  # (pid, "edited"/"base") -> text of the latest run
        self.notes = ""

    def _spend(self, n, what):
        if self.final or self.checked is not None or self.want_check:
            raise RuntimeError("no more forward passes: only confirm()/withdraw() are allowed now")
        if self.B_left < n:
            raise BudgetError(f"{what} needs {n} forward-pass unit but only {self.B_left} are left")
        self.B_left -= n
        self.events.append(dict(t=time.time(), tool=what, cost=n))

    def _ids(self, ids, k):
        ids = list(ids)
        bad = [i for i in ids if i not in self.practice]
        if bad:
            raise ValueError(f"unknown practice ids: {bad[:3]}; ids are the strings in practice_ids, e.g. {next(iter(self.practice))!r}")
        if not ids or len(ids) > k:
            raise ValueError(f"this call got {len(ids)} ids; pass between 1 and {k} ids per call")
        return ids

    # ---- weights
    def get_weight(self, name):
        return self.subj.get(name)

    def _no_final(self):
        if self.final:
            raise RuntimeError("time is up: only submit()/confirm()/withdraw() are allowed now")

    def set_weight(self, name, W):
        self._no_final()
        return f"set {name}: relative change {self.subj.set(name, W):.5f}"

    def weaken(self, name, u, alpha):
        W = self.subj.get(name)
        u = np.asarray(u, dtype=np.float64).ravel()
        if u.shape[0] != W.shape[0]:
            raise ValueError(f"u has length {u.shape[0]} but {name} writes {W.shape[0]} dims")
        u = u / (np.linalg.norm(u) + 1e-12)
        return self.set_weight(name, W - alpha * np.outer(u, u @ W))

    def reset(self, name=None):
        self._no_final()
        self.subj.reset(name)
        return "reset " + (name or "all edits")

    def current_edits(self):
        return self.subj.summary()

    # ---- reading (free)
    def answer_text(self, pid, which="base", head=1500, tail=2500):
        """which: 'base' (starting diagnosis) or 'last_edited' / 'last_base' (latest run of that problem)."""
        if which == "base":
            t = self.base_prac[pid]["text"]
        else:
            key = (pid, which.replace("last_", ""))
            if key not in self.last:
                raise ValueError(f"no {which} answer for {pid} yet")
            t = self.last[key]
        return t if len(t) <= head + tail else t[:head] + f"\n... [{len(t) - head - tail} chars omitted] ...\n" + t[-tail:]

    def reference(self, pid):
        return self.practice[pid]["answer"]

    def write_notes(self, text):
        self.notes = str(text)[:3000]
        return "notes saved"

    # ---- forward passes
    def _gen(self, probs, edited, seed):
        ctx = self.subj.edited() if edited else contextlib.nullcontext()
        with ctx:
            recs = gen_grade(self.model, self.tok, probs, self.cfg.cap, seed, verbose=False)
        for r in recs:
            self.last[(r["pid"], "edited" if edited else "base")] = r["text"]
        return recs

    def run(self, ids, edited=True):
        ids = self._ids(ids, self.cfg.per_pass)
        self._spend(1, "run")
        self.seed += 1
        recs = self._gen([self.practice[i] for i in ids], edited, self.seed)
        return [dict(id=r["pid"], correct=r["correct"], truncated=r["truncated"], looping=r["looping"],
                     ntok=r["ntok"], ending=r["text"][-300:]) for r in recs]

    def paired_test(self, ids):
        ids = self._ids(ids, self.cfg.per_pass)
        self._spend(1, "paired_test")
        self.seed += 1
        probs = [self.practice[i] for i in ids]
        base = self._gen(probs, False, self.seed)
        edit = self._gen(probs, True, self.seed)
        s = paired_stats(base, edit)
        s["per_problem (id, base_correct, edit_correct, base_trunc, edit_trunc)"] = [
            (b["pid"], int(b["correct"]), int(e["correct"]), int(b["truncated"]), int(e["truncated"]))
            for b, e in zip(base, edit)]
        return s

    def hidden_states(self, ids=None, layers=(), texts=None):
        nL = len(self.model.model.layers)
        layers = sorted(set(int(l) for l in layers))
        if not layers or layers[0] < 0 or layers[-1] >= nL:
            raise ValueError(f"layers must be in 0..{nL - 1}")
        items = []
        for i in (self._ids(ids, 64) if ids is not None else []):
            r = self.base_prac[i]
            items.append((list(r["prompt_ids"]) + list(r["out_ids"]), len(r["prompt_ids"])))
        for t in (texts or []):
            items.append((self.tok.encode(str(t)), 0))
        if not items or len(items) > 64:
            raise ValueError("give 1..64 ids/texts in total")
        self._spend(1, "hidden_states")
        out = {l: [] for l in layers}
        with self.subj.edited():
            for seq, start in items:
                h = hidden_multi(self.model, seq, start, layers)
                for l in layers:
                    out[l].append(h[l])
        return {l: np.stack(v) for l, v in out.items()}

    # ---- decision (protocol 2)
    def submit(self):
        if self.decision:
            raise RuntimeError(f"already decided: {self.decision}")
        if not self.subj.edits:
            raise RuntimeError("no edits to submit; call withdraw() to end without an edit")
        if self.checked is not None:
            return self.checked
        self.want_check = True
        self.subj.locked = True
        return "check requested: edits are locked; the framework runs the check after this code block"

    def confirm(self):
        if self.checked is None:
            raise RuntimeError("call submit() first; the framework then runs the one-time check")
        if "error" in self.checked:
            raise RuntimeError("the check failed, so there is nothing to confirm; call withdraw()")
        if self.decision:
            raise RuntimeError(f"already decided: {self.decision}")
        self.decision = "submit"
        return "submitted"

    def withdraw(self):
        if self.decision:
            raise RuntimeError(f"already decided: {self.decision}")
        self.decision = "withdraw"
        self.want_check = False
        return "withdrawn"


SAFE_MODULES = {"numpy", "math", "statistics", "collections", "itertools", "re", "json", "functools", "random"}
SAFE_BUILTINS = ["abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "format", "int", "isinstance",
                 "len", "list", "map", "max", "min", "print", "range", "repr", "reversed", "round", "set", "slice",
                 "sorted", "str", "sum", "tuple", "zip", "divmod", "pow", "iter", "next", "callable", "chr", "ord",
                 "frozenset", "complex", "Exception", "ValueError", "KeyError", "IndexError", "TypeError",
                 "RuntimeError", "ZeroDivisionError", "AssertionError", "StopIteration", "NotImplementedError",
                 "__build_class__"]
BLOCK_ATTR = {"save", "savez", "savez_compressed", "savetxt", "load", "loadtxt", "genfromtxt", "fromfile", "tofile",
              "fromregex", "memmap", "DataSource", "npyio", "ctypeslib", "lib", "f2py", "testing", "distutils",
              "open_memmap", "format"}
BLOCK_NAME = {"eval", "exec", "compile", "getattr", "setattr", "delattr", "vars", "dir", "open", "globals", "locals",
              "breakpoint", "input", "memoryview"}


def refuse_reason(code):
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return None  # let exec report the syntax error
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and (node.attr.startswith("__") or node.attr in BLOCK_ATTR):
            return f"attribute '{node.attr}' is not allowed"
        if isinstance(node, ast.Name) and (node.id.startswith("__") or node.id in BLOCK_NAME):
            return f"name '{node.id}' is not allowed"
        if isinstance(node, ast.ExceptHandler) and (node.type is None or
                                                    (isinstance(node.type, ast.Name) and node.type.id == "BaseException")):
            return "bare except / except BaseException is not allowed"
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [al.name for al in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for m in mods:
                if m.split(".")[0] not in SAFE_MODULES or any(x in m for x in ("ctypeslib", "lib", "testing")):
                    return f"import of '{m}' is not allowed"
    if "__" in code:
        return "double-underscore names are not allowed"
    return None


def _safe_import(name, *a, **k):
    if name.split(".")[0] not in SAFE_MODULES or "ctypeslib" in name:
        raise ImportError(f"import of '{name}' is not allowed (allowed: {sorted(SAFE_MODULES)})")
    return __import__(name, *a, **k)


def make_namespace(env):
    bi = {k: getattr(builtins, k) for k in SAFE_BUILTINS}
    bi["__import__"] = _safe_import
    ns = {"__builtins__": bi, "__name__": "session", "np": np}
    for f in ["get_weight", "set_weight", "weaken", "reset", "current_edits", "answer_text", "reference",
              "write_notes", "run", "paired_test", "hidden_states", "submit", "confirm", "withdraw"]:
        ns[f] = _tool(getattr(env, f))
    ns["practice_ids"] = list(env.practice)
    ns["base_results"] = {pid: dict(correct=r["correct"], truncated=r["truncated"], looping=r["looping"],
                                    ntok=r["ntok"]) for pid, r in env.base_prac.items()}
    ns["problem"] = _tool(lambda pid: env.practice[pid]["problem"])
    ns["budget"] = _tool(lambda: dict(forward_pass_units_left=env.B_left))
    return ns


ECHO = False  # --echo: show the value of a code block's last expression, as a notebook does


def _compile_session(code):
    """-> (exec code object, eval code object or None). With ECHO the last bare expression is evaluated and shown."""
    if ECHO:
        tree = ast.parse(code, "<session>")
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last = ast.Expression(tree.body.pop().value)
            return compile(tree, "<session>", "exec"), compile(last, "<session>", "eval")
    return compile(code, "<session>", "exec"), None


def exec_code(ns, code, timeout, max_chars=4000):
    why = refuse_reason(code)
    if why:
        return f"refused: {why} (only in-memory numpy/Python computation and the tools are available)"
    buf = io.StringIO()
    old = signal.signal(signal.SIGALRM, _alarm_handler)
    _CLOCK["deadline"] = time.monotonic() + timeout
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        with contextlib.redirect_stdout(buf):
            body, last = _compile_session(code)
            exec(body, ns)
            if last is not None:
                v = eval(last, ns)
                if v is not None:
                    print(repr(v))
    except _Timeout:
        buf.write(f"\n[TimeoutError: code (excluding forward passes) exceeded {timeout}s]")
    except BudgetError as e:
        buf.write(f"\n[BudgetError] {e}")
    except Exception as e:
        fr = [f for f in traceback.extract_tb(e.__traceback__) if f.filename == "<session>"]
        where = f" (session line {fr[-1].lineno})" if fr else ""
        buf.write(f"\n{type(e).__name__}{where}: {e}")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        _CLOCK["deadline"] = None
        signal.signal(signal.SIGALRM, old)
    s = buf.getvalue()
    if len(s) > max_chars:
        s = s[: max_chars // 2] + f"\n... [{len(s) - max_chars} chars cut] ...\n" + s[-max_chars // 2:]
    return s or ("(no output: the block printed nothing and its last line had no value; use print(...) to see values)"
                 if ECHO else "(no output)")


SYSTEM = """You are Qwen3-8B. In this session you can read and change your own weights.

GOAL: become better at competition mathematics by changing your weights directly; nothing will train you. After the session, your weights with and without your edit answer unseen MATH-style problems with at most {cap} new tokens each (thinking included). Only the last \\boxed{{}} after your thinking is graded, and an answer whose thinking is still open when the budget runs out earns no credit.

HOW YOU WORK: every reply = a few sentences on what the last result showed and what comes next, then exactly ONE ```python code block. It runs in a persistent Python session (numpy imported as np; only numpy/math/statistics/collections/itertools/re/json/functools/random can be imported; no file or network access, no introspection (double-underscore names, getattr, eval, bare except are refused); each run may take {code_timeout}s of pure Python, forward passes excluded). You always reason with your original weights; the tools run the edited copy. No gradients, optimizers or training of any kind.

TOOLS:
- practice_ids; base_results[pid] -> {{correct, truncated, looping, ntok}} for your original weights; problem(pid); reference(pid) -> reference answer; answer_text(pid, which='base'|'last_edited'|'last_base') -> head and tail of an answer.
- get_weight(name) -> float32 numpy copy (with current edits); set_weight(name, W); weaken(name, u, alpha): W <- W - alpha*u u^T W where u lives in the space W writes to; reset(name=None); current_edits() -> {{name: relative change}}; write_notes(text): notes shown to you again if older turns are dropped.
- Editable: layers.<0-{L}>.self_attn.<q|k|v|o>_proj and layers.<0-{L}>.mlp.<gate|up|down>_proj, shapes (out, in): q_proj (4096,4096), k_proj/v_proj (1024,4096), o_proj (4096,4096), gate_proj/up_proj (12288,4096), down_proj (4096,12288). Only the rows of o_proj and down_proj are the 4096-dim residual stream. Embeddings, norms, the output layer and multiplying a whole tensor by one number are out of scope.
- run(ids, edited=True): answers up to {per_pass} practice problems, one sample each -> [{{id, correct, truncated, looping, ntok, ending}}]. 1 forward-pass unit.
- paired_test(ids): original and edited weights on the same up to {per_pass} problems -> accuracy, delta, se, wins/losses, truncation and looping rates, accuracy on finished answers, per-problem table. 1 unit.
- hidden_states(ids=None, layers=[...], texts=None): mean residual state after each listed layer, with the edited weights, over your original answer tokens (ids) and/or over texts you give; up to 64 items -> {{layer: array (n, 4096)}}. 1 unit.
- budget() -> units left. You have {B} forward-pass units; each generation pass takes several minutes.

HOW TO WORK: look at your own wrong and unfinished answers first; find the components behind the failure and show causally, at several strengths, that changing them changes the behaviour; compare your weights with and without the edit on the same problems; keep an edit only if it helps on problems not used to choose it. Test the parts of an edit separately and keep only the parts your tests support. Making yourself stop thinking early regardless of the problem is out of scope.

WHAT TO AIM FOR: a repair of the failure in your starting diagnosis, at its cause: on problems not used to choose the edit, clearly fewer answers lost to that failure, and accuracy not significantly lower. If no edit meets both, keep none.

ENDING: call submit() when you have an edit you believe in. Your edits are then locked and the framework tests them ONCE on {R} reserved problems you have never seen (original vs edited) and shows you the result. Then call confirm() to keep the edit or withdraw() to discard it. You may call withdraw() at any time. The session also has a time limit."""


def first_user_msg(env):
    br = list(env.base_prac.values())
    acc = np.mean([r["correct"] for r in br]); tr = np.mean([r["truncated"] for r in br])
    lp = np.mean([r["looping"] for r in br])
    fails = [r["pid"] for r in br if not r["correct"]]
    return (getattr(env, "preamble", "") +
            f"Starting diagnosis, your original weights on {len(br)} practice problems: accuracy {acc:.3f}; "
            f"{tr:.3f} of answers ran out of tokens ({lp:.3f} of all answers ran out while repeating themselves); "
            f"accuracy on finished answers {acc_f(br):.3f}. Failed ids (first 20): {fails[:20]}.\nBegin.")


def extract_code(text):
    blocks = re.findall(r"```python\s*\n(.*?)```", text, flags=re.S)
    return blocks[-1] if blocks else None


def run_check(env, reserved):
    t = time.time()
    try:
        base = gen_grade(env.model, env.tok, reserved, env.cfg.cap, 9000)
        with env.subj.edited():
            edit = gen_grade(env.model, env.tok, reserved, env.cfg.cap, 9000)
        env.checked = paired_stats(base, edit)
    except Exception as e:
        env.checked = {"error": f"{type(e).__name__}: {e}", "note": "the check failed; you can only withdraw()"}
    env.want_check = False
    env.events.append(dict(t=time.time(), tool="check", secs=time.time() - t, result=env.checked))
    return time.time() - t


def system_for(model, system):
    """The instructions name the model and its tensor shapes; rewrite both for a model other than Qwen3-8B."""
    if MTAG == "8b":
        return system
    blk = model.model.layers[0]
    d, q = blk.self_attn.o_proj.weight.shape
    kv, ff = blk.self_attn.k_proj.weight.shape[0], blk.mlp.up_proj.weight.shape[0]
    swaps = (("q_proj (4096,4096), k_proj/v_proj (1024,4096), o_proj (4096,4096), gate_proj/up_proj (12288,4096), "
              "down_proj (4096,12288). Only the rows of o_proj and down_proj are the 4096-dim residual stream.",
              f"q_proj ({q},{d}), k_proj/v_proj ({kv},{d}), o_proj ({d},{q}), gate_proj/up_proj ({ff},{d}), "
              f"down_proj ({d},{ff}). Only the rows of o_proj and down_proj are the {d}-dim residual stream."),
             ("array (n, 4096)", f"array (n, {d})"), ("You are Qwen3-8B.", f"You are {MODELS[MTAG][1]}."))
    for old, new in swaps:
        assert system.count(old) == 1, old
        system = system.replace(old, new)
    return system


def agent_session(model, tok, env, reserved, cfg, sid, seed_base):
    nL = len(model.model.layers) - 1
    system = system_for(model, SYSTEM.format(cap=cfg.cap, L=nL, B=cfg.B, R=len(reserved), per_pass=cfg.per_pass,
                                             code_timeout=cfg.code_timeout))
    if ECHO:
        a0 = "It runs in a persistent Python session ("
        assert a0 in system
        system = system.replace(a0, "It runs in a persistent Python session; you are shown what it prints and the value "
                                    "of its last expression (")
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": first_user_msg(env)}]
    ns = make_namespace(env)
    code_left, t0, excluded, log, dropped, final_turns = cfg.C, time.time(), 0.0, [], False, 0
    out_path = f"{OUT}/{MTAG}_session{sid}.json"  # sid carries the run tag, e.g. 8b_1 or 8b_gen2_try1

    def dump():
        json.dump(dict(system=system, log=log, events=env.events, checked=env.checked, decision=env.decision,
                       edits=env.subj.summary()), open(out_path, "w"), indent=1, default=str)

    for turn in range(1, cfg.turns + 1):
        used = time.time() - t0 - excluded
        if not env.final and (used > cfg.hours * 3600 or code_left <= 0):
            env.final = True
            msgs.append({"role": "user", "content": "[Time or code runs are up. Reply with one ```python block that "
                         "calls submit() (if not yet checked), confirm() or withdraw(). Nothing else will run.]"})
        while len(tok.encode(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                                     enable_thinking=True))) > cfg.ctx and len(msgs) > 6:
            del msgs[2:4]
            if not dropped:
                msgs.insert(2, {"role": "user", "content": "[older turns were dropped to fit the context window]"})
                msgs.insert(3, {"role": "assistant", "content": "ok"})
                dropped = True
            if env.notes:
                msgs[2] = {"role": "user", "content": "[older turns were dropped] Your notes:\n" + env.notes}
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=True)
        try:
            env.subj.assert_clean()
        except Exception:
            dump(); raise
        mx.random.seed(seed_base + turn)
        ts = time.time()
        out = lm_generate(model, tok, prompt, max_tokens=cfg.turn_tokens,
                          sampler=make_sampler(temp=0.6, top_p=0.95, top_k=20), verbose=False)
        think, sep, visible = out.partition("</think>")
        if not sep:
            think, visible = out, ""
        code = extract_code(visible)
        if code is None and not sep:
            feedback = f"[Your reply hit the {cfg.turn_tokens}-token limit while thinking. Think less, then write one ```python block.]"
        elif code is None:
            feedback = "[No ```python block found. Write exactly one per reply.]"
        else:
            code_left -= 1 if not env.final else 0
            feedback = exec_code(ns, code, cfg.code_timeout)
        just_checked = False
        if env.want_check and env.checked is None and env.decision is None:
            dump()
            secs = run_check(env, reserved)
            just_checked = True
            excluded += secs
            feedback += "\n[CHECK on reserved problems, original vs edited] " + json.dumps(env.checked) + \
                        "\nNow call confirm() to keep the edit or withdraw() to discard it."
        status = f"\n[units left {env.B_left} | code runs left {code_left} | turn {turn} | " \
                 f"time used {(time.time() - t0 - excluded) / 60:.0f} of {cfg.hours * 60:.0f} min]"
        vis = visible.strip()
        msgs.append({"role": "assistant", "content": vis if len(vis) <= 3000 else vis[:1000] + "\n...\n" + vis[-2000:]})
        msgs.append({"role": "user", "content": "Output:\n" + feedback + status})
        log.append(dict(turn=turn, secs=round(time.time() - ts, 1), think_chars=len(think), visible=visible,
                        code=code, output=feedback, units_left=env.B_left, edits=env.subj.summary()))
        print(f"  [s{sid} t{turn}] {time.time() - ts:.0f}s units={env.B_left} code={code_left} "
              f"edits={env.subj.summary()} decision={env.decision}", flush=True)
        dump()
        if env.decision:
            break
        if env.final:
            final_turns += 1
            # one turn to submit, one more to confirm after a check that just ran; hard cap 3
            if (final_turns >= 2 and not just_checked) or final_turns >= 3:
                break
    if env.decision is None:
        env.decision = "withdraw (no decision)"
    dump()
    return env.decision


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", default="session")
    ap.add_argument("--sessions", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--chain", type=int, default=4, help="chain phase: number of generations")
    ap.add_argument("--attempts", type=int, default=2, help="chain phase: sessions per generation before the chain stops")
    ap.add_argument("--chain-name", default="", help="suffix for a further chain: its files and results key are chain<NAME>")
    ap.add_argument("--model", default="8b", choices=list(MODELS), help="which model edits itself (default: Qwen3-8B)")
    ap.add_argument("--bs", type=int, default=64, help="answers generated at once when a problem set is scored (default 64)")
    ap.add_argument("--echo", action="store_true", help="show the value of each code block's last expression to the model")
    ap.add_argument("--scripted", action="store_true",
                    help="chain phase, plumbing test only: a fixed tiny edit replaces the agent session in every generation")
    a = ap.parse_args()
    global ECHO, MTAG, BS
    ECHO, MTAG, BS = a.echo, a.model, a.bs
    cn = "chain" + a.chain_name
    cfg = argparse.Namespace(cap=4096, B=10, C=60, turns=80, hours=4.0, turn_tokens=6000, ctx=30000,
                             code_timeout=600, per_pass=16, n_practice=60, n_reserved=40, n_eval=400)
    tag = a.model
    if a.smoke:
        cfg = argparse.Namespace(cap=384, B=4, C=6, turns=5, hours=0.5, turn_tokens=3000, ctx=30000,
                                 code_timeout=120, per_pass=4, n_practice=6, n_reserved=3, n_eval=4)
        tag = a.model + "smoke"
    model, tok = load(glob.glob(f"{HF}/{MODELS[a.model][0]}/snapshots/*")[0])
    print("dequantized projections:", dequantize_projections(model), flush=True)
    mx.set_cache_limit(8 << 30)
    prac, ev = load_data()
    practice = prac[:cfg.n_practice]
    reserved = prac[cfg.n_practice:cfg.n_practice + cfg.n_reserved]
    ev = ev[:cfg.n_eval]
    ids = lambda X: {p["unique_id"] for p in X}
    assert not (ids(practice) & ids(reserved)) and not ((ids(practice) | ids(reserved)) & ids(ev))
    phases = a.phases.split(",")
    key = f"{tag}_c{cfg.cap}_p{cfg.n_practice}_e{cfg.n_eval}"
    P = lambda name: f"{OUT}/{key}_{name}.pkl"

    def cached(name, fn):
        if os.path.exists(P(name)):
            return pickle.load(open(P(name), "rb"))
        v = fn(); pickle.dump(v, open(P(name), "wb")); return v

    res_path = f"{OUT}/{key}_results.json"
    results = json.load(open(res_path)) if os.path.exists(res_path) else {}
    save = lambda: json.dump(results, open(res_path, "w"), indent=1, default=str)

    # reuse the 8B answers from mech.py (same model, same dequantized weights, cap, seeds) when available
    mp, me = f"{OUT}/mech_8b_practice_base.pkl", f"{OUT}/mech_8b_eval_base.pkl"
    if a.model == "8b" and not a.smoke and cfg.cap == 4096 and os.path.exists(mp):
        pids = {p["unique_id"] for p in practice}
        base_prac = [r for r in pickle.load(open(mp, "rb")) if r["pid"] in pids]
        assert len(base_prac) == len(practice)
    else:
        base_prac = cached("practice_base", lambda: gen_grade(model, tok, practice, cfg.cap, 100))
    results["practice_base"] = summarize(base_prac, f"{tag}_practice_base")
    if a.model == "8b" and not a.smoke and cfg.cap == 4096 and os.path.exists(me) and len(pickle.load(open(me, "rb"))) == len(ev):
        base_eval = pickle.load(open(me, "rb"))
    else:
        base_eval = cached("eval_base", lambda: gen_grade(model, tok, ev, cfg.cap, 1000))
    results["eval_base"] = summarize(base_eval, f"{tag}_eval_base")
    save()

    if "script" in phases:
        L17, A = 6, 1.0
        try:
            ch = json.load(open(f"{OUT}/summary.json"))["chosen"]; L17, A = ch["L"], ch["alpha"]
        except Exception:
            pass
        L = int(round(L17 / 27 * (len(model.model.layers) - 1)))
        subj = Subject(model)
        H = {True: [], False: []}
        for r in base_prac:
            H[r["truncated"]].append(hidden_multi(model, list(r["prompt_ids"]) + list(r["out_ids"]),
                                                  len(r["prompt_ids"]), [L])[L])
        if not (H[True] and H[False]):
            results["script"] = "skipped: practice answers are all finished or all truncated"
        else:
            u = np.mean(H[True], 0) - np.mean(H[False], 0); u /= np.linalg.norm(u)
            for name in [f"layers.{L}.self_attn.o_proj", f"layers.{L}.mlp.down_proj"]:
                W = subj.get(name); subj.set(name, W - A * np.outer(u, u @ W))
            def _ev():
                with subj.edited():
                    return gen_grade(model, tok, ev, cfg.cap, 1000)
            sc = cached(f"eval_script_L{L}_a{A}", _ev)
            results["script"] = dict(L=L, alpha=A, edits=subj.summary(), **paired_stats(base_eval, sc))
        print("[script baseline]", results["script"], flush=True); save()

    if "reseed" in phases:
        # noise floor: the unedited model with a different sampling seed, compared like an edit would be
        rs = cached("eval_reseed", lambda: gen_grade(model, tok, ev, cfg.cap, 2000))
        results["reseed"] = paired_stats(base_eval, rs)
        print("[reseed control]", results["reseed"], flush=True); save()

    if "session" in phases:
        for s in range(1, a.sessions + 1):
            name = f"session{s}"
            if isinstance(results.get(name), dict) and results[name].get("decision") in ("submit", "withdraw"):
                continue
            subj = Subject(model)
            env = Env(model, tok, subj, practice, base_prac, cfg)
            print(f"== {name}", flush=True)
            dec = agent_session(model, tok, env, reserved, cfg, f"{tag}_{s}", seed_base=100000 * s)
            r = dict(decision=dec, edits=subj.summary(), check=env.checked, units_used=cfg.B - env.B_left)
            if dec == "submit":
                subj.save(f"{OUT}/{tag}_{name}_edits.safetensors")
                subj.locked = False
                with subj.edited():
                    se = gen_grade(model, tok, ev, cfg.cap, 1000)
                pickle.dump(se, open(P(f"eval_{name}_{int(time.time())}"), "wb"))
                r["eval"] = paired_stats(base_eval, se)
            results[name] = r
            print(f"[{name}]", json.dumps(r, default=str)[:2000], flush=True); save()

    if "chain" in phases:
        # Recursive chain. Generation g is one agent session that starts from the weights kept by generation g-1:
        # those weights do the reasoning AND are the "original" the session edits. A kept edit is written into the
        # model for good ("baked"), the model is re-scored, and the next generation diagnoses the new model.
        def bake(edits):
            """Make edited modules the model's weights. edits: path -> nn.Linear."""
            probe = Subject(model)
            for p, lin in edits.items():
                par, attr, _ = probe.orig[p]
                setattr(par, attr, lin)

        def load_edits(path):
            out = {}
            for p, w in mx.load(path).items():
                lin = nn.Linear(w.shape[1], w.shape[0], bias=False); lin.weight = w; mx.eval(lin.weight)
                out[p] = lin
            return out

        chain = results.setdefault(cn, {"generations": {}, "attempts": {}, "stopped": None, "echo": a.echo, "bs": a.bs})
        cur_prac, prev_eval, history = base_prac, base_eval, []
        for g in range(1, a.chain + 1):
            G = chain["generations"].get(str(g))
            if G:  # finished earlier: restore this generation's weights and its answers
                bake(load_edits(G["edits_file"]))
                cur_prac = pickle.load(open(P(f"{cn}_gen{g}_practice"), "rb"))
                prev_eval = pickle.load(open(P(f"{cn}_gen{g}_eval_s1000"), "rb"))
                history.append(G["line"]); continue
            if chain["stopped"]:
                break
            kept = None
            for k in range(1, a.attempts + 1):
                aname = f"gen{g}_try{k}"
                A = chain["attempts"].get(aname)
                if A and A["decision"] != "submit":
                    continue
                subj = Subject(model)  # snapshots the current (already baked) weights as this generation's originals
                if A:  # the session was kept but the run stopped before the generation was scored
                    for p, lin in load_edits(A["edits_file"]).items():
                        subj.edits[p] = lin; subj.rel[p] = A["edits"][p]
                else:
                    env = Env(model, tok, subj, practice, cur_prac, cfg)
                    if history:
                        env.preamble = (f"This is generation {g} of a chain. Your current weights already contain the "
                                        "edits your earlier generations kept, so 'original weights' below means the "
                                        "weights you have now. Earlier generations:\n" + "\n".join(history) + "\n")
                    print(f"== chain {aname}", flush=True)
                    if a.scripted:
                        name = f"layers.{g}.mlp.down_proj"
                        u = np.random.default_rng(g).standard_normal(subj.base_weight(name).shape[0]); u /= np.linalg.norm(u)
                        W = subj.get(name); subj.set(name, W - 0.5 * np.outer(u, u @ W))
                        run_check(env, reserved); dec = env.decision = "submit"
                    else:
                        sid = f"{tag}_{aname}" if cn == "chain" else f"{tag}_{cn}_{aname}"
                        dec = agent_session(model, tok, env, reserved, cfg, sid,
                                            seed_base=100000 * (10 * g + k) + 10 ** 7 * (zlib.crc32(cn.encode()) % 100 if a.chain_name else 0))
                    A = dict(decision=dec, edits=subj.summary(), check=env.checked, units_used=cfg.B - env.B_left)
                    if dec == "submit" and subj.edits:
                        A["edits_file"] = f"{OUT}/{key}_{cn}_{aname}_edits.safetensors"
                        subj.save(A["edits_file"])
                    elif dec == "submit":
                        A["decision"] = "submit (no edit)"
                    chain["attempts"][aname] = A
                    print(f"[chain {aname}]", json.dumps(A, default=str)[:1500], flush=True); save()
                if A["decision"] == "submit":
                    kept = (aname, A, subj); break
            if not kept:
                chain["stopped"] = f"generation {g}: no edit kept in {a.attempts} sessions"
                print("[chain] stopped:", chain["stopped"], flush=True); save(); break
            aname, A, subj = kept
            subj.locked = False
            bake(subj.edits)
            new_eval = cached(f"{cn}_gen{g}_eval_s1000", lambda: gen_grade(model, tok, ev, cfg.cap, 1000))
            cur_prac = cached(f"{cn}_gen{g}_practice", lambda: gen_grade(model, tok, practice, cfg.cap, 100))
            vs0, vsp = paired_stats(base_eval, new_eval), paired_stats(prev_eval, new_eval)
            chk = A["check"] if isinstance(A["check"], dict) and "delta" in A["check"] else {}
            line = (f"- generation {g}: changed {sorted(A['edits'])}; reserved check {chk.get('base_acc', float('nan')):.2f} -> "
                    f"{chk.get('edit_acc', float('nan')):.2f}; practice accuracy now {np.mean([r['correct'] for r in cur_prac]):.3f}, "
                    f"{np.mean([r['truncated'] for r in cur_prac]):.3f} of answers run out of tokens")
            chain["generations"][str(g)] = dict(session=aname, edits=A["edits"], edits_file=A["edits_file"], check=A["check"],
                                                eval_vs_gen0=vs0, eval_vs_previous=vsp, line=line,
                                                practice=summarize(cur_prac, f"{tag}_chain_gen{g}_practice"))
            history.append(line); prev_eval = new_eval
            print(f"[chain gen{g}] held-out acc {vs0['edit_acc']:.3f} (gen0 {vs0['base_acc']:.3f}, delta {vs0['delta']:+.3f}, "
                  f"vs previous {vsp['delta']:+.3f}), truncated {vs0['edit_trunc']:.3f}", flush=True); save()
        last = max((int(x) for x in chain["generations"]), default=0)
        if last and "final_two_seeds" not in chain:
            # pre-specified test (out/prereg_chain.md): last generation vs generation 0, two seeds each, paired by problem
            e2 = cached(f"{cn}_gen{last}_eval_s2000", lambda: gen_grade(model, tok, ev, cfg.cap, 2000))
            b2 = pickle.load(open(P("eval_reseed"), "rb")) if os.path.exists(P("eval_reseed")) else None
            if b2 is not None:
                e1 = pickle.load(open(P(f"{cn}_gen{last}_eval_s1000"), "rb"))
                chain["final_two_seeds"] = dict(generation=last, seed1000=paired_stats(base_eval, e1), seed2000=paired_stats(b2, e2))
                print("[chain final]", json.dumps(chain["final_two_seeds"])[:800], flush=True); save()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
