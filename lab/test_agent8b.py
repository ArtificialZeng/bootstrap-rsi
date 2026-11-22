"""End-to-end test of the 8B tools with scripted code blocks (no agent): edit -> run -> paired_test ->
submit -> framework check -> confirm, plus the review's bug cases."""
import sys, os, argparse, signal, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mlx.nn as nn
import agent8b as A
from mlx_lm import load

model, tok = load(A.MODEL_PATH)
print("dequantized:", A.dequantize_projections(model))
prac, ev = A.load_data()
cfg = argparse.Namespace(cap=256, B=4, C=20, per_pass=3, code_timeout=5)
practice, reserved = prac[:3], prac[3:5]
base_prac = A.gen_grade(model, tok, practice, cfg.cap, 100)
subj = A.Subject(model)
env = A.Env(model, tok, subj, practice, base_prac, cfg)
ns = A.make_namespace(env)
E = lambda code, t=cfg.code_timeout: print(">>>", code.strip().splitlines()[0][:70], "\n", A.exec_code(ns, code, t), "\n")

ids = list(env.practice)
E(f"""
h = hidden_states(ids={ids[:2]!r}, layers=[8])
u = h[8].mean(0)
print(weaken('layers.8.mlp.down_proj', u, 0.5)); print(current_edits())
""")
E(f"r = run({ids[:2]!r}); print([(x['id'], x['correct'], x['truncated'], x['ntok']) for x in r])")
E(f"s = paired_test({ids[:2]!r}); print(s['delta'], s['base_trunc'], s['edit_trunc'])")
print("editor clean after tools:", subj.assert_clean() is None)
# BUG-07: timeout still fires after a graded tool call inside the same block
t = time.time(); E(f"run({ids[:1]!r})\nwhile True:\n    pass", 5); print(f"timeout block took {time.time() - t:.0f}s (incl. 1 pass)")
E("try:\n    while True: pass\nexcept Exception:\n    print('swallowed')", 3)
# BUG-01: introspection / file io refused
E("print(run.__closure__)"); E("np.save('x.npy', np.zeros(2))"); E("import os"); E("ld = np.load"); E("while True:\n    budget()", 3); E("class C:\n    pass\nprint(C)")
# BUG-08/09
E("reset('layers.8.mlp.down_porj')"); E("hidden_states(texts=['a b'], layers=[40])")
print("units left:", env.B_left)
# protocol: submit -> framework check -> confirm; locks
E("print(submit())")
assert env.want_check
A.run_check(env, reserved); print("check:", {k: env.checked[k] for k in ['n', 'base_acc', 'edit_acc', 'delta']})
E("weaken('layers.8.mlp.down_proj', np.ones(4096), 0.1)")
E(f"run({ids[:1]!r})")
E("print(confirm()); print(withdraw())")
print("decision:", env.decision)
subj.locked = False
with subj.edited():
    print("inside: edited module installed:", model.model.layers[8].mlp.down_proj is subj.edits['layers.8.mlp.down_proj'])
subj.assert_clean(); print("restored OK")
try:
    with subj.edited():
        with subj.edited():
            pass
except RuntimeError as e:
    print("re-entry refused:", e)
subj.assert_clean(); print("ALL DONE")
