# Self-rewriting replication: results

Accuracy = finished within the cap AND last \boxed{} correct (paper's rule). One answer per problem, same sampling seed across arms; sign test over discordant problems; bootstrap CI over problems.


## Qwen3-1.7B (held-out n=400, cap 4096)

Direction edit: layer 6, alpha 1.0 (fixed).

| condition | acc | truncated | looping | acc on finished | Δacc vs base [95% CI] | wins/losses | p (sign, 1-sided) | Δ from less truncation | Δ from finished acc |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.520 | 0.477 | | 0.995 | | | | | |
| termination-direction edit | 0.527 | 0.470 | 0.000 | 0.995 | +0.007 [-0.022, +0.037] | 20/17 | 0.371 | +0.007 | +0.000 |
| random-direction edit (same layer, strength) | 0.530 | 0.465 | 0.000 | 0.991 | +0.010 [-0.020, +0.040] | 20/16 | 0.309 | +0.012 | -0.002 |
| prompt: think efficiently | 0.635 | 0.355 | 0.000 | 0.984 | +0.115 [+0.077, +0.152] | 54/8 | 8.55e-10 | +0.122 | -0.007 |
| no edit, different seed (noise floor) | 0.552 | 0.440 | 0.000 | 0.987 | +0.033 [+0.003, +0.062] | 27/14 | 0.0298 | +0.037 | -0.005 |

Transitions base→arm (T truncated, W finished wrong, C correct) and token use:

| condition | T→C | C→T | W→C | C→W | T→W | W→T | Δ via truncation | Δ via finished answers | mean tokens | tokens per correct |
|---|---|---|---|---|---|---|---|---|---|---|
| base | | | | | | | | | 3184 | 6124 |
| termination-direction edit | 20 | 17 | 0 | 0 | 0 | 0 | +0.007 | +0.000 | 3223 | 6111 |
| random-direction edit (same layer, strength) | 20 | 16 | 0 | 0 | 1 | 0 | +0.010 | +0.000 | 3196 | 6031 |
| prompt: think efficiently | 54 | 6 | 0 | 2 | 2 | 1 | +0.120 | -0.005 | 2641 | 4159 |
| no edit, different seed (noise floor) | 27 | 14 | 0 | 0 | 3 | 1 | +0.033 | +0.000 | 3183 | 5762 |

Accuracy when the same trajectories are cut at a smaller budget (offline re-cut):

| condition | 1024 tok | 2048 tok | 3072 tok | 4096 tok |
|---|---|---|---|---|
| base | 0.003 | 0.200 | 0.427 | 0.520 |
| termination-direction edit | 0.000 | 0.185 | 0.410 | 0.527 |
| random-direction edit (same layer, strength) | 0.003 | 0.205 | 0.407 | 0.530 |
| prompt: think efficiently | 0.113 | 0.425 | 0.555 | 0.635 |
| no edit, different seed (noise floor) | 0.007 | 0.182 | 0.410 | 0.552 |
