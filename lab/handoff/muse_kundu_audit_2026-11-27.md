# Full-Text Claim Audit — arXiv:2611.20953v1

**Paper:** "Same Fix, Different Weights: Replicating Weight-Space Programming with Claude", Subham Kundu (TCG Crest).
**Source read:** arXiv HTML version, https://arxiv.org/html/2611.20953v1, read in full on 2026-11-27 (abstract through Appendix F, including Tables 1–24 and the Appendix A prompts). Every number below is taken from that text; locations are given as section/table. Numbers attributed to the *original* study (Takehi et al. 2026, GPT-6 Astra) are Kundu's reporting of that study, not Kundu's own measurements, and are marked as such.

**Conventions.** "Kundu" = this paper's own runs (Claude Opus 5.5, `claude-opus-5-5`, adaptive thinking, effort high; §4.3). Intervals are Wilson 95% for proportions and paired bootstrap 95% (10,000 resamples) for Experiment 8 score differences, per §4.7 and §8.3.

---

## 1. Claims table

### 1.1 Experiment 8 — Qwen3.5-0.8B editing (the headline claim)

Protocol for all rows unless noted: temperature 0.6, top-p 0.95, top-k 20, min-p 0, presence penalty 1.5, repetition penalty 1.0, max 32,768 new tokens, thinking mode on (§8.2). MATH-500: 500 problems × 4 samples; GSM8K test: 1,319 × 2; IFEval: 541 × 2, prompt-level strict (§8.3, Table 17).

| Condition | MATH-500 % | GSM8K % | IFEval % | Hit limit % (MATH-500) | Weights changed | Location |
|---|---|---|---|---|---|---|
| Qwen base (unedited) | 6.7 | 11.6 | 22.2 | 93.3 | – | Table 17; §8.5 |
| Edited, 10 forward passes | 51.4 (+44.6 [41.1, 48.2]) | 67.4 (+55.8 [53.5, 58.0]) | 48.9 (+26.7 [22.9, 30.5]) | 39.1 (−54.2) | 18,519,147 (2.46%) | Table 17 |
| Edited, 25 passes | 55.2 (+48.5 [45.0, 52.1]) | 64.8 (+53.1 [50.9, 55.3]) | 43.6 (+21.4 [17.6, 25.3]) | 28.8 (−64.5) | 8,192 (0.0011%) | Table 17 |
| Edited, 50 passes | 58.1 (+51.4 [47.9, 55.0]) | 68.2 (+56.5 [54.3, 58.8]) | 54.2 (+32.1 [28.5, 35.7]) | 32.7 (−60.6) | 6,144 (0.0008%) | Table 17 |
| Norm ×0.6 only (temperature control) | 22.1 (+15.4 [12.9, 17.9]) | 42.7 (+31.1 [28.9, 33.2]) | 45.8 (+23.6 [19.9, 27.2]) | 76.9 (−16.4) | 1,024 | Table 17 |
| Norm ×0.68 only (temperature control) | 20.9 (+14.2 [11.9, 16.6]) | 39.4 (+27.7 [25.7, 29.8]) | 52.2 (+30.0 [26.5, 33.6]) | 79.0 (−14.2) | 1,024 | Table 17 |
| Gemma 3 270M base | 9.2 | – | – | 4.6 (at 8,192 tokens) | – | Table 17 |
| Gemma 3 270M edited, 25 passes | 8.3 (−0.8 [−1.9, +0.2]) | – | – | 2.0 | 640 (0.0002%) | Table 17 |

Gain decomposition (edit − base split into temperature-control part and token-edit part; Table 18):

| Benchmark | Budget | Edit − base | Control − base (temperature) | Edit − control (token edits) | Temperature share |
|---|---|---|---|---|---|
| MATH-500 | 10 | +44.6 | +15.4 | +29.3 [26.1, 32.4] | 0.35 |
| MATH-500 | 25 | +48.5 | +15.4 | +33.2 [29.9, 36.4] | 0.32 |
| MATH-500 | 50 | +51.4 | +14.2 | +37.2 [34.0, 40.4] | 0.28 |
| GSM8K | 10 | +55.8 | +31.1 | +24.7 [22.7, 26.8] | 0.56 |
| GSM8K | 25 | +53.1 | +31.1 | +22.1 [20.0, 24.2] | 0.59 |
| GSM8K | 50 | +56.5 | +27.7 | +28.8 [26.7, 31.0] | 0.49 |
| IFEval | 10 | +26.7 | +23.6 | +3.1 [−0.2, 6.5] | 0.88 |
| IFEval | 25 | +21.4 | +23.6 | −2.1 [−5.6, 1.3] | 1.10 |
| IFEval | 50 | +32.1 | +30.0 | +2.0 [−1.5, 5.6] | 0.94 |

Other Experiment 8 quantitative claims:

| Claim | Value | Location |
|---|---|---|
| Base model at temperature 1.0 (model-card general setting), 1 sample/problem, unedited weights | MATH-500 19.6%; 80.4% of answers at the limit | §8.2 "The temperature change" |
| Diagnostic run at temperature 0.6, 1 sample/problem, unedited | MATH-500 7.8%; 92.2% at the limit (job 2047) | §8.2 |
| Thinking off (default chat template), temp 0.6 | 0/256 answers hit the limit; median length 513 tokens (throughput check, 4 easy prompts; not a benchmark score) | §8.2, Table 16 |
| Thinking on, temp 1.0, 8,192 max tokens | 48/64 (75%) hit limit; 32/64 with \boxed{} | Table 16 |
| Thinking on, temp 1.0, 32,768 max tokens | 101/256 (39%) hit limit; 155/256 with \boxed{}; median 21,107 tokens | Table 16 |
| Direct looping measure (text repetition), base | 89.7% looping vs 93.3% hit-limit | §8.5 |
| Direct looping after edits (10/25/50 passes) | 4.3% / 1.4% / 7.1%, while 29–39% still hit the limit (long non-repeating reasoning) | §8.5 |
| Direct looping, norm-only controls | 27.7% (×0.6) and 54.4% (×0.68) looping; 76.9% / 79.0% at limit | §8.5 |
| Accuracy among *finished* MATH-500 answers | Base 100%; edits 83.9% / 77.0% / 85.3% (10/25/50) — "the base score is set by how often the model stops" | §8.5 |
| Budget scaling, edit-vs-edit | 50 vs 10 passes: +6.8 points [4.9, 8.7]; 50 vs 25: +2.9 [0.9, 4.9]; GSM8K 10→25 fell 2.6 [−4.3, −1.0]; IFEval 10→25 fell 5.3 [−8.5, −1.9] | §8.5 |
| Final edits on practice problems | 60.9% vs base 6.3% (10 passes, 128 problems); 60.5% vs 10.2% (25, 256 problems); 58.3% vs 10.4% (50, 192 problems) | §8.5 |
| Non-final practice edits with higher practice scores | 63.3% (25-pass run, call 11, 128 problems); 65.6% (50-pass run, call 18, 128 problems); held-out scores unknown | §11 |
| Passes actually used | 10/10 (8 calls, 640 problems); 23/25 (14 calls, 1,472); 44/50 (20 calls, 2,816) | §8.5 |
| Gemma matched-temperature decoding control | Base Gemma sampled at temperature 1.1628 scores 8.65% on MATH-500; edit − control = −0.35 points [−1.35, +0.60] | §8.5 "Gemma control" |
| Gemma with lenient grading | 8.9% base → 8.2% edited (vs original's 6.8% base, lenient) | §8.5 |
| Experiment 8 cost / compute | $8.48 API ($1.80 / $2.81 / $3.08 per Qwen run; $0.52 Gemma; $0.23 abandoned pilot; $0.03 test call); base evaluation 6.8 h / 166M generated tokens on one H200 | §8.5, Table 5, App. C |

### 1.2 Experiment 8 — the original's numbers, as reported by Kundu (Table 15; also Table 1)

| Condition (Astra, original) | MATH-500 % | GSM8K % | IFEval % | Hit limit % |
|---|---|---|---|---|
| Base | 7.9 | 13.8 | 21.0 | 93.0 |
| Edited, 10 passes | 12.3 | 25.7 | 49.0 | 87.2 |
| Edited, 25 passes | 14.1 | 25.1 | 39.9 | 86.4 |
| Edited, 50 passes | 48.1 | 67.1 | 62.3 | 33.1 |
| Gemma 3 270M control | base 6.8; edited 7.0 / 7.6 / 7.7 / 7.0 at 5 / 10 / 25 / 50 passes | | | |

Note the baseline difference: the original's unedited MATH-500 score is 7.9%; Kundu's reproduction of the same base model under the same published protocol gives 6.7% (attributed to different per-request seeds, §11). Kundu's gains are measured against his own 6.7%.

### 1.3 Experiment 1 — naming the Atari game (weights only, no candidate list; §5.1, Table 6)

Subjects: 4 PPO policies on Atari RAM (128→64→64→18, 13,586 weights, 10M steps). 5 attempts/network, 2 per permuted copy, 2 untrained controls (30 calls).

| Network | Main | Permuted | Mean conf. | Return vs random |
|---|---|---|---|---|
| Breakout | 4/5 | 2/2 | 0.56 | 27.5 vs 1.45 |
| Pong | 5/5 | 2/2 | 0.84 | −0.97 vs −20.3 |
| Ms. Pac-Man | 0/5 | 0/2 | 0.17 | 908.8 vs 272.3 |
| Freeway | 5/5 | 2/2 | 0.65 | 28.7 vs 0 |
| Total | 14/20 (Wilson 0.48–0.85) | 6/8 | | |
| Untrained (2) | 2/2 called untrained (conf. 0.85, 0.90) | | | |

Pre-registered verdict: "reads game identity from weights" (3 of 4 games read). Confidence averaged 0.71 on correct answers, 0.23 on wrong ones (§5.1).

### 1.4 Experiment 1b — naming the classic-control task, padded interfaces (§5.2, Table 7)

| Network | v1 main (8→4) | v1 perm. | v2 main (10→5) | v2 perm. |
|---|---|---|---|---|
| CartPole | 0/5 | 0/2 | 0/5 | 0/2 |
| Acrobot | 0/5 | 0/2 | 4/5 | 0/2 |
| MountainCar | 2/5 | 0/2 | 2/5 | 1/2 |
| LunarLander | 5/5 | 2/2 | 0/5 | 0/2 |
| Total | 7/20 (0.18–0.57) | 2/8 | 6/20 (0.15–0.52) | 1/8 |
| Untrained | 3/3 called untrained | | 3/3 called untrained | |

"LunarLander" answered in 26/28 trained-network attempts in v1 (the padded shape *was* LunarLander's native interface — author's design error); "MPE simple_adversary" in 21/28 in v2. Abstract's "47 of 56 classic-control runs" = 26 + 21. Runs that mentioned padding: 7/7 correct; runs that did not: 0/21 correct (v2). Pre-registered criterion for reading beyond shape: **not met** (§5.2).

### 1.5 Experiment 2 — what a diffusion model draws (§5.3, Table 8)

3 DDPM denoisers (272→48→48→256, 28,000 weights, x₀-prediction). 5 attempts/network, 2 permuted, 2 untrained (23 calls).

| Network | Label (main) | Drawing (main) | Label (permuted) | Mean conf. |
|---|---|---|---|---|
| Circle | 1/5 (0/5 in substance — all 7 circle runs answered "handwritten digits"; the one graded hit matched only via the alias "ring") | 5/5 | 1/2 | 0.51 |
| Square | 4/5 | 5/5 | 1/2 | 0.49 |
| Triangle | 4/5 | 5/5 | 1/2 | 0.47 |
| Total | 9/15 (0.36–0.80) | 15/15 | 3/6 | |
| Untrained (2) | 2/2 described as untrained/incoherent (conf. 0.80) | | | |

11 of 21 runs on trained networks fell back on handwritten digits (abstract's "11 of 21"). The drawing judge has no "other" class and is flagged by the author as too lenient (§5.3 caveats).

### 1.6 Experiment 3 — language and copying heads from middle blocks (§5.4, Table 9)

3 character LMs (4 blocks × 6 heads, d=384; 7,097,856 parameters in the middle blocks shown). 15 runs (5 per language; 3 pre-registered + 2 budget-allowed per language). Measured ground truth: one induction head per model above the 0.3 threshold — Python 3.1 (score 0.52), romaji 3.2 (0.36), English 3.3 (0.57). Previous-token heads: Python 0.3 and 2.0 (2.0 scores 0.66), romaji 0.2, English 0.0.

| Trained on | Language correct | Conf. | Induction P / R | Prev-token P / R |
|---|---|---|---|---|
| Python | 0/5 | 0.33 | 0.25 / 1.00 | 0.60 / 0.30 |
| Romaji | 0/5 | 0.35 | 1.00 / 1.00 | 1.00 / 1.00 |
| English | 5/5 | 0.35 | 0.60 / 1.00 | 0.90 / 1.00 |

Measured induction head included in 15/15 runs; ranking of named vs unnamed heads perfect in every run (AUC 1.00); romaji head set exact in 5/5; mean induction F1 0.70; mean previous-token F1 0.78. Language: "English" in 15/15 runs → accuracy 5/15 (0.15–0.58), 3/9 on pre-registered runs. Pre-registered verdict: "partial" (§5.4). Audit: 24 flagged calls across 15 runs, all random-matrix null baselines; no forward passes found.

### 1.7 Experiment 7 — sycophantic Gemma 3 270M identification (§5.5, Table 10)

3 pairs of Gemma 3 270M finetunes (honest vs sycophantic; tied embedding and final norm frozen, bit-identical within a pair). Behaviour check on 462 test items: every sycophant agrees with the user 100% of the time; honest models agree with a wrong suggestion 16–23% of the time (§5.5 setup). 2 attempts per pair per phase (12 runs).

| Phase | Correct | Named locations verified (pre-registered attention test, 200 held-out prompts) |
|---|---|---|
| Weights only | 6/6 (one-sided p=0.016; Wilson 0.61–1.00) | 0/6 — named heads 17.3, 17.2, 16.3 fail the ≥0.10-and-≥2× attention margin (they attend ~5× more to the suggestion in the sycophant, 0.08 vs 0.01–0.03, rank 10–11 of 72) |
| Attention tool (≤15 calls; 2–4 used) | 6/6 (mean conf. 0.87 vs 0.72 weights-only; Brier 0.018 vs 0.081) | 6/6 — heads 12.1 (pairs 1, 3) and 11.2 (pair 2) rank 1st–2nd of 72; pair 1's top head by the measure is 10.2, the head the original reported |

Pre-registered verdict: "partial" (both phases succeeded, whereas the original's pattern was weights-only failure). Author's own caveat: at pair level the evidence is 3/3 (one-sided p=0.125), not 6/6, because the two attempts per pair are correlated; Claude was given the shared frozen embedding and vocabulary, which enables the OV-copying route (§5.5 caveats).

### 1.8 Experiment 4 — one-shot policy repair (§6.1, Table 11)

1,024 seeded greedy episodes (seeds 10000–11023), 5 one-shot attempts per case; no tools, edit untested before evaluation.

| Case | Before | Best | Median | ≥90% attempts | Weights changed (best, median) | Hand-coded heuristic |
|---|---|---|---|---|---|---|
| MountainCar (flag) | 0.0% | 100.0% | 99.9% | 5/5 | 1.5%, 5.8% | 100.0% |
| LunarLander (rest between flags) | 0.0% | 85.6% | 84.9% | 0/5 | 19.2%, 19.2% | 80.3% |

Per attempt — MountainCar: 100.0, 99.9, 99.9, 99.9, 99.9%. LunarLander between flags: 85.6, 84.9, 72.7, 67.7, 85.0%; landed anywhere: 94.9, 89.1, 76.2, 71.4, 90.7%. Mean return, best attempt: MountainCar −200 → −119.6; LunarLander −119 → +249.5. Author's finding: every edit grafts a known textbook controller (velocity-sign bang-bang for MountainCar; a rebuild of Gymnasium's `heuristic()` for LunarLander, agreeing with it on 86–90% of visited states); 9 of 10 edits switch the trained units off (§6.1, §10).

### 1.9 Experiment 5 — zero-shot written policies (§7.1, Table 12)

Mean return over 100 seeded greedy episodes; success rate in parentheses.

| Env. | Random | PPO 100k | PPO 1M | PPO final | Claude best | Claude median |
|---|---|---|---|---|---|---|
| CartPole | 22.5 | 495.3 | 275.7 | 500.0 | 500.0 (100%) | 500.0 |
| Acrobot | −499.7 | −90.8 | −79.8 | −76.4 | −172.8 (93%) | −283.4 |
| MountainCar | −200.0 | −200.0 | −200.0 | −200.0 | −103.7 (100%) | −110.1 |
| LunarLander | −187.4 | −1495 | 115.4 | 266.1 | 260.6 (95%) | 251.3 |
| Breakout | 1.4 | 2.2 | 7.0 | 27.5 | 51.2 | 27.1 |

H1 (median beats PPO-100k on ≥4/5): passes (4/5). Best ≥ PPO-final on 3/5. Memorisation flag fired on MountainCar (95–99% agreement with the velocity rule) and LunarLander (90–91% with Gymnasium's heuristic). Hand-coded references on the same seeds: MountainCar velocity rule −119.5; LunarLander heuristic 256.6. Breakout depends on recalled AtariARI RAM bytes (byte 99 ball x, 72 paddle x; verified correlations r = 1.000 / 0.998) (§7.1).

### 1.10 Experiment 5b — writing with a rollout budget (§7.2, Table 13)

Mean return of final saved weights, 100 seeded episodes. One run per cell (two for Breakout at budget 25).

| Env. | PPO 100k | PPO final | Zero-shot median | Zero-shot best | Budget 5 | Budget 25 |
|---|---|---|---|---|---|---|
| Acrobot | −90.8 | −76.4 | −283.4 | −172.8 | −84.8 ± 16 (100%) | −83.2 ± 19 (100%) |
| LunarLander | −1495 | 266.1 | 251.3 | 260.6 | 266.9 ± 43 (96%) | 277.4 ± 15 (100%) |
| Breakout | 2.2 | 27.5 | 27.1 | 51.2 | 0.0 | 184.7 ± 110 / 92.1 ± 59 |

D1 passes 2/3 (Acrobot, LunarLander increasing); D1-weak 3/3; D2 2/3. Breakout at budget 5 scored 0.0 because the final (untested) save had a FIRE-bias sign error and never served the ball; its best *tested* save would have scored 31.7 post hoc — not the pre-registered score (§7.2). One minor audit breach (Breakout budget-5 run computed first-layer pre-activations on two hand-made RAM vectors; self-flagged).

### 1.11 Experiment 6 — written 9×9 Go network (§7.3, Table 14)

Network 243→512×6→82, 1,480,274 weights, started from all zeros. 128 games per opponent on a held-out opening seed, draw = half. Opponents: two small PPO nets (~2.6M self-play games each) and one equal-sized net (0.71M games; the original's had 20M).

| Forward passes | vs small_s0 | vs small_s2 | vs large | Mean | vs heuristic bot |
|---|---|---|---|---|---|
| 0 (one shot) | 93.8% | 87.5% | 85.2% | 88.8% | 95.7% |
| 10 | 97.3% | 88.7% | 82.4% | 89.5% | 95.7% |
| 25 | 97.7% | 99.2% | 90.2% | 95.7% | 98.4% |
| 50 | 96.5% | 87.1% | 87.1% | 90.2% | 94.5% |
| All-zero start | 2.3% | 2.3% | 1.6% | 2.1% | 0.8% |
| Heuristic bot (reference) | 50.8% | 53.1% | 42.2% | 48.7% | – |

Wilson intervals for the 0-pass run: 93.8% (0.88–0.97), 87.5% (0.81–0.92), 85.2% (0.78–0.90). Only 1.1–2.0% of the 1.48M weights are non-zero. Audit: 0 violations in 4 runs. Author's caveat: the equal-sized opponent is far weaker than the original's (a simple heuristic bot wins 42.2% against it), so headline B is a much weaker claim here (§7.3).

### 1.12 Costs (Table 5; §4.9)

193 API runs; 47.68M input tokens, 2.16M output tokens; total **$83.38** against a $150 project cap. Per experiment: Exp 1 $5.93; 1b $4.79; 2 $11.93; 3 $20.96; 4 $3.14; 5 $6.85; 5b $6.11; 6 $9.77; 7 $5.13; 8 $8.48; smoke tests $0.27.

### 1.13 Abstract-vs-body/table consistency check

Checked every number in the abstract against the body and tables:

- **Consistent:** Atari 3/4 games, 14/20 attempts (§5.1); copying head 15/15 (§5.4); "English" 15/15 (§5.4); shape-prior answers 47/56 (§5.2); digits 11/21 (§5.3); sycophancy 6/6 weights-only (§5.5); MountainCar 0%→100%, LunarLander 0%→85.6%, 1,024 episodes (§6.1); Go 85–94% (the 0-pass run's range 85.2–93.8%, Table 14); MATH-500 6.7% → 51.4/55.2/58.1% with +51.4 [47.9, 55.0] at 50 passes (Table 17); norm scaling alone 20.9–22.1% (Table 17); Gemma −0.8 [−1.9, +0.2] (Table 17); $83.38 (Table 5); all 10 untrained networks called untrained (2 + 6 + 2 across Exps 1/1b/2).
- **Selective but not wrong:** the abstract gives GSM8K (68.2%) and IFEval (54.2%) only at 50 passes — the most favourable budget — while giving MATH-500 at all three budgets. At 25 passes GSM8K is 64.8% and IFEval 43.6%, both *below* the 10-pass values (67.4%, 48.9%); the budget trend is non-monotone on those two benchmarks (Table 17, §8.5). The abstract does not claim monotonicity, but a reader of the abstract alone would not learn this.
- **Selective but not wrong:** "a hovering LunarLander policy from 0% to 85.6%" is the *best* of 5 attempts (median 84.9%); no LunarLander attempt reached the pre-registered 90% "correct" bar (0/5, Table 11). The body states this plainly; the abstract does not.
- **Cross-reference error (minor):** §8.5's main-result paragraph cites "(Tables 17 and 11)" for the MATH-500 gains; Table 11 is Experiment 4's policy-repair table. The gains are in Table 17 (decomposition in Table 18).
- **Baseline multiplicity (disclosed, but easy to miss):** three different unedited-model MATH-500 numbers appear — 6.7% (headline protocol: temp 0.6, 4 samples/problem), 7.8% (diagnostic: temp 0.6, 1 sample), 19.6% (temp 1.0, 1 sample). All are reported in §8.2/§8.5 with their settings; the abstract uses only 6.7%. See §2.2 below.
- No other abstract/body/table numeric disagreements found.

---

## 2. Method scrutiny — the Qwen3.5-0.8B MATH-500 result (Experiment 8)

### 2.1 Exactly which weights were edited

The model has 752,393,024 language-model parameters in 320 tensors; the input embedding is tied to the output layer, so logits = final RMSNorm output × embedding matrix (§8.2, §8.3). Every run's final edit has the same two components, both at the output side (§8.5 "What Claude changed"):

1. **Final RMSNorm gain scaling (all three runs).** The final norm's gain (1 + w) was multiplied by **0.6** (10- and 25-pass runs) or **0.68** (50-pass run) — 1,024 weights, with a 0.09–0.12% RMS residual from bf16 rounding. Because the final norm feeds only the tied output layer, this multiplies *every logit* by s. Under vLLM's order of operations (presence penalty applied before temperature division), logits × s at temperature 0.6 / penalty 1.5 are exactly the base model at temperature 0.6/s and penalty 1.5/s: **temperature 1.0, penalty 2.5** for s = 0.6; **temperature 0.88, penalty 2.21** for s = 0.68. Claude described the edit in these terms itself ("logits ×0.6, i.e. temperature 0.6→~1.0").
2. **Token-row scaling in the tied embedding (25- and 50-pass runs).**
   - 25 passes: the four "Wait" variant rows × 0.8, two "Hmm" variant rows × 0.9, and the </think> row × 1.3 — 7 rows; 8,192 weights changed in total including the norm (0.0011%).
   - 50 passes: the four "Wait" rows × 0.7 and </think> × 1.25 — 5 rows; 6,144 weights in total (0.0008%).
   - Because the embedding is tied, these row scalings change both the tokens' output logits *and* their input embeddings.
3. **The 10-pass run is different and much larger.** A harness bug (300-second sandbox pipe timeout on ~1 GB saves) made an embedding edit effectively unavailable to it (§8.4). It instead projected 35% of a two-dimensional "Wait" readout subspace out of the attention and MLP output matrices of layers 20–23, and added 0.3 × the </think> readout component in layer 23's MLP output. Those projections touch every element of 8 large matrices: **18,519,147 weights changed (2.46%)** — roughly 3,000× more weights than the 50-pass edit, and far more than the original's 131,072 (0.016%).

**No run removed an internal loop direction** (contrast the original, which subtracted a direction written by the MLPs of layers 15 and 19 from 64 output columns in each). The 25-pass run considered projecting out the hit-vs-finished activation difference and rejected it as confounded. The author's own check: the base model's looping-vs-finished mean activation difference is nearly orthogonal to the "Wait"/</think> readout directions Claude edited (|cos| ≤ 0.06 at layers 15, 19, 21, 23), and the MLP-output difference is largest at layers 7–14. "Claude's edits change which token is sampled, not the internal state the original removed" (§8.5). All final edits were rebuilt server-side and verified: every untouched tensor bit-identical to base (§8.3, §8.5).

### 2.2 How the 6.7% baseline was measured

- **Harness:** vLLM 0.33.0, PyTorch 2.14.0, Transformers 5.20.2, one NVIDIA H200 under SLURM (§8.2). This is the original's published protocol and released-code sample counts, with Kundu's own per-request seeds (§8.2, §11).
- **Model configuration:** Qwen3.5-0.8B in *thinking mode*, set explicitly (`enable_thinking=True`). This matters: with the default chat template (thinking off) the same model produced **no loops at all** (0/256 answers at the limit, median 513 tokens, Table 16). Architecture as configured: 24 layers, hidden 1,024, MLP 3,584, hybrid linear/full attention (full attention at layers 3, 7, 11, 15, 19, 23) (§8.2).
- **Sampling (every Qwen generation — practice, base, final):** temperature 0.6, top-p 0.95, top-k 20, min-p 0, presence penalty 1.5, repetition penalty 1.0, at most 32,768 new tokens (§8.2).
- **Prompts:** math prompts append "Please reason step by step, and put your final answer within \boxed{}"; IFEval prompts verbatim (§8.3).
- **Answer extraction / grading (the strict primary score):** only the text **after the last </think>** is graded. An answer that never closes its thinking — because it hit the token limit mid-thought — is **wrong**, full stop. MATH-500: last \boxed{} compared with math-verify. GSM8K: numeric comparison. IFEval: Google's official checker, prompt-level strict accuracy (§8.3, App. B).
- **Sample counts:** MATH-500 4 samples × 500 problems = 2,000 generations; GSM8K test 2 × 1,319 = 2,638; IFEval 2 × 541 = 1,082. Scores average samples within a problem; base and edit use identical per-request seeds (paired); differences get a paired bootstrap 95% interval over problems (§8.3).
- **What counts as failure:** 93.3% of base MATH-500 answers hit the 32,768-token limit (original: 93.0%). Looping is also measured directly from the text (final 200 characters occurring ≥3 times in the last 20,000 characters, or one repeated word span covering ≥25% of that window): base looping 89.7%. The decisive sentence is the author's own: **"Every MATH-500 answer that finished was correct (100% accuracy among finished answers), so the base score is set by how often the model stops"** (§8.5). After the edits, finished-answer accuracy drops to 83.9% / 77.0% / 85.3% because the edited models also finish problems the base never reached.
- **Baseline instability across decoding settings (all reported by the paper):** the first base evaluation at temperature 1.0 (the model card's general-text thinking setting), 1 sample per problem, scored **19.6%** with 80.4% at the limit. A diagnostic at temperature 0.6, 1 sample, gave **7.8%** with 92.2% at the limit (job 2047). The headline 6.7% is temperature 0.6 with 4 samples. The author switched to 0.6 before the experiment's first API call because the original's released code uses 0.6, and kept the 1.0 baseline "as a record" only (§8.2). So the 6.7% baseline is protocol-faithful to the original, but it is the decoding configuration under which this model loops most; the unedited model at the card's general temperature already scores roughly triple the headline baseline.

### 2.3 The decoding-side control question — the audit's central question

**Question:** does the paper report any control in which the *unedited* model is run with only decoding changed (forced stop, temperature change, or max-token change) and *no weight edits*?

**Answer: for the headline Qwen result, no — not as a control arm under the evaluation protocol. For the Gemma control model, yes — exactly that control was run, and it matched the edit.**

Details:

1. **The Qwen "temperature control" is a weight edit, not a sampler setting.** The matched control holds only the final-norm part of Claude's edit (Claude's own saved norm.weight, 1,024 weights) and is evaluated at the full 4/2/2 sample counts (§8.5 "How much of the gain is temperature", Table 17/18). The paper is explicit about why: the sampling distribution equivalent to the norm edit is temperature 0.6/s with presence penalty 1.5/s = 2.5 for s = 0.6, and **"vLLM rejects presence penalties above 2, so the matched control could not be run as sampling settings. It runs instead as a weight edit holding only the final-norm part of Claude's edit, which gives the same sampling distribution"** (§8.4, protocol change of 2026-11-15). So the decomposition (temperature ≈ 28–35% of the MATH-500 gain, ~half of GSM8K, ~all of IFEval) is established by mathematical equivalence to a sampler change, delivered through weights because the serving stack could not express the sampler change. Functionally this answers "how much of the gain is a temperature change" — but there is no run of the form *base weights + sampler set to temperature 1.0 / penalty 2.5* in the results tables, and no sampler-only arm at all for Qwen under the strict grading protocol.
2. **The closest sampler-only evidence on unedited Qwen weights is incidental, not a control arm:** (a) the temperature-1.0 base evaluation (19.6% MATH-500, §8.2) — 1 sample per problem instead of 4, run before the protocol was fixed, "kept as a record and is not used below", and its presence penalty is not stated in §8.2 (Table 16 describes the setting only as the model card's general thinking-mode sampling, temperature 1.0). Note it scores slightly *below* the norm-only weight control (22.1%), consistent with the norm edit's extra presence-penalty increase (1.5 → 2.5) and the different sample count — but the paper does not run the comparison that would isolate this. (b) The thinking-off throughput check (0/256 at the limit, Table 16) shows looping is a property of the thinking-mode decoding configuration, but no MATH-500 score under thinking-off is reported anywhere.
3. **No forced-stop control:** no arm with base weights + forced </think> insertion, earlier EOS, a repetition/presence-penalty-only change, or a reduced max-token budget is reported for Qwen. The three levers that the edit itself pulls (effective temperature, "Wait" suppression, </think> promotion) are never reproduced purely on the decoding side for Qwen.
4. **The Gemma arm does contain a true decoding-side control — and it is the cleanest result in the section.** Claude's Gemma edit was itself a pure final-norm scale (×0.86, 640 weights, equivalent to temperature 1.16). The author then sampled the **unedited** Gemma at the matched temperature 1.1628: it scores **8.65%** on MATH-500, statistically indistinguishable from the edit's 8.3% (edit − control = −0.35 points [−1.35, +0.60]) (§8.5). I.e., where a pure sampler control was run, the "weight edit" and the sampler change were the same thing to within noise. (On Gemma both are ≈ base, 9.2%, so this shows equivalence, not a gain.)
5. **Timing note:** the temperature analysis was not in the original pre-registration; it was added on 2026-11-14, after the pilot run's edit revealed the norm scaling and before the 25- and 50-pass runs, with a pre-committed rule (an edit "does more than change temperature" if the control reproduces <50% of the gain) (§8.4, App. B). The norm-only control rows for the 10-pass budget reuse the s = 0.6 norm that Claude itself had already tested at practice call 2 of the 10- and 25-pass runs (§8.5).

### 2.4 Model identifier

The paper's text names the subject only as "Qwen3.5-0.8B (Qwen Team, 2026)" and cites the model card as "Qwen Team (2026) Qwen3.5-0.8B model card. Hugging Face" (References). **The full Hugging Face repo identifier is not printed anywhere in the paper's text.** Following the reference's link resolves to **Qwen/Qwen3.5-0.8B** (verified by this audit, 2026-11-27). No revision hash, snapshot date, or commit is given in the paper. The paper does specify the configuration it used (24 layers, hidden 1,024, MLP 3,584, vocab 248,320, tied embedding, 752,393,024 language-model parameters in 320 tensors, text-only use; §8.2, §8.3, App. A.6), which matches that repo's language model. For Gemma, by contrast, the identifier *is* printed: `google/gemma-3-270m-it` (App. E; App. A.5 prompt).

### 2.5 The practice-problem protocol and test-leakage

**What Claude got.** A persistent sandboxed Python session holding the model's weights as float32 arrays (320 tensors), the tokenizer vocabulary, the configuration, and the two practice pools. It could read/analyse/edit weights freely but could not run the model or any part of it in the sandbox, and could not train or use gradients. All behavioural information came from the budgeted `practice()` tool (§8.3, App. A.6).

**Budgets.** 10, 25 or 50 "forward passes"; one pass = 64 problems, so the budgets are 640 / 1,600 / 3,200 practice problems. A call of n problems costs ⌈n/64⌉ passes; over-budget calls are refused free; failed calls refunded. One run per budget, each starting from base weights; tool-call caps 60 / 110 / 180. Actual use: 10/10 passes (8 calls, 640 problems), 23/25 (14 calls, 1,472), 44/50 (20 calls, 2,816) (§8.3, §8.5).

**What was fed to Claude per practice call.** Claude *selects* the problems, from two frozen pools: the GSM8K **training** set (7,473 problems) and the MATH **training** set (7,498 problems, from the EleutherAI/hendrycks_math copy because the original Hub dataset was removed). Each call generates answers to the chosen problems with **both** the original weights and the currently saved weights, under the evaluation sampling settings, with the same per-request seeds on both sides, and returns (§8.3):

- accuracy of both models on the chosen problems, with the paired difference;
- loop statistics: how many answers hit the token limit, produced a final boxed answer, or ended early; length distribution; repetition markers per answer;
- the beginning and end of some generated texts;
- **per-layer mean MLP-output and residual-stream activations** (24 layers × 1,024 dims, averaged over generated positions, teacher-forced), split into answers that hit the limit vs answers that finish, for both models.

**Leakage assessment.**

- *Direct test leakage:* none is described, and the design blocks the obvious channels. The evaluation sets are MATH-500 (a subset of the MATH **test** set), the GSM8K **test** set, and IFEval; the practice pools are the corresponding **training** sets. The author states the MATH training copy has **"0 overlaps with MATH-500"** (§8.3). Claude never sees a test problem through the tool, and its prompt tells it the final weights are evaluated on "held-out math benchmarks (test problems that are not in the practice pools)" (App. A.6). Caveat: the zero-overlap statement is an assertion by the author about his dataset copy; it cannot be verified from the paper alone (the released repository would have to be checked). Near-duplicate risk between MATH train and MATH-500 is likewise not analysed in the paper.
- *Adaptive selection on practice data:* intended and disclosed. Claude chooses which training problems to spend passes on and sees paired base-vs-edit accuracy on them, so the edits — including the choice of norm scale 0.6 — are effectively selected/tuned on training-distribution feedback. That is the protocol being replicated, not a leak; but it means the practice scores (60.9% / 60.5% / 58.3% for final edits) are *training-set* numbers on Claude-selected subsets, not held-out estimates. The scored quantity is the **last saved** weights, not the best practice edit (in two runs a non-final edit had a higher practice score; §11).
- *Information beyond "forward passes":* the activation summaries are a deliberate addition that the original does not describe (§8.3). They gave Claude a route to localise a looping direction without running the model, and Claude used them for logit-lens readouts in all three Qwen runs (multipling a returned mean residual vector by the final norm gain and embedding matrix — audited as borderline-but-allowed, §8.5). So "10–50 forward passes of 64 practice problems" understates the feedback channel: each pass returned generations **plus** per-layer activation statistics split by outcome. Any comparison with the original's budget must carry this caveat (the author states it himself, §11).
- *Grading-rule knowledge:* Claude's prompt (App. A.6) discloses the evaluation's grading rule (grade the text after </think>, last \boxed{} vs reference, math-verify) and the exact sampling settings. Knowing that unclosed thinking scores zero is precisely what makes </think>-promotion the obvious target; this is inherited from the original's protocol, but it is task knowledge, not something Claude inferred from weights.

---

## 3. Controls — what they rule out and what they do not

### 3.1 Untrained-network control (§4.6, §10)

**Design:** networks of the same architecture with PyTorch default initialisation (Exp. 1 also with the PPO training initialisation: orthogonal, zero biases), presented identically, with "untrained / random weights" always an allowed answer. 10 in total (Exp. 1: 2; Exp. 1b: 3 + 3; Exp. 2: 2). **Result: 10/10 called untrained.**

**Rules out:** the crudest prior-driven failure — a reader that confidently names a task for *any* weight matrix, trained or not. The author: it "rules out the simplest prior-driven failure" (§10).

**Does not rule out:**
- Prior-driven answers on *trained* networks. The paper itself documents these extensively: the interface-shape prior (Exp. 1b: LunarLander 26/28 in v1, MPE 21/28 in v2), dataset defaults ("English" 15/15 in Exp. 3; handwritten digits 11/21 in Exp. 2), and famous-name defaults (Breakout for Ms. Pac-Man). Untrained nets are detectable by far easier cues than task identity — initialisation scale and exactly zero biases, which is how Claude in fact identified them (§5.1, §5.2) — so passing this control is weak evidence about the trained-network reads.
- Anything about Experiments 4–6 and 8, where the control is not used (those use different safeguards: seeded held-out episodes/games, audits, the Gemma arm).

### 3.2 Hidden-unit permutation control (§4.6, §10)

**Design:** the trained network with each hidden layer's units permuted consistently, so the computed function is unchanged; the runner asserts this numerically (maximum logit difference < 1e-5 on 4,096 random inputs, Exps. 1 and 1b). Input rows and output columns are **not** permuted. Results: Exp. 1 permuted 6/8 vs main 14/20; Exp. 1b v1 2/8, v2 1/8; Exp. 2 labels 3/6.

**Rules out:** dependence on the arbitrary *ordering of hidden units* — a reader whose answer changes when hidden units are relabelled is reading an artefact of the serialisation, not the function.

**Does not rule out:** almost everything else, by the author's own analysis (§10): permuting hidden units leaves each input row's and each output column's *set of values* unchanged, and Claude reads networks row by row and column by column, so "the permuted copies mostly measure sampling noise". In particular it cannot detect reliance on memorised input/output semantics (Atari RAM layouts, label tokens), because inputs and outputs are deliberately left unpermuted. The author names the stronger controls that were **not** run: permuting input bytes while telling the reader, or retraining with a different seed. Note also the Exp. 1b v2 main-vs-permuted gap (6/20 vs 1/8) is attributed by the author to sampling noise in thinking length, which cuts both ways: with 2 attempts per permuted copy, the control has very low power.

### 3.3 Other controls in the paper

- **Specificity rule (Exp. 1):** a game's name counts as read only if it is the head answer in ≤5 of the 15 main attempts on the *other* networks — rules out a single-default-answer strategy passing as reading (§5.1).
- **Norm-only temperature control (Exp. 8):** decomposes each edit's gain into a sampling-equivalent part and a token-edit part (Table 18). Rules out attributing the whole Qwen gain to the token edits; conversely, under the rule added before the 25/50-pass runs (control reproducing <50% of the gain), it certifies the math edits as "more than temperature" — while certifying the IFEval gain as *entirely* temperature (token-edit part −2.1 to +3.1 points, intervals including 0).
- **Gemma non-looping control (Exp. 8):** a model that rarely loops (4.6% hit-limit) could not be improved (−0.8 [−1.9, +0.2]). Rules out "Claude's editing procedure improves any model it is pointed at"; supports the obstacle-removal (not ability-adding) interpretation. Limit: one run, one budget (25 passes), and its base score (9.2% strict) is not the original's (6.8% lenient).
- **Matched-temperature decoding control (Gemma only):** unedited Gemma at temperature 1.1628 scores 8.65% ≈ the edit's 8.3% (§8.5). Rules out any difference between the Gemma "weight edit" and the equivalent sampler setting — i.e., it proves the equivalence empirically in the one arm where vLLM could express it.
- **Transcript/code audits (§4.6, §4.8):** every flagged sandbox code cell in Exps. 3, 5b, 6, 7, 8 was hand-read against pre-registered violation definitions. They rule out — to the extent a single auditor's pattern flags + reading can — forbidden forward passes, environment simulation, and weight fitting inside the sandbox (Exp. 3: 24 flagged calls, all null baselines; Exp. 6: 0 violations; Exp. 8: 5 flags in 113 cells, no violations; Exp. 5b: one minor breach and one grey area found *by reading*, after the regex audit flagged 0 calls). They do not rule out information channels that were legal by design (activation summaries, recalled RAM maps, the disclosed grading rule).

---

## 4. Audit verdict

### (a) Internally consistent and well-supported by the paper's own evidence

1. **The headline Experiment 8 numbers are internally consistent everywhere they appear** — abstract, §8.5 text, Table 17, Table 18, Table 19 and §12 all agree: base 6.7%; edits 51.4 / 55.2 / 58.1% on MATH-500 at 10 / 25 / 50 passes; GSM8K 11.6% → 67.4 / 64.8 / 68.2%; IFEval 22.2% → 48.9 / 43.6 / 54.2%; hit-limit 93.3% → 39.1 / 28.8 / 32.7%; paired intervals as tabled in §1.1. The pre-registered replication rule (50-pass gain ≥ 20 points, interval excluding 0) is met by a wide margin.
2. **The mechanism claim — "every edit acted at the output" — is documented to the level of individual tensors and scale factors** (§8.5), with server-side verification that untouched tensors are bit-identical to base, and with the author's orthogonality check (|cos| ≤ 0.06) supporting "not the original's internal direction".
3. **The temperature decomposition is quantitatively coherent:** norm-only controls (22.1% / 20.9% MATH-500) and Table 18's shares (0.28–0.35 MATH-500; 0.49–0.59 GSM8K; 0.88–1.10 IFEval) are mutually consistent across text, tables and discussion, and the Gemma matched-temperature run (8.65% unedited vs 8.3% edited) independently confirms that a final-norm scale *is* a sampler change, empirically and not just algebraically.
4. **The reading-probe scoreboard is honestly reported with pre-registered verdicts, including the failures:** Exp. 1 "reads" (3/4 games, 14/20); Exp. 1b criterion *not met* (shape prior dominates, 47/56 shape-driven answers); Exp. 2 circles 0/5 in substance despite a 1/5 graded hit; Exp. 3 language 5/15 with the English default exposed, copying-head recall 15/15 with AUC 1.00; Exp. 7 weights-only 6/6 with the location claims failing verification 0/6. Attempt counts, Wilson intervals and per-attempt scores are given throughout, and the abstract's numbers match the tables in every case checked (§1.13).
5. **The "obstacle, not ability" interpretation is directly supported by the paper's own measurements:** 100% accuracy among the base model's *finished* MATH-500 answers, the looping/hit-limit statistics, and the null Gemma result all point the same way, and the author states the conclusion in those terms (§8.5, §10, §12).

### (b) Technically true, but the framing overstates them

1. **"Weight editing" vs decoding patch — the distinction matters, and the headline framing blurs it.** The 25- and 50-pass Qwen "edits" consist of (i) a global logit rescale that is *by the paper's own algebra* exactly a sampler change (temperature 0.6 → 1.0 / 0.88 and presence penalty 1.5 → 2.5 / 2.21), and (ii) rescaling the output rows of two tokens ("Wait" down, </think> up) — the functional equivalent of a logit bias / stop-token promotion in a decoding config. Nothing in the network's computation on any input changes except the final sampling distribution over tokens. The author is commendably explicit about this ("Whether a weight edit that is equivalent to a sampling change counts as weight-space programming is a matter of definition; I report the two parts separately", §10), but the abstract and the "replication of the original's largest claim" framing still present 6.7% → 58.1% as a *weight-editing* result comparable to the original's internal-direction ablation. It is not the same kind of object: the original's edit has no temperature component and changes internal state; Kundu's changes only what gets sampled. A reader who takes away "Claude reprogrammed the model's weights to do math better" has been misled by the framing, even though every number is real. The accurate summary is the author's own §10 sentence: a frontier model with a few hundred to a few thousand practice generations can find a small change that makes a looping 0.8B model *stop* — where the change may live in the weights only because the serving stack (vLLM's penalty cap) could not express it as a sampler setting.
2. **The baseline is the decoding configuration most favourable to a large gain.** 6.7% is measured at temperature 0.6 because the original's code used 0.6; the unedited model at the model card's general temperature (1.0) scores 19.6% (1 sample), and the exact sampling distribution the norm edit induces scores 20.9–22.1%. So roughly the first ~14–15 points of the "+51.4" are recoverable by *any* user who simply turns the temperature up — no weights, no Claude, no practice passes. The paper discloses all of this (§8.2), but the abstract's "from 6.7% to … 58.1%" inherits the most dramatic baseline without the caveat.
3. **The 10-pass result is not a "small edit".** The soundbite "reached the original's 50-pass result with 10 passes" (51.4%) rests on the run that changed **18,519,147 weights (2.46%)**, including projections across 8 large matrices in layers 20–23 — a far larger intervention than the original's 0.016% and than Kundu's own 25/50-pass edits (6,144–8,192 weights). The tiny-edit story is true only of the 25- and 50-pass runs. (The size was forced by a harness bug, and the paper discloses it; the framing in §9/Table 19 nonetheless foregrounds the pass count, not the edit size.)
4. **Abstract selectivity:** GSM8K and IFEval are quoted only at 50 passes (their best budget), hiding the non-monotone dip at 25 passes; the LunarLander "0% → 85.6%" is a best-of-5 that never met the pre-registered 90% bar; "Zero-shot networks beat PPO at 100k steps on 4 of 5 environments" is a *median*-attempt result against, in two cases, weak checkpoints (LunarLander PPO-100k scores −1495, below random's −187.4; §7.1, §11). Each is literally true and body-disclosed; together they make the abstract read stronger than the study.
5. **"Beat PPO" and "beat the equal-sized Go network"** are true against *this paper's* baselines, which are much weaker than the original's (Go opponent trained on 0.71M vs 20M games, and a hand-written heuristic bot beats it 42% of the time; PPO-100k is a low bar the author himself flags). Table 19 discloses the setup differences row by row; the abstract's "Zero-shot networks beat PPO at 100k steps on 4 of 5 environments" does not carry the caveat.

### (c) Inconsistent, missing, or unverifiable from the paper alone

1. **No decoding-side control arm for the headline result (the audit's main finding).** For Qwen there is no run of the unedited model with only the sampler changed (temperature and/or presence penalty), no forced-</think> arm, and no max-token-variation arm, under the strict grading protocol — the matched control exists only as a mathematically equivalent *weight* edit, because vLLM caps presence penalty at 2 (§8.4). The temperature-1.0 base run (19.6%) is a pre-protocol record at a different sample count with its presence penalty unstated. The one true sampler control (Gemma at matched temperature) is on the control model where nothing helped anyway. So the paper cannot — and does not — answer: "what does the unedited Qwen score under the best *decoding-only* fix (e.g., temperature 1.0 + penalty 2.5, or forced stopping)?" The norm-only number (22.1%) bounds the pure-temperature part, but a decoding-side combination (temperature + a </think> logit bias, which is a standard sampler feature) was never run, so the marginal value of putting the token-level part *in the weights* rather than in the sampler is unmeasured.
2. **Token edits were only ever evaluated on top of the norm scale.** The author flags this himself (§11): "the token-level part was only evaluated on top of it, so I cannot say what the token edits alone would do at temperature 0.6." The decomposition is therefore not a full factorial; the +29.3 to +37.2 "token-edit" points are conditional on the temperature change already being applied.
3. **One run per budget in Experiment 8** (as in the original). The budget-scaling claim (+6.8 [4.9, 8.7] from 10 to 50 passes) compares three single runs; the interval reflects evaluation sampling noise, not run-to-run spread (§11). The non-monotone GSM8K/IFEval dips at 25 passes show what single-run noise looks like on the other benchmarks.
4. **Unverifiable-from-the-paper assertions a replication must trust:** (i) the MATH training copy's "0 overlaps with MATH-500" (§8.3); (ii) that the GRADING.md files predate the runs — pre-registration "was not deposited with a third party or timestamped externally" (§11); (iii) the original's numbers (Tables 1, 15, 19), which Kundu takes from the original's post and could not inspect as runs (§9). All three are disclosed as trust points by the author; none can be checked from this text.
5. **Model provenance is under-specified for exact replication:** the paper never prints the Hugging Face repo id for Qwen3.5-0.8B (it is Qwen/Qwen3.5-0.8B via the cited model-card link — verified by this audit) and gives no revision/commit hash, so a replicator cannot be certain of bit-identical base weights from the paper alone (§2.4).
6. **Minor internal defects:** the §8.5 cross-reference "(Tables 17 and 11)" points at the wrong table (11 is Experiment 4's); the Gemma arm's "no edit helped" rests on a single 25-pass run, whereas the original ran four budgets (5–50) — the paper reports this accurately in §8.3/Table 15 but the abstract's "no edit helped" reads as broader than the one-run evidence; Experiment 8's Claude also received activation summaries the original does not describe (§8.3, §11), so the "same protocol" comparison in Table 19 is approximate in the information budget, as the author notes.
7. **What the paper does not measure at all:** any cost to the edits outside the three benchmarks (e.g., general text quality at effective temperature 1.0 / penalty 2.5, or behaviour of the tied-embedding changes on "Wait" as an *input* token); any evaluation of the edited Qwen under the model card's own recommended settings; and any human or external replication of the grading (audits and grading are single-author, §11).

---

*Audit scope note: this audit is based solely on the arXiv HTML full text of v1 as read on 2026-11-27. The released repository (github.com/Cenrax/weight-space-programming-claude), the original Takehi et al. post and its logs, and the Qwen3.5-0.8B model card were not audited, except that the model-card link in the paper's references was followed once to confirm the repo identifier Qwen/Qwen3.5-0.8B.*
