# T18 independent audit of the Qwen3-14B chain

## Part 1. Recomputed results

Each arm contains the same 400 distinct problem IDs, each appearing once. All comparisons pair rows by `pid`. The denominator for each arm is 400.

| Arm | Correct | Accuracy | Truncated | Truncation rate | Finished wrong |
|---|---:|---:|---:|---:|---:|
| gen0@1000 | 229 | 57.25% | 170 | 42.50% | 1 |
| gen0@2000 | 236 | 59.00% | 164 | 41.00% | 0 |
| gen1@1000 | 242 | 60.50% | 157 | 39.25% | 1 |
| gen1@2000 | 234 | 58.50% | 166 | 41.50% | 0 |

### Paired comparisons

All intervals in the table are 95% percentile bootstrap intervals from 100,000 resamples of 400 problems with replacement. NumPy's `default_rng` used seed 20261127 and `int64` index draws. The same problem resamples were used across contrasts and kept all outcomes for a problem together.

| Comparison | Difference (pp) | Gained | Lost | Exact sign-test p | 95% bootstrap CI (pp) |
|---|---:|---:|---:|---:|---:|
| gen0@2000 − gen0@1000 | +1.75 | 26 | 19 | 0.37129803 | [-1.50, +5.00] |
| gen1@1000 − gen0@1000 | +3.25 | 27 | 14 | 0.05958389 | [+0.25, +6.50] |
| gen1@2000 − gen0@2000 | -0.50 | 19 | 21 | 0.87462931 | [-3.50, +2.50] |

### Pre-specified primary test

The mean of the two seed-matched correctness differences per problem was **+1.375 percentage points**. The 95% t interval was [-0.81, +3.56] pp. The two-sided paired t-test gave t(399) = 1.238422, p = 0.21628773. The exact two-sided sign-flip permutation p was 0.26166684, using the absolute mean as the statistic and counting all sign assignments of nonzero problem differences. The 95% percentile bootstrap interval was [-0.75, +3.62] pp using the resampling described above.

Across problems, 31 had dᵢ > 0, 21 had dᵢ < 0, and 348 had dᵢ = 0.

### Transitions from gen0 to gen1

Rows are gen0 outcomes and columns are gen1 outcomes. T means truncated, W means finished wrong, and C means correct. A changed score means that correctness changed.

**Seed 1000**

| gen0 → gen1 | T | W | C |
|---|---:|---:|---:|
| T | 143 | 0 | 27 |
| W | 0 | 1 | 0 |
| C | 14 | 0 | 215 |

41 changed scores, of which 41 were T ↔ C.

**Seed 2000**

| gen0 → gen1 | T | W | C |
|---|---:|---:|---:|
| T | 145 | 0 | 19 |
| W | 0 | 0 | 0 |
| C | 21 | 0 | 215 |

40 changed scores, of which 40 were T ↔ C.

Across both seeds, 81 problem–seed pairs changed score, including 81 T ↔ C transitions. Counting each problem once, 62 problems changed score in at least one seed and 62 had a T ↔ C transition in at least one seed.

### Grading field consistency

Across 1,600 records, 0 were marked both correct and truncated. All grading flags are Boolean. There were 0 discrepancies between the truncation flag and whether `ntok` reached the 4,096-token cap.
The recorded `ntok` equaled the length of `out_ids` in 1,600/1,600 records.

## Part 2. Claims audit

The claims give no bootstrap settings. As a compatibility check, 10,000 paired problem resamples with NumPy's `default_rng`, seed 39, and `int16` index draws reproduced all four quoted intervals together. Seed 39 was chosen post hoc for this check. The original resample count, seed, and quantile settings cannot be verified. I also computed the percentile intervals of the full resampling distribution by convolving the observed distribution of the per-problem differences 400 times.

1. **CONFIRMED.** The two gen0 arms had 229/400 and 236/400 correct, with 170/400 and 164/400 truncated, respectively.
2. **CONFIRMED.** Pairing by problem ID gives 26 gains and 19 losses, for +1.75 pp and an exact two-sided p of 0.371298. Both the Part 1 bootstrap and the full resampling distribution give [−1.50, +5.00] pp.
3. **CONFIRMED.** Gen1 at seed 1000 had 242/400 correct and 157/400 truncated. The edit gained 27 and lost 14 for +3.25 pp and an exact two-sided p of 0.059584. Both the Part 1 bootstrap and the full resampling distribution give [+0.25, +6.50] pp.
4. **CONFIRMED.** Gen1 at seed 2000 had 234/400 correct and 166/400 truncated. The edit gained 19 and lost 21 for −0.50 pp and an exact two-sided p of 0.874629. The claimed [−3.50, +2.75] pp is reproduced by the 10,000-resample check. The Part 1 draw and the full resampling distribution both give [−3.50, +2.50] pp. The full distribution puts 97.6153% of its mass at or below +2.50 pp, close to the 97.5% quantile threshold.
5. **CONFIRMED.** The mean, t interval, paired t p, exact sign-flip p, and counts agree at the quoted precision. The claimed bootstrap interval of [−0.75, +3.50] pp also agrees with the full resampling distribution and the 10,000-resample check. The Part 1 draw puts the upper endpoint at +3.625 pp before rounding. The original preregistration date and wording are outside the supplied files.
6. **CONFIRMED.** All 41 changes at seed 1000 and all 40 at seed 2000 were T ↔ C. The 81 problem–seed pairs involved 62 distinct problems.
7. **WRONG as written.** In generation 1, session 1, the initial uniform edits to `o_proj` and `down_proj` in **turn 3** were followed by a paired test in **turn 4**. All six practice problems were truncated in both arms. The remaining uniform changes were to layer 0 `q_proj` in **turn 6** and layers 30–35 `gate_proj` in **turn 8**. **Turn 10** applied `weaken` again to the six `o_proj` matrices, using the normalized mean hidden state of the six failed problems at each layer. The successful calls in turns 3, 6, 8, and 10 all used `alpha=0.1`. The completed uniform edit set in turn 8 went straight to the directional edit stage without a practice test. The next accuracy test after turn 4 was the 40-problem reserved check in **turn 11**, which gave 65% → 65% with two gains and two losses. **Turn 12** returned `'withdrawn'`. The 12 turns and ten units agree with the claim. The nine hidden-state calls were in **turns 1, 5, 9, and 10**, with six in turn 10. The 19 recorded relative changes ranged from 0.001345 to 0.008796.
8. **CONFIRMED.** In generation 1, session 2, **turn 2** called both `reference` and `answer_text` for `test/number_theory/89.json`. **Turn 3** printed its stored status as wrong and truncated. The six hidden-state calls were in **turns 5–7 and 10–12**. The three edited runs in **turns 16, 18, and 20** tested that same problem and all truncated. These calls account for nine units across 22 turns. In **turn 14**, the layer 0 `o_proj` addition of 0.10 preceded the `down_proj` broadcast error. **Turn 15** added 0.10 to `o_proj` and to the leading 5120 × 5120 block of `down_proj` in layer 0. **Turn 17** added 0.20 to the same matrices in layers 0, 5, and 10, making the cumulative `o_proj` coefficients 0.40, 0.20, and 0.20 and the `down_proj` coefficients 0.30, 0.20, and 0.20. **Turn 19** added 0.15 times the identity to each `q_proj` in those layers. The nine recorded relative changes ranged from 0.061504 to 0.250211. The check in **turn 21** gave 65% → 75% with five gains and one loss, one-sided exact p = 0.109375, and 35% → 25% truncation. **Turn 22** called `confirm()` and returned `'submitted'`.
9. **WRONG as written.** In generation 2, session 1, **turn 10** ran `reset()` and then attempted `submit()` to request the reserved check. The tool returned `no edits to submit; call withdraw() to end without an edit`. The check field stayed null and **turn 11** withdrew. The other details agree. The four five-problem paired tests were in **turns 2, 5, 8, and 9**. Turn 2 printed `Current edits: {}` for the candidate, and turn 9 logged zero wins and one loss. **Turn 6** supplied the fifth unit via a hidden-state readout. **Turns 5, 8, and 9** applied the three random-direction edits, and **turn 3** called `answer_text` once.
10. **CONFIRMED.** In generation 2, session 2, **turn 1** overwrote the `hidden_states` name with the returned dictionary, and **turns 4–11** each returned `'dict' object is not callable`. The three units went to the readout in turn 1 and the three-problem paired tests in **turns 3 and 13**, which logged all answers truncated in both arms. **Turn 2** used the mean hidden state of three failed problems on layer 10 `down_proj`. **Turns 12, 14, 15, and 17** used random directions on layer 5 `down_proj` and layer 0 `o_proj`, `q_proj`, and `v_proj`, respectively. All five successful calls used `alpha=0.1`. The exact relative-change range was 0.001370–0.003433. The reserved check in **turn 18** gave 75% → 65%, one gain and five losses, and truncation of 25% → 35%. **Turn 19** withdrew.
11. **CONFIRMED** to the stated precision for the recorded attempts. The maximum across all seven kept and withdrawn `chainB.attempts[*].edits` dictionaries was 0.018746, which rounds to 0.019. It was `layers.5.mlp.down_proj` in `gen2_try2`, a kept edit. The largest recorded withdrawn change was 0.017787 for `layers.10.mlp.gate_proj` in `gen3_try1`. The 8B results file records attempt-level edit snapshots, so the maximum across intermediate edits within attempts cannot be checked.
12. **CONFIRMED.** Generation 1, session 2 logged `NameError` with `name 'type' is not defined` in **turns 5 and 11**. Both turns had already consumed a hidden-state readout unit.

## Part 3. Sceptical review

- Claims 3–5 give limited support for a repeatable accuracy gain. The seed-1000 positive bootstrap interval accompanies an exact sign-test p of 0.0596, the seed-2000 estimate is negative, and the primary t and bootstrap intervals include zero. The problem-level intervals condition on these two seeds. The paired difference in edit effect between seeds, which the claims omit, is +3.75 pp with a 95% t interval of [−0.72, +8.22] pp.
- Claim 6 counts problem–seed pairs. Of the 62 problems with any changed score, 19 changed in both seeds and 10 of those changed in opposite directions. Reporting that overlap shows how much the aggregate depends on which answer was sampled.
- Claim 6 establishes that reaching the cap accompanied every change in score. The finished wrong count was at most one per arm, and only two of the 81 changed pairs involved a response flagged `looping`. Paired token-length distributions, with the mass at 4,096 shown explicitly and flags tabulated for T → C versus C → T, would help distinguish a shift in completion from a claim about repetitive text or reasoning.
- Claim 8’s +10 pp reserved result was used to select the retained edit. Five gains and one loss give p = 0.21875 on a two-sided exact test. The three practice runs varied the edit while repeating one problem, and it truncated every time. The logs provide no isolated comparison of the projection components or strength on a common paired sample.
- Claim 10’s −10 pp is a 40-problem result for one generation-2 candidate. Generation 2, session 1 has no completed check, and the inverse aggregate win/loss counts of claims 8 and 10 leave the identities of the reserved problems that flipped unknown. The JSON files contain only aggregate reserved scores.
- The claims omit a check of practice versus evaluation IDs. All six distinct practice IDs named in the four session logs were absent from the 400 evaluation IDs. The reserved check summaries omit their IDs, so the same overlap check is unavailable for those problems.
- The evaluation files support a level and subject breakdown of the paired changes. For example, the two-seed mean is +3.38 pp on level 2 (74 problems) and −0.95 pp on level 5 (105 problems). A breakdown with denominators and intervals would show where the pooled estimate comes from.
- Claim 11 compares the largest single-matrix change in recorded edit snapshots. The retained 14B maximum is 0.250211, about 13.3 times the recorded 8B maximum of 0.018746. The edit sets and chain outcomes are bundled, so this comparison alone cannot identify how edit size relates to accuracy or chain length.

Run `python3 -B audit.py` to reproduce Part 1 and `python3 -B audit.py --compare` to reproduce the bootstrap compatibility check, full-resampling intervals, 8B edit maxima, and seed contrast.
