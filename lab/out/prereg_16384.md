# Pre-specified analysis for the 16,384-token cap (written 2026-11-22 06:20 +08, before any 16,384 result existed)

State when written: `ext-q35-16384` had started at 06:09 and had saved no batch; no `*.cap16384*` file existed.

Model Qwen3.5-4B, 400 held-out MATH-500 problems, answers continued from the 8,192 files to 16,384 with the arm's weights.

**Primary test (one comparison, alpha 0.05 two-sided):** two-seed edit vs two-seed base at 16,384.
d_i = [(s8 - base) + (s8@2000 - reseed)] / 2 per problem; paired t-test over the 400 d_i, with the mean and its 95% CI.
The exact sign-flip p and the problem bootstrap CI are reported next to it as sensitivity checks.

**Secondary (reported with nominal p, labelled exploratory):**
1. concise prompt vs two-seed edit at 16,384 (does the prompt still match the edit at the paper's budget?).
2. s8 vs s8rand at 16,384 (seed 1000).
3. edit+prompt (s8concise) vs edit alone and vs prompt alone at 8,192.
4. Finished-wrong counts and W<->C transitions in every arm (is any gain not via truncation?).

The 4,096-8,192 results already seen (edit +5.1 points at 8,192, nominal p=0.0044; not significant after Holm correction
over all 60 comparisons in that table) are exploratory and are what motivated this test.

## Addendum, 2026-11-22 15:04 +08
The seed-1000 half finished at 15:02 and has been looked at: at 16,384 base 68.2%, s8 74.8%, concise 70.5%
(s8 - base +6.5 points, 69/43, p=0.018; concise - s8 -4.2, 44/61, p=0.12). The seed-2000 half (reseed, s8@2000)
had saved no 16,384 batch at this time. The primary test above is unchanged and will pool both halves; the
seed-2000 pair alone (s8@2000 - reseed) is the part of it that is still unseen and will be reported separately.
Decided after seeing the seed-1000 half (so exploratory): a second seed of the concise prompt (concise@2000) and the
16,384 extension of the edit+prompt arm, to compare prompt and edit with two seeds each at 16,384.

## Outcome, 2026-11-22 23:49 +08 (numbers from out/q35_caps.json)
Primary test, 16,384 cap, two seeds pooled: edit - base = **+4.9 points, 95% CI [+1.2, +8.6], paired t p = 0.0095**
(exact sign-flip p = 0.011; bootstrap CI [+1.1, +8.5]; 81 problems better, 62 worse). The pre-specified test passes at alpha 0.05.
By seed: seed 1000 +6.5 (69/43, p=0.018, seen before the addendum); seed 2000, the unseen half, +3.25 (49/36, p=0.19),
same direction but not significant on its own. Accuracies: base 68.2% / 70.0%, edit 74.8% / 73.3%.
Secondary: concise prompt (one seed) vs two-seed base +1.4 [-2.9, +5.7], p=0.53; concise vs two-seed edit -3.5 [-7.9, +0.9], p=0.12.
Finished-but-wrong answers at 16,384: base 0, 0; edit 2, 1; concise 0. s8rand at 16,384 was still running.

## Random-sign control at 16,384, 2026-11-23 01:31 +08 (secondary, seed 1000)
s8rand 59.75% (truncated 40.25%, no finished-wrong answers). s8rand - base = -8.5 points [-13.5, -3.5], 35/69, p=0.0011;
s8 - s8rand = +15.0 points [+9.5, +20.5], 97/37, p=2.2e-7. One draw of random signs; a second draw is queued.
