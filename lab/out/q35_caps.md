# Qwen3.5-4B, 400 held-out MATH-500 problems, one answer per problem and seed

Arms were generated at a 4096-token cap; truncated answers were then continued from their exact token prefix to
8192 (and 16384 where shown) with the arm's weights applied. Caps in between are offline re-cuts of the same
trajectories. s8 = the paper's session-8 edit (dt_bias +0.5 on all linear-attention layers).

## Accuracy by cap

| arm | 4096 | 6144 | 8192 | 12288 | 16384 |
|---|---|---|---|---|---|
| base (seed 1000) | 0.247 | 0.365 | 0.455 | 0.583 | 0.682 |
| base (seed 2000) | 0.255 | 0.398 | 0.500 | 0.595 | 0.700 |
| s8 edit (seed 1000) | 0.258 | 0.412 | 0.510 | 0.642 | 0.748 |
| s8 edit (seed 2000) | 0.255 | 0.435 | 0.547 | 0.670 | 0.733 |
| random-sign control | 0.217 | 0.305 | 0.398 | 0.515 | 0.598 |
| concise prompt (seed 1000) | 0.383 | 0.470 | 0.535 | 0.640 | 0.705 |

## Finished-but-wrong answers (count out of 400)

| arm | 4096 | 6144 | 8192 | 12288 | 16384 |
|---|---|---|---|---|---|
| base (seed 1000) | 0 | 0 | 0 | 0 | 0 |
| base (seed 2000) | 0 | 0 | 0 | 0 | 0 |
| s8 edit (seed 1000) | 0 | 0 | 0 | 1 | 2 |
| s8 edit (seed 2000) | 0 | 0 | 0 | 0 | 1 |
| random-sign control | 0 | 0 | 0 | 0 | 0 |
| concise prompt (seed 1000) | 0 | 0 | 0 | 0 | 0 |

## Output tokens per correct answer

| arm | 4096 | 6144 | 8192 | 12288 | 16384 |
|---|---|---|---|---|---|
| base (seed 1000) | 14705 | 13897 | 13773 | 14052 | 14218 |
| base (seed 2000) | 14439 | 12776 | 12398 | 13536 | 13525 |
| s8 edit (seed 1000) | 14272 | 12193 | 12024 | 12213 | 12158 |
| s8 edit (seed 2000) | 14461 | 11593 | 11125 | 11461 | 12122 |
| random-sign control | 16942 | 17089 | 16405 | 16979 | 17681 |
| concise prompt (seed 1000) | 8193 | 9169 | 9984 | 11013 | 11892 |

## Differences in accuracy points [95% CI], problems won/lost, two-sided p (nominal, uncorrected)

| comparison | 4096 | 6144 | 8192 | 12288 | 16384 |
|---|---|---|---|---|---|
| s8 - base | +1.0 [-3.5, +5.5], 43/39, p=0.74 | +4.8 [-0.2, +9.8], 63/44, p=0.081 | +5.5 [+0.5, +10.8], 67/45, p=0.047 | +6.0 [+0.8, +11.2], 67/43, p=0.028 | +6.5 [+1.2, +11.8], 69/43, p=0.018 |
| s8@2000 - reseed | +0.0 [-4.2, +4.0], 35/35, p=1 | +3.8 [-0.8, +8.2], 50/35, p=0.13 | +4.8 [+0.0, +9.2], 53/34, p=0.053 | +7.5 [+2.8, +12.2], 63/33, p=0.0029 | +3.2 [-1.5, +7.8], 49/36, p=0.19 |
| reseed - base | +0.8 [-3.0, +4.5], 30/27, p=0.79 | +3.2 [-1.0, +7.5], 45/32, p=0.17 | +4.5 [+0.0, +9.0], 52/34, p=0.066 | +1.2 [-3.2, +5.8], 43/38, p=0.66 | +1.8 [-2.2, +6.0], 39/32, p=0.48 |
| s8rand - base | -3.0 [-6.8, +0.8], 24/36, p=0.16 | -6.0 [-10.5, -1.5], 32/56, p=0.014 | -5.8 [-10.5, -1.2], 34/57, p=0.021 | -6.8 [-11.5, -1.8], 34/61, p=0.0073 | -8.5 [-13.5, -3.5], 35/69, p=0.0011 |
| s8 - s8rand | +4.0 [-0.5, +8.2], 47/31, p=0.089 | +10.8 [+5.8, +15.8], 76/33, p=4.7e-05 | +11.2 [+6.0, +16.2], 79/34, p=2.8e-05 | +12.8 [+7.5, +18.0], 87/36, p=4.8e-06 | +15.0 [+9.5, +20.5], 97/37, p=2.2e-07 |
| concise - base | +13.5 [+9.2, +17.8], 68/14, p=1.1e-09 | +10.5 [+5.8, +15.2], 70/28, p=2.6e-05 | +8.0 [+3.2, +12.8], 65/33, p=0.0016 | +5.8 [+1.0, +10.8], 62/39, p=0.028 | +2.2 [-2.5, +7.0], 53/44, p=0.42 |
| concise - s8 | +12.5 [+8.2, +16.8], 67/17, p=3.5e-08 | +5.8 [+1.5, +10.0], 50/27, p=0.012 | +2.5 [-2.3, +7.5], 56/46, p=0.37 | -0.2 [-5.2, +4.8], 51/52, p=1 | -4.2 [-9.2, +0.8], 44/61, p=0.12 |
| concise - reseed | +12.8 [+8.5, +17.0], 67/16, p=1.4e-08 | +7.2 [+3.0, +11.5], 52/23, p=0.0011 | +3.5 [-1.2, +8.2], 51/37, p=0.17 | +4.5 [-0.2, +9.2], 56/38, p=0.079 | +0.5 [-4.2, +5.2], 47/45, p=0.92 |
| concise - s8@2000 | +12.8 [+8.5, +17.0], 66/15, p=8.6e-09 | +3.5 [-0.8, +8.0], 46/32, p=0.14 | -1.2 [-6.0, +3.2], 42/47, p=0.67 | -3.0 [-7.8, +1.8], 41/53, p=0.26 | -2.8 [-7.5, +2.0], 44/55, p=0.31 |
| s8 (2 seeds) - base (2 seeds) | +0.5 [-2.5, +3.5], +51/-47, p=0.74 | +4.2 [+0.9, +7.6], +76/-53, p=0.014 | +5.1 [+1.6, +8.6], +84/-56, p=0.0044 | +6.8 [+3.1, +10.4], +88/-54, p=0.0003 | +4.9 [+1.2, +8.6], +81/-62, p=0.0095 |
| concise - base (2 seeds) | +13.1 [+9.3, +17.0], +88/-23, p=7.4e-11 | +8.9 [+4.9, +12.8], +90/-35, p=1.2e-05 | +5.8 [+1.6, +9.9], +87/-49, p=0.0064 | +5.1 [+0.8, +9.4], +87/-51, p=0.019 | +1.4 [-2.9, +5.7], +78/-52, p=0.53 |
| concise - s8 (2 seeds) | +12.6 [+9.0, +16.3], +98/-26, p=4.4e-11 | +4.6 [+1.0, +8.2], +80/-43, p=0.012 | +0.6 [-3.5, +4.8], +81/-62, p=0.77 | -1.6 [-5.8, +2.6], +81/-67, p=0.45 | -3.5 [-7.9, +0.9], +77/-71, p=0.12 |

Wrong<->correct flips summed over every single-run comparison and cap: 4.
Single-run rows: exact sign test, bootstrap CI. Two-seed rows: per-problem mean over seeds 1000 and 2000,
paired t-test and its CI (the exact sign-flip p and bootstrap CI are in q35_caps.json as `p_flip`, `ci_boot`).
The p values are not corrected for the number of comparisons in this table; the one pre-specified test is
the two-seed edit vs base difference at 16384 (prereg_16384.md).
The same unedited model at another seed differs by several points at the larger caps (row `reseed - base`),
so single-run differences of that size are not reliable on their own.
