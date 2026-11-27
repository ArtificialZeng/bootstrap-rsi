# T18: independent audit of the Qwen3-14B chain numbers

Work only with the files in this directory (`data/`, `claims_to_check.md`). Do not read anything outside it, do not run any model, do not import mlx, do not use the GPU, do not modify the files in `data/`. Use `python3` (numpy and scipy are installed). Write your report to `REPORT.md` in this directory and any script you write to `audit.py`. Write the report in English.

## What the data are

A language model (Qwen3-14B) answered 400 held-out math problems, one sampled answer per problem, with a cap of 4096 new tokens. Each `.pkl` in `data/` is a pickled list of 400 dicts, one per problem (`pid` identifies the problem; `correct` and `truncated` are the graded outcome; an answer that hit the cap is graded wrong).

- `14b_c4096_p60_e400_eval_base.pkl`: unedited model, sampling seed 1000 ("gen0@1000")
- `14b_c4096_p60_e400_eval_reseed.pkl`: unedited model, sampling seed 2000 ("gen0@2000")
- `14b_c4096_p60_e400_chain_gen1_eval_s1000.pkl`: model after one self-edit, seed 1000 ("gen1@1000")
- `14b_c4096_p60_e400_chain_gen1_eval_s2000.pkl`: same edited model, seed 2000 ("gen1@2000")
- `14b_session14b_gen{1,2}_try{1,2}.json`: the four editing sessions (fields `log` = list of turns with `code`, `output`, `units_left`; `events` = tool calls that cost a test unit; `checked` = result of the 40-problem reserved check; `decision`; `edits` = relative change per changed weight matrix).
- `8b_c4096_p60_e400_results.json`: results of an earlier chain on Qwen3-8B (key `chainB`), needed only for one comparison below.

## Part 1: recompute, without looking at `claims_to_check.md`

From the four pkl files, pairing by `pid` (check that the four files cover the same 400 pids, each once):

1. Accuracy and truncation rate of each of the four arms; the number of finished-but-wrong answers in each.
2. gen0@2000 − gen0@1000, gen1@1000 − gen0@1000, gen1@2000 − gen0@2000: difference in points, number of problems gained and lost, two-sided exact sign-test p, and a 95% bootstrap interval over problems (say how many resamples and which seed).
3. The pre-specified primary test: for each problem d_i = mean over the two seeds of (gen1 − gen0) with seeds matched (1000 with 1000, 2000 with 2000). Mean of d_i in points, 95% t interval, two-sided paired t-test p, a sign-flip permutation p, a bootstrap interval, and the counts of problems with d_i > 0 and d_i < 0.
4. For each of the two seeds, the 3x3 transition table between truncated (T), finished wrong (W) and correct (C) from gen0 to gen1. How many changed-score problems in total, and how many of them are T<->C?
5. Is there any sign that the grading fields are inconsistent (for example `correct` true together with `truncated` true)?

Write these numbers into `REPORT.md` first.

## Part 2: compare

Now open `claims_to_check.md`. For each numbered claim say CONFIRMED, WRONG (give the right value and how you got it) or CANNOT CHECK (say what is missing). For claims about the sessions, verify them against the four session JSON files and quote the turn number that supports or contradicts each one.

## Part 3: what a sceptical reviewer would say

In at most ten bullet points: which sentences in the claims overstate what these data support, and which analyses that these files allow are missing. Be specific; no general advice.
