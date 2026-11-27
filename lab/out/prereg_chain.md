# Pre-specified analysis for the recursive self-rewrite chain (written 2026-11-23 01:43 +08, before any chain session ran)

State when written: no 8B session, no chain file and no 8B held-out answers exist (only a plumbing smoke test from 2026-11-20).

**Setup.** Qwen3-8B, MATH-500 split as before (60 practice problems the model may use, 40 reserved problems for the
framework's one check per session, 400 held-out problems the model never sees), 4096-token cap, truncated answers wrong.
Generation g is one self-editing session (agent8b.py, the paper's protocol 2) that starts from the weights kept by
generation g-1; a kept edit is written into the model and the next generation reasons with, diagnoses and edits that
model. Up to 4 generations, up to 2 sessions per generation; the chain stops when a generation keeps no edit.

**Primary test (one comparison, alpha 0.05 two-sided):** last generation that kept an edit vs generation 0 on the 400
held-out problems, seeds 1000 and 2000 for both, d_i = mean over seeds of (last - gen0), paired t-test over problems.
If no generation keeps an edit, the result is "the chain did not start" and there is no test.

**Reported with it, labelled exploratory:** held-out accuracy and truncation rate of every generation (seed 1000);
each generation vs the previous one; the T/W/C transition table (is any gain not via truncation?); every session's
decision, including withdrawn ones; the unedited model at another seed as the noise floor.

**Known weaknesses, fixed in advance:** the same 40 reserved problems check every generation, so later checks are
mildly adaptive; one chain is one sample of a stochastic process; a 4096-token cap may be the dominant lever again.
A gain that comes only from fewer truncated answers will be reported as that, not as improved problem solving.

## Addendum 1 (written 2026-11-23 10:37 +08, before any held-out answer of an edited model existed)

State when written: generation 0 is scored (held-out accuracy 0.5225 at seed 1000, 0.5375 at seed 2000; 47.5% / 46.0%
truncated). Generation 1, session 1 has ended and its held-out scoring is running; no held-out result of an edited
model has been seen. What has been seen: the session transcript and its reserved check (40 problems: 0.625 -> 0.700,
3 problems gained, 0 lost, one-sided sign p = 0.125).

**What the transcript shows.** In 14 turns the model never read a failed answer, never looked at a hidden state and
never ran a test (0 of 10 forward-pass units used). It scaled one coordinate (index 0) of layer 0's attention and MLP
outputs down by 10% several times in a row, submitted, and confirmed after the check.

**A defect in the harness.** The session's Python tool showed the model only what it printed. A tool call written as
the last line of a block (`current_edits()`, `paired_test(...)`) therefore came back as "(no output)", and the
model wrote that it was submitting "despite the lack of output". The tools were usable only through `print`, which
the instructions did not say. This is our defect, not the model's choice.

**Deviation.** The first chain ("chain A") is paused once generation 1 is scored; its generations 2-4 are not run for
now. Its generation-1 held-out result is reported as exploratory (seed 1000 only), with this reason. The pre-specified
primary test is not applied to chain A.

**Chain B (the pre-specified chain from here on).** Identical to the setup above except for one change: the tool shows
the value of a block's last expression, as a notebook does, and the instructions say so (`agent8b.py --echo
--chain-name B`). Same model, split, cap, gate, 4 generations, 2 sessions per generation, same generation-0 answers.
Session seeds differ from chain A's. **Primary test, unchanged:** last generation of chain B that kept an edit vs
generation 0 on the 400 held-out problems, seeds 1000 and 2000 for both, per-problem mean over seeds, paired t-test,
alpha 0.05 two-sided (computed by `chain_stats.py --chain-name B`). If no generation keeps an edit: "the chain did
not start", no test. Everything else as listed above, exploratory.

**Added to the weaknesses:** the 40-problem gate has no threshold and the model may keep an edit on a difference well
inside noise (as in chain A, generation 1); a kept edit is therefore not evidence of a gain, only the held-out test is.

## Addendum 2 (written 2026-11-23 16:56 +08, before any Qwen3-14B answer existed and before any held-out answer of chain B's edited model was scored)

State when written: chain B's generation-1 session has ended. It used 8 of its 10 test units (one hidden-state readout,
three paired tests on 16 practice problems, four runs), kept four edits (layers 10 and 5, relative changes 0.0009 to
0.0031) and confirmed after a reserved check of 0.625 -> 0.650 (3 gained, 2 lost). The held-out scoring of that
generation is running; no result of it has been seen. No Qwen3-14B checkpoint was on disk until today and no Qwen3-14B
answer of any kind exists.

**Why a second model.** In both 8B sessions the model's diagnosis was thin (chain A: blind edits; chain B: it read no
failed answer and did not undo edits it had itself measured as harmful). The study being replicated used a 27B model.
A chain on a larger model of the same family asks whether the outcome depends on the editor's size.

**Chain 14B (a second pre-specified chain, on another model).** `agent8b.py --model 14b --phases reseed,chain --chain 4
--attempts 2 --echo`, checkpoint mlx-community/Qwen3-14B-8bit, projections dequantized to bf16 as for 8B. Everything
else is chain B's protocol unchanged: the same 60 / 40 / 400 split, 4096-token cap, session limits, gate without a
threshold, up to 4 generations, 2 sessions per generation, held-out seeds 1000 and 2000, generation 0 scored first
with its own reseed noise floor. The instructions differ only in the model's name and its tensor shapes. Results key
`14b_c4096_p60_e400`.

**Primary test for chain 14B:** the same statistic as for chain B, applied to chain 14B alone: last generation that kept
an edit vs generation 0, 400 held-out problems, seeds 1000 and 2000 for both, per-problem mean over seeds, paired
t-test, two-sided (computed by `chain_stats.py --key 14b_c4096_p60_e400`). If no generation keeps an edit: "the chain
did not start", no test.

**Two primary tests.** Chain B and chain 14B each have one primary test. Both are reported whatever they show, each
with its uncorrected p and against a Bonferroni level of 0.025. A claim that self-editing improves a model across
generations needs, in the chain it is made for, a primary test below 0.025 and a held-out gain that exceeds that
model's reseed difference; a gain made only of fewer truncated answers is reported as that. Chain 14B does not change
chain B's test or its reading. Comparisons between the two models are exploratory.

**Order of running.** One GPU job at a time; chain 14B starts after chain B has finished or stopped. A short plumbing
run of the 14B code path (384-token cap, 6 / 3 / 4 problems, results key `14bsmoke_...`) runs first and is not data.

---

## Addendum 3 (2026-11-25 03:58 +08): two error messages reworded for chain 14B

State when written: chain B's four generations have been scored on the held-out set at seed 1000 (53.25%, 51.25%,
52.00%, 51.00% against 52.25% for generation 0). Its final seed-2000 evaluation is running and the primary test has
not been computed. No Qwen3-14B answer of any kind exists; the 14B plumbing run has not started.

**What was found.** When a session passed more than 16 ids to `run` or `paired_test`, the tool answered
"give 1..16 practice ids". Chain B's generation-4 first session read this as "ids must be the integers 1 to 16",
made 14 failed test calls (12 of them with integer ids), ran no test, submitted, and withdrew after a reserved
check of 62.5% vs 62.5%. The same message appeared once or twice in four other chain B sessions, which recovered.
The message was ambiguous; the limit itself is stated in the instructions.

**Change, for chain 14B only.** The two messages now read "this call got N ids; pass between 1 and 16 ids per call"
and "unknown practice ids: [...]; ids are the strings in practice_ids, e.g. '...'". Nothing else changes: instructions,
tools, limits, splits, seeds, gate and primary test are as in Addendum 2. Chain B is not rerun and its primary test is
unchanged; the generation-4 session above is reported as it happened.

---

## Outcome of chain B (entered 2026-11-25 07:20 +08, after the run; `chain_stats.py --chain-name B`)

All four generations kept an edit (generation 1 in its first session, generations 2 to 4 in their second).
Held-out accuracy at seed 1000: generation 0 52.25%, then 53.25%, 51.25%, 52.00%, 51.00%. Generation 0 at seed 2000:
53.75% (reseed difference +1.5, 21 gained / 15 lost). Generation 4 at seed 2000: 51.25%.

**Primary test (generation 4 vs generation 0, seeds 1000 and 2000, per-problem mean over seeds, paired t-test,
two-sided):** difference -1.875 points, 95% CI [-4.14, +0.39], p = 0.104. Sign-flip p = 0.129; bootstrap interval
[-4.125, +0.375]; 23 problems with a positive and 35 with a negative mean difference. By seed: -1.25 (14 / 19) at
seed 1000, -2.5 (17 / 27) at seed 2000.

Reading under the rules above: no improvement is claimed for chain B. The chain kept edits and the final model is
not better than generation 0; the interval does not exclude zero, so a loss is not established either.

Not part of the plan and reported as such: the queue that should have started the 14B runs after chain B could not
reach the GPU compiler (the runner had outlived an app restart) and every queued job exited within seconds, before
loading a model or generating anything. The queue was restarted at 07:10 with the same commands. No 14B answer
existed before that restart.

## Addendum 4 (written 2026-11-25 08:45 +08, before any held-out answer of Qwen3-14B existed)

The first start of chain 14B (07:27) generated the 60 practice answers of the unedited model (accuracy 68.3%,
truncated 31.7%, 60 answers in one batch) and then failed 40 minutes into the first held-out scoring pass with a GPU
out-of-memory error (`kIOGPUCommandBufferCallbackErrorOutOfMemory`). The process exited before writing any held-out
answer: no `eval_base` file, no results file and no session file exist for 14B. Scoring passes had generated 64
answers at once, the setting used for Qwen3-8B; the 14B model's weights and its per-token cache are larger.

Change, for chain 14B only: scoring passes generate 32 answers at once (`--bs 32`). This applies to every pass of the
chain alike: generation 0 at both seeds, every later generation, the practice answers and the reserved check. The
batch size changes which random draws each answer receives and nothing else: model, prompts, sampling settings, seeds,
cap, grading, splits, session limits, retention rule and the primary test are as written above. The 60 practice
answers from the failed start were generated in one batch of 60 and are set aside
(`14b_c4096_p60_e400_practice_base.bs64.pkl`); they are regenerated at the new batch size so that generation 0 and
its successors are sampled the same way. The practice accuracy of the failed start (68.3%) was seen before this
addendum; it is not an outcome of the plan and no decision above depends on it. The runner now also logs peak memory
after each pass.

The log of the failed start is kept as `chain14b.oom_fail.log`.

## Outcome of chain 14B (entered 2026-11-27 09:14 +08, after the run; `chain_stats.py --key 14b_c4096_p60_e400`)

Generation 0 (unedited Qwen3-14B, 400 held-out problems, 4096-token cap, 32 answers at once): 57.25% at seed 1000
(42.50% truncated), 59.00% at seed 2000 (41.00% truncated). Seed difference +1.75 points, bootstrap interval
[-1.50, +5.00], 26 gained / 19 lost, sign-test p = 0.371.

Generation 1: session 1 withdrew (reserved check 65.0% -> 65.0%, 2 / 2). Session 2 confirmed nine changed matrices
(relative changes 0.06 to 0.25) on a reserved check of 65.0% -> 75.0% (5 gained / 1 lost, one-sided p = 0.109).
Held-out at seed 1000: 60.50% (39.25% truncated), +3.25 points against generation 0, bootstrap interval
[+0.25, +6.50], 27 / 14, sign-test p = 0.060. At seed 2000: 58.50% (41.50% truncated), -0.50 points, interval
[-3.50, +2.75], 19 / 21, p = 0.875. All 81 changed problems over the two seeds moved between a truncated and a
correct answer.

Generation 2: session 1 withdrew without requesting the check. Session 2 withdrew after a reserved check of
75.0% -> 65.0% (1 gained / 5 lost). No edit was kept in two sessions, so the chain stopped under the rule above and
generation 1 is the last retained generation.

**Primary test (generation 1 vs generation 0, seeds 1000 and 2000, per-problem mean over seeds, paired t-test,
two-sided):** difference +1.375 points, 95% CI [-0.81, +3.56], p = 0.216. Sign-flip p = 0.262; bootstrap interval
[-0.75, +3.50]; 31 problems with a positive and 21 with a negative mean difference.

Reading under the rules above (each chain's primary test against 0.025): no improvement is claimed for chain 14B.
The difference is positive, below the seed difference of the unedited model, and its interval includes zero. The
chain retained one generation, so it gives no observation of gains accumulating across generations.
