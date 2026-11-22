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
