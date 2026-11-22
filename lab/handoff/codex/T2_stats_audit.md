# T2 统计审计 → 写 `handoff/codex/T2_stats_audit_out.md` 和 `handoff/codex/test_stats.py`
1. 独立地（不看我们实现的前提下先自己写）实现：配对准确率差 δ̂ 与 SE（论文 Eq.3，见 protocol_spec.md）、单侧符号检验（只用不一致对）、按题 bootstrap 95% CI、单侧符号翻转置换检验（论文 490 题用的那种），以及精确分解：acc=(1−t)·a_f ⇒ Δ = (t_b−t_e)·a_b + (1−t_e)(a_e−a_b)。
2. 在 `handoff/codex/test_stats.py` 里用合成数据（含边界：全对、全错、全截断、无答完、n=1、长度不一致）对比你的实现与 `agent8b.paired_stats`、`analyze.decompose`、`analyze.bootstrap_ci`（可以 import，它们 import 时不会加载模型；如果 import agent8b 会触发 mlx 导入就改为把函数源码复制进测试里对比，**不要** import mlx）。用 `../.venv/bin/python handoff/codex/test_stats.py` 跑。
3. 功效分析：400 题、每题 1 答案、基线准确率 ~0.6、不一致对比例 ~25% 时，单侧 α=0.05 下能以 80% 功效检测到的最小 Δ 是多少？若改成每题 2 答案呢？“reseed 噪声地板”应如何用于解释结果？
4. 报告：发现的任何不一致或错误（附行号）、建议。
