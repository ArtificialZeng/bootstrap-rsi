# T15：预算曲线图脚本 + 统计复核

注意：COMMON.md 里提到的 `queue.sh` 已换成 `qrun.sh`，规则不变（不占 GPU、不 import mlx、不改现有文件）。只写下面列出的新文件。

## A. 图脚本 `handoff/codex/fig_caps.py`
读 `out/q35_caps.json`（由 `cap_table.py` 生成；键是上限，里面有 `arms`（acc、trunc、tok_per_correct）和 `pairs`（delta、ci、p 等）），以后会多出 16384 档与新臂（如 `s8concise`、`s8@2000`），脚本要对缺失的臂/档位容错。输出纯 SVG（不依赖 matplotlib；参考 `handoff/codex/make_figs.py` 的写法与配色），到 `--out` 指定目录（默认 `handoff/codex/figs_caps/`，不要写进 `out/`）：
1. `acc_vs_cap.svg`：横轴 token 上限，纵轴准确率；线：基线（两个种子取平均，并用浅色带标出两个种子的范围）、s8 编辑（同样处理）、随机符号对照、简洁提示（以及存在时的 s8concise）。
2. `delta_vs_cap.svg`：横轴上限；纵轴相对基线的差值（百分点）及 95% 区间误差棒，三条：`s8 (2 seeds) - base (2 seeds)`、`concise - base (2 seeds)`、`s8rand - base`；画出 0 线。
3. `tokens_per_correct.svg`：8192（或最大可用档）下各臂“每答对一题的输出 token”的条形图。
图中文字用英文，字号在 800px 宽下可读；每张图 `<title>` 与一行脚注说明“answers truncated at 4096 were continued to the larger cap; intermediate caps are offline re-cuts”。写完后用 `../.venv/bin/python -B handoff/codex/fig_caps.py` 实际跑一次并确认生成了 3 个 SVG、XML 可解析。

## B. 统计复核 `handoff/codex/T15_stats_review_out.md`
独立复核 `cap_table.py` 与 `out/q35_caps.json`（不要假定它是对的）：
1. 自己写一小段独立代码，从 `out/mech_q35-4b_eval_*.pkl` 与 `*.cap8192.pkl` 重新计算 4096 与 8192 的各臂准确率、`s8 - base` 两种子合并差值与区间，核对是否与 json 一致（列出任何不一致）。pkl 里是 dict 列表，字段有 pid、correct、truncated、ntok、out_ids、text 等。
2. 审查统计方法：两个种子的逐题平均后对题目做 bootstrap / 符号翻转置换检验是否恰当？“种子”在这里的含义（4096 段：每 200 题一块，种子 seed+块号；续写段：种子 3000+cap+批号，批的构成随臂不同）是否破坏配对解释？有没有更合适的检验（例如把题目作为随机效应的混合模型、McNemar 的合并）？给出你认为应当写进论文的首选统计量，并用它重新算出 8192 时 `s8 vs base` 的 p 值与区间。
3. 多重比较：我们看了 5 个档位 × 多个比较；哪些结论在保守校正后仍成立？
4. 检查离线重裁（5120/6144/7168）是否有偏：这些档位的回答在 4096 处经历了一次“续写接缝”（重新预填、换种子），而如果直接以 8192 为上限一次生成则没有接缝——这对“某臂在某档的准确率”是否构成系统偏差？是否对各臂对称？
5. 检查答完的题 100% 正确这一点：抽 15 个 8192 档“答完”的回答，核对 `\boxed{}` 与标准答案（标准答案可经 `common.load_data()` 得到；注意 `common.py` 顶部会 import mlx——不要 import 它，自己读数据文件或从 pkl 里已有的 `correct` 字段配合 text 人工核对）。
每条结论给文件行号或你运行的代码与输出。
