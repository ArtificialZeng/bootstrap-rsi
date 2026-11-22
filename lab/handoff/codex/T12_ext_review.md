# T12：审查 ext.py（把各实验臂延长到更大 token 预算）并写多预算分析脚本

背景：Qwen3.5-4B 在 4096 token 上限下基线截断率 75.2%、准确率 24.7%；论文的评测上限是 16384。`../../ext.py`（即 exp/ext.py）把每个臂中被截断的回答从原 token 前缀继续生成到 8192、再到 16384，继续生成时施加该臂的权重编辑。

已有 4096 结果（400 题留出，配对同种子）：base acc 0.247 / trunc 0.752；s8（dt_bias +0.5）acc 0.258，胜/负 43/39；s8rand（随机符号 ±0.5）acc 0.217，胜/负 24/36。

请做两件事，只读 exp/ 下现有文件，不运行模型、不占 GPU、不修改现有文件、不写 out/：

1. **审查 `ext.py`**（对照 `mech.py`、`common.py`）：逐条列出正确性问题（按严重度），尤其是：
   - “前缀 + 续写 = 该臂在新上限下的一个合法样本”这一论证是否成立；续写时换种子、批大小不同、左填充、线性注意力层（Gated DeltaNet，`create_ssm_mask`）在长前缀重新预填时是否会与原生成不一致；
   - `tok.decode(out_ids)` 与原 `text` 的差异会不会影响评分（`</think>` 切分、最后一个 `\boxed{}`）；
   - 记录格式是否与 `analyze.py` 的 `budget_curve / transitions / tokens` 兼容（`ntok`、`truncated`、`correct`）；
   - 断点续跑、`--limit` 测试文件是否会污染正式缓存。
   每条给出具体的最小修补建议（代码片段）。写到 `T12_ext_review_out.md`。

2. **写 `analyze_caps.py`**（新文件，放在 handoff/codex/）：输入 `--model q35-4b --arms base,s8,s8rand,concise,reseed --caps 4096,8192,16384`，读取 `out/mech_<model>_<file>[.cap<cap>].pkl`（文件名从 `out/mech_<model>_summary.json` 的 `arm_files` 取；4096 无后缀），缺文件就跳过该格。输出 markdown + JSON：
   - 每个上限、每个臂：准确率、截断率、完成回答准确率；相对 base 的 Δ、配对 bootstrap 95% CI、胜/负、双侧与单侧符号检验 p；3×3 转移分解（经截断 / 经完成回答）；
   - s8 对 s8rand 的直接配对比较；
   - “同 token 预算”视角：每个臂平均 token、每答对一题的 token；
   - 一个“增益随预算变化”的汇总表（行=臂，列=上限，格=Δ[CI]）。
   复用 `analyze.py` 里的函数（import），并附一个不依赖真实数据的自测（合成记录）。在 `T12_ext_review_out.md` 末尾说明用法。
