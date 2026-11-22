# T5 对抗式实验设计审查 → 写 `handoff/codex/T5_design_critique_out.md`
读 `mech.py`、`agent8b.py`、`common.py`、`analyze.py`、`queue.sh`、`handoff/protocol_spec.md`、`handoff/review_agent8b.md`（DeepSeek 的代码审查，已修）。**不要重复代码层面的 bug**，专注于“结论是否站得住”的效度问题，例如：
- 批量生成的采样种子是 batch 级而非逐题，配对是否仍有效；
- MATH-500 可能在 Qwen3 训练集中的污染对“截断 vs 答完”结论的影响；
- 层/α 在练习题上选择、在留出集上评估是否足够防止过拟合；α=1.0（全额移除）与论文 α=0.05 的差别；
- “think efficiently” prompt 对照是否公平；
- 续写反事实（在截断处接着写 4096 token）是否等价于“2 倍预算重新生成”，偏差方向；
- reseed 噪声地板是否足以解释单次结果；
- 8B 自我改写会话的提示比论文转述更“直给”，对结论的影响；
- 任何能让怀疑者说“你的结论不成立”的点。
每条：问题 → 为什么重要 → 能否用现有数据/低成本补救（给具体做法）→ 严重度。最后列“发表前必须做”的前 3 项。
