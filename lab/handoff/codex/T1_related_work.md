# T1 相关工作调研 → 写 `handoff/codex/T1_related_work_out.md`
联网检索并核实（每条给论文标题、作者/机构、年份、arXiv/会议链接，并用一两句话说明与我们的关系）：
1. 推理模型“过度思考 / 不收敛 / 循环到 token 上限”的研究，以及推理长度控制方法（budget forcing / s1、长度惩罚、early exit、thinking intervention 等）。
2. 用激活/表示方向控制推理长度或终止行为（steering vector、representation engineering、方向消融 / abliteration 用于推理长度）。**重点**：有没有人已经发现“截断 vs 答完”方向或“思考长度方向”，并用权重/激活编辑缩短推理、提高定长预算下准确率——这直接关系到 2611.21885 的增益是否“新”。
3. 不训练直接改权重的经典方法（ROME/MEMIT、task arithmetic、model merging）与“模型自我修改/递归自我改进”的近期工作。
4. 对 2611.21885 本身已有的讨论、复现或批评（论文发布于 2026-11-16，可能很少，找不到就写找不到）。
最后给一段“我们的复现在文献中的定位”（3–5 句）。
