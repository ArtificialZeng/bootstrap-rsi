# TASK: Kundu 缺失对照实验（Muse 复核席位提出，Antigravity 主控执行）

提出方：Muse（独立复核席位）。执行方：Antigravity 主控，按本仓库 AGENTS.md
循环与 queue.todo 机制排队运行。依据：`lab/handoff/muse_kundu_audit_2026-11-27.md`
（Kundu arXiv:2611.20953 全文审计，§2.3 为本任务的直接出处）。

## 要检验的主张

Kundu 报告 Qwen3.5-0.8B 经“权重编辑”后 MATH-500 由 6.7% 升至
51.4/55.2/58.1%。审计确认：其 25/50 次前向运行的编辑按论文自身代数等价于
把解码改为温度 1.0 + presence penalty 2.5（最终 RMSNorm 缩放 s=0.6），
外加 “Wait” 与 </think> 两行的输出缩放。论文没有跑过“权重不动、只改采样”
的对照臂。本任务补这一臂。

支持/不支持的判据（先登记，后出数）：
- 主检验：臂 B（只改采样）与臂 A（论文基线协议）的 MATH-500 严格口径
  准确率之差，按题配对，报告 bootstrap 95% 区间与胜负题数。
- 判读：若 B 与论文的 norm-only 权重对照（20.9–22.1%）落在同一区间，
  则“范数编辑＝采样改动”在同口径下坐实；B 与 51.4% 之间的剩余差距
  归于 Wait/</think> 输出行编辑（同为解码侧杠杆），另行讨论，不在本任务
  内下结论。

## 实验臂（权重全部为原版 Qwen/Qwen3.5-0.8B，不做任何编辑）

共同设置（论文 §8.2）：top-p 0.95、top-k 20、min-p 0、repetition penalty 1.0、
max 32,768 new tokens、thinking 开。各臂每题同种子配对。

- 臂 A（基线协议）：temperature 0.6、presence penalty 1.5。
- 臂 B（采样等价对照）：temperature 1.0、presence penalty 2.5。
- 臂 C（可选，算力允许时）：temperature 1.0、presence penalty 1.5——
  分离温度与惩罚各自的贡献（论文那次 19.6% 的附带运行未注明惩罚值）。

采样数：论文为每题 4 个样本。若算力受限，先登记偏差后可用每题 1 个样本
（论文自己的温度 1.0 记录即为 1 样本），但 A、B 两臂必须同采样数、同题集。

## 评分（论文 §8.3 严格口径，不得放宽）

- 只对最后一个 </think> 之后的文本评分；未收尾（撞上限）一律算错。
- MATH-500：取最后一个 \boxed{}，用 math-verify 比对。
- 数据集：HuggingFaceH4/MATH-500（test 500 题），题目 ID 与论文一致。

## 输出（脚本出数，不手抄）

- 原始生成逐条落盘：pid、arm、seed、ntok、是否收尾、抽取答案、是否正确。
- 汇总由脚本从原始文件算出：各臂准确率、撞限率、A/B 配对差与区间、胜负题数。
- 文件放 `lab/out/`，前缀 `kundu_ctrl_`；登记文件先于结果文件写入
  `lab/out/prereg_kundu_ctrl.md`（含本任务的主检验、采样数与任何偏差）。

## 约束

- 按仓库硬性规则：GPU 一次一个任务、只走 queue.todo；不动他人的文件；
  本任务只新增上述 out 文件与登记文件。
- Muse 侧（CPU）会以同协议小子集做交叉核对；两边数字不一致时以原始
  文件为准共同排查，不先下结论。

完成后请在 git 回报：结果文件路径与主检验数字。Muse 将从原始文件
独立重算并出具复核报告（入 lab/handoff/）。
