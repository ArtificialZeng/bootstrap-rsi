# T11 截断解剖（只读 CPU 分析）→ 写 `handoff/codex/truncation_anatomy.py`、`handoff/codex/T11_out.md`
数据（只读，不要改、不要移动）：`out/practice_base.pkl`（Qwen3-1.7B，100 道练习题）和 `out/mech_1.7b_eval_base.part0.pkl`（同模型，200 道留出题；后续会出现 `.part1.pkl` 和其他臂，你的脚本要能对任意这类 pkl 运行）。每条记录有 `text`（含 `<think>` 推理）、`ntok`、`truncated`、`correct`、`looping`、`pid`、`level`。截断 = 输出到 4096 token 上限仍未结束。
问题：模型被截断时到底在干什么？
1. 为每条**截断**轨迹做启发式分类（给出可复现规则与阈值）：
   - **循环**：末尾大段逐字/近逐字重复（可用 zlib 压缩率、n-gram 重复率、最后 k 段的编辑距离）；
   - **反复验算**：答案已经出现过（思考里出现过某个 `\boxed{}` 或“so the answer is X”），之后在反复核查 / 换方法重算；记录“首次出现候选答案”的位置（token 比例）以及该候选是否等于标准答案（标准答案需从 `../.venv/bin/python -c` 读 `common.load_data()`；可以 import common，但**不要 import mlx 相关模块**——若 common 会触发 mlx 导入，就直接读 `~/.cache/huggingface/hub/datasets--HuggingFaceH4--MATH-500/snapshots/*/test.jsonl`）；
   - **仍在推进**：没有候选答案、没有明显重复；
   - 其他。
2. 统计各类占比（按难度 level 分层），以及“截断但思考中已出现正确候选答案”的比例——这一类意味着“只要让它停下来就能答对”。
3. 对答完的轨迹给出长度分布与 level 的关系。
4. 每类挑 1–2 个**短摘录**（各不超过 400 字符）作为例子。
5. 用这些结论回答：对这个模型，“让它更早结束”（论文那类编辑的作用）最多能挽回多少分？与“给 2 倍预算”相比如何？（后者实测结果会在 `out/mech_1.7b_cont.pkl` 出现，若尚不存在就只给前者。）
禁止：跑模型、import mlx、改任何已有文件、动 `out/` 下文件。用 `../.venv/bin/python -I` 运行你的脚本。
