## 背景（所有 Codex 任务共用）
我们在复现并检验 arXiv 2611.21885《Language Models Can Rewrite Their Own Weights》（UIUC 等，Qwen3.5-27B 不经训练改写自身权重，第二种协议 session 8 在 490 道题上 +6.6 分）。论文协议细节见 `handoff/protocol_spec.md`（已核对过 LaTeX 源）。
我们的实验（工作目录 = 当前目录 `exp/`）：
- `mech.py`：任意尺寸 Qwen3（1.7B/4B/8B，MLX，Apple M3 Max）上的“截断方向消融”机制实验：方向 u = 截断答案与答完答案的隐藏状态均值差，W ← W − α u uᵀW 作用于某层 o_proj 与 down_proj；对照组 random 方向 / “think efficiently” prompt / reseed 噪声地板；以及把截断答案续写 4096 token 的反事实。MATH-500，400 道留出题，每题 1 答案，上限 4096 token，截断算错，只看最后一个 \boxed{}。
- `agent8b.py`：Qwen3-8B 在受限 Python 会话里改写自己权重的小规模协议 2 复现（自己读失败、读隐藏状态、改、配对测试、submit→框架在 40 道预留题上检查→confirm/withdraw）。
- `analyze.py`：汇总到 `out/results.md|json`，把 Δacc 精确分解为“截断减少”与“答完题准确率变化”两项。
- 1.7B 练习集初步：准确率 0.61，截断 0.38，答完题准确率 0.984——几乎所有错误都来自截断。
- 小样扫描：层 6、α=1 使 40 道练习题准确率 0.55→0.625、截断 0.45→0.35、答完题准确率 1.00→0.962。

## 硬性规则
- GPU 正在跑实验队列（`queue.sh`）。**不要运行任何模型、不要 import mlx、不要下载模型权重、不要安装包到 `../.venv`。**
- **不要修改任何已有文件**（*.py、*.sh、out/ 下任何东西、handoff/ 下别人的文件）。只写你自己的输出文件。
- 读文件可以随便读。纯 CPU 的小型 Python（numpy/scipy 已在 `../.venv/bin/python`）可以跑。
- 结论要可核查：引用给链接或文件行号；不确定就写“不确定”，不要编造。
- 用中文写。
