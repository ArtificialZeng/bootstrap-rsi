# ROADMAP.md — 顶刊突破研究路线图

> 更新日期：2026-11-27。硬件：M3 Max 128 GB 统一内存（实测可流畅原生 4-bit 运行 `Qwen3.5-27B-4bit` ~16 GB 与 `Qwen3.5-35B-A3B-4bit` ~20 GB；严禁全量反量化为 BF16）。

## 核心科学问题

现有无梯度权重自改写研究（Takehi 等、AHA、Fischetti 2611.21432、Fan 等 2611.21885）几乎全部停留在 **单步 + 截断减少（T→C）**，而在我们已完成的前两代递归链（Qwen3-8B Chain B、Qwen3-14B Chain）中暴露了严重的 **保留幻觉（Retention ≠ Improvement）**。本路线图围绕三大顶刊级问题推进：

1. **突破 `T→C` 天花板**：无梯度直接权重改写能否在未饱和高难题库（AIME 1983–2025、AMC23、OlympiadBench Math/Physics）上产生统计显著的 **`W→C`（完成且答错 → 完成且答对）** 净增益？
2. **原生架构杠杆与闭式权重编译**：
   - **MoE 路由自改写（Router Self-Rewiring）**：利用 `Qwen3.5-35B-A3B-4bit` 的 256 专家稀疏路由层（全精度/8-bit `mlp.gate`），从模型自身同一题的 `C` vs `W` 完成轨迹对中闭式更新路由权重，以零反向传播、零推理延迟开销解锁非线性专家能力。
   - **闭式零空间低秩编译（Null-Space Ridge LoRA Compilation）**：在 `Qwen3.5-27B-4bit` / `35B-A3B-4bit` 上通过 `LoRALinear` 闭式编译对比推理方向与自检脚手架（带正交零空间保护以压制 `C→W` 退化）。
   - **混合架构组合**：将 Gated DeltaNet `dt_bias`（抑制冗思与截断）与 Router / Null-Space LoRA（修复完成错误 `W→C`）跨代正交叠加。
3. **递归自进化与瓶颈因果分解（Phase Transition & Bottleneck Decomposition）**：
   - 系统分离 **编辑者智商（9B / 27B / 35B-A3B vs 强外部 Oracle 编辑者上界）** × **权重写入原语（手写 numpy vs 闭式零空间 LoRA vs MoE Router 重路由 vs 推理时 steering / 提示词）** × **选择门统计功效（无门槛 40 题噪声门 vs 严格统计门槛）**。

## 执行优先级队列（严格 ≤ 3–4 小时/单任务）

| 阶段 | 任务 | 目标与通过标准 | 状态 |
|---|---|---|---|
| **P0** | 封存 Qwen3-14B Chain 最终双种子主检验 | `chain_stats.py --key 14b_c4096_p60_e400` 出数，修正 `chain14b.tex` 事实误差 | 等待 PID 19865 收尾 |
| **P1** | `Qwen3.5-9B / 27B / 35B-A3B` 4-bit 吞吐与显存基准 | 10 分钟实测 `bs=1 / 16 / 32 / 64` 真实 `tok/s` 与峰值内存 | 准备中（下载中） |
| **P2** | 构建 `exp/v2/` 高速科研引擎 | 整机 GPU 锁、缓存复用、`LoRALinear` + MoE Router 读写接口、闭式零空间编译器、高难题库统一加载与评分器、独立统计检验 | 开发中 |
| **P3** | 高难题库基线摸底（AIME / AMC23 / OlympiadBench） | 确认 `acc|finished ∈ [40%, 75%]` 且 `pass@k >> pass@1`（存在自生成配对 `C/W` 轨迹矿脉） | 待运行 |
| **P4** | **突破 Pilot 1**：MoE Router Self-Rewiring (`35B-A3B-4bit`) | 检验 `mlp.gate` 闭式对比路由更新能否带来独立留出 `W→C` 净增益，对比 `reseed` / 随机路由 / 提示词 | 待运行 |
| **P5** | **突破 Pilot 2**：Null-Space Ridge LoRA + 自检内化 (`27B-4bit` & `35B-A3B-4bit`) | 检验零空间保护闭式低秩编译（对比轨迹 & 自检上下文内化）对 `W→C` 与跨领域（Olympiad Physics）迁移的作用 | 待运行 |
| **P6** | **突破 Pilot 3**：多代正交递归自进化链 + 强 Oracle 上界对照 | Gen 1（截断/冗思修复）→ Gen 2（残余 `W` 错误路由/子空间修复）→ Gen 3（跨代复利与饱和检验），严格统计门控制 | 待运行 |
| **P7** | 预注册全量确认实验 & 顶刊论文定稿 | 多随机种子全量留出评估、消融、理论/标度分析、图表自动生成、LaTeX 顶刊手稿 | 待运行 |
