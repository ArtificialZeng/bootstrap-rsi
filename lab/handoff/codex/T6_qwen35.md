# T6 同族模型可行性 → 写 `handoff/codex/T6_qwen35_out.md`
论文用 Qwen3.5（Gated DeltaNet 线性注意力 + 全注意力混合）2B/4B/9B/27B；我们用的是 Qwen3。请核查：
1. Hugging Face 上 Qwen3.5 的小模型（≤9B）是否公开、确切仓库名、参数量、许可证；是否有 mlx-community 的 MLX 版本（bf16/8bit），各约多大。
2. 本机已装的 `../.venv/lib/python3.12/site-packages/mlx_lm`（0.32.2）是否支持 Qwen3.5 架构：在 `mlx_lm/models/` 下找对应模型文件（如 qwen3_5*.py / qwen3_next*.py / gated delta 相关），说明其模块命名（线性注意力层的 time-step bias / A_log / dt_bias 参数叫什么、o_proj/down_proj 在哪），以便把论文 session 8 的编辑（“所有线性注意力层的 time-step bias +0.5”）在本机复现。
3. 给出一个具体建议：在 M3 Max 128GB 上用哪个 Qwen3.5 尺寸、预计每次 400 题×4096 token 评测需要多久（按 1.7B≈300 tok/s、8B 约 100–150 tok/s 的实测量级估算），以及需要改 `mech.py` 的哪些地方（只写建议，不改代码）。
**不要下载任何模型。**只读网页与本地源码。
