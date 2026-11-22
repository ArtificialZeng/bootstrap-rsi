# T8 GPU 吞吐优化（只分析，不跑 GPU）→ 写 `handoff/codex/T8_throughput_out.md`
瓶颈是本机唯一的 Apple M3 Max（128 GB 统一内存）上的 MLX 批量生成。当前 `common.generate` 用 `batch_generate(..., completion_batch_size=64, prefill_batch_size=16)`，上限 4096 新 token，每个条件 400 题。实测：Qwen3-1.7B bf16 约 225–320 tok/s，Qwen3-8B 8bit 小批量约 20 tok/s（与别的任务抢 GPU 时）。
请读 `../.venv/lib/python3.12/site-packages/mlx_lm/generate.py`（BatchGenerator / GenerationBatch / PromptProcessingBatch）、`mlx_lm/models/cache.py`、`mlx_lm/models/qwen3.py`、`qwen3_5.py`、`gated_delta.py`，以及各模型 config（`~/.cache/huggingface/hub/models--*/snapshots/*/config.json`：Qwen3-1.7B、Qwen3-4B、Qwen3-8B-8bit、Qwen3.5-4B-MLX-bf16）。
回答：
1. 每个模型每个序列在 4096+提示长度时的 KV/状态内存（Qwen3.5 只有 1/4 层是全注意力，线性注意力层是固定大小状态）；权重内存；在 128 GB 里留 20 GB 余量时，`completion_batch_size` 理论上限。
2. MLX 批量解码在 Apple Silicon 上何时从带宽受限变成算力受限；把 batch 从 64 提到 128/256 对各模型的预期吞吐提升（给出推理过程，标明不确定性）。
3. `batch_generate` 是否有尾部效应（最后少数长序列单独跑）；把同权重的多个条件（base / concise / reseed / 续写）合并成一次调用是否能显著提速；有无其他低风险提速开关（prefill_step_size、kv 量化等），及其对结果可比性的影响（**换 batch 是否改变采样结果分布**——这一点很重要，因为同一模型的各条件必须可比）。
4. 给出每个模型推荐的 `completion_batch_size` / `prefill_batch_size`，以及需要我做的最小 GPU 基准测试（几分钟内能跑完的）来验证。
禁止跑模型、占 GPU、改任何已有文件。
